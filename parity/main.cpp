#include "synthesis/AcousticModel.h"
#include "synthesis/DnniInference.h"
#include "synthesis/DnniReader.h"
#include "synthesis/FeatureNormalizer.h"
#include "synthesis/NeuralVocoder.h"
#include "synthesis/PeriodicExcitation.h"
#include "synthesis/PhonemeTiming.h"
#include "synthesis/PitchModel.h"
#include "synthesis/VocoderFilterBank.h"
#include "synthesis/VocoderResidual.h"
#include "synthesis/VocoderSpectralHeads.h"
#include "synthesis/VoiceConfiguration.h"
#include "synthesis/VoiceDatabase.h"

#include <juce_core/juce_core.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sv::synthesis;

namespace
{
[[noreturn]] void fail(const juce::String& message)
{
    throw std::runtime_error(message.toStdString());
}

void check(const juce::Result& result, const char* context)
{
    if (result.failed())
    {
        fail(juce::String(context) + ": " + result.getErrorMessage());
    }
}

DnniTensor readNpy(const juce::File& file)
{
    juce::MemoryBlock bytes;
    if (!file.loadFileAsData(bytes) || bytes.getSize() < 10 || std::memcmp(bytes.getData(), "\x93NUMPY", 6) != 0)
    {
        fail("cannot read .npy file " + file.getFullPathName());
    }
    const auto* data = static_cast<const char*>(bytes.getData());
    const auto major = static_cast<unsigned char>(data[6]);
    const std::size_t lengthBytes = major == 1 ? 2 : 4;
    std::size_t headerLength = 0;
    for (std::size_t index = 0; index < lengthBytes; ++index)
    {
        headerLength |= static_cast<std::size_t>(static_cast<unsigned char>(data[8 + index])) << (8 * index);
    }
    const auto headerStart = 8 + lengthBytes;
    const std::string header(data + headerStart, headerLength);
    if (header.find("'<f4'") == std::string::npos || header.find("'fortran_order': False") == std::string::npos)
    {
        fail("only C-order little-endian float32 .npy files are supported: " + file.getFullPathName());
    }
    const auto open = header.find('(', header.find("'shape'"));
    const auto close = header.find(')', open);
    std::vector<std::size_t> shape;
    for (const auto& token : juce::StringArray::fromTokens(juce::String(header.substr(open + 1, close - open - 1)), ",", ""))
    {
        if (token.trim().isNotEmpty())
        {
            shape.push_back(static_cast<std::size_t>(token.trim().getLargeIntValue()));
        }
    }
    DnniTensor tensor;
    tensor.frames = shape.empty() ? 1 : shape[0];
    tensor.channels = shape.size() < 2 ? 1 : shape[1];
    tensor.values.resize(tensor.frames * tensor.channels);
    if (bytes.getSize() != headerStart + headerLength + tensor.values.size() * sizeof(float))
    {
        fail("unexpected .npy payload size in " + file.getFullPathName());
    }
    std::memcpy(tensor.values.data(), data + headerStart + headerLength, tensor.values.size() * sizeof(float));
    return tensor;
}

void writeNpy(const juce::File& file, const float* values, std::size_t frames, std::size_t channels, bool vector = false)
{
    juce::String header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + juce::String(static_cast<juce::int64>(frames)) + (vector ? juce::String(",") : ", " + juce::String(static_cast<juce::int64>(channels))) + "), }";
    while ((10 + header.length() + 1) % 64 != 0)
    {
        header += " ";
    }
    header += "\n";
    juce::MemoryOutputStream stream;
    stream.write("\x93NUMPY\x01\x00", 8);
    stream.writeShort(static_cast<short>(header.length()));
    stream.write(header.toRawUTF8(), static_cast<size_t>(header.length()));
    stream.write(values, frames * channels * sizeof(float));
    if (!file.replaceWithData(stream.getData(), stream.getDataSize()))
    {
        fail("cannot write " + file.getFullPathName());
    }
}

void writeNpy(const juce::File& file, const DnniTensor& tensor)
{
    writeNpy(file, tensor.values.data(), tensor.frames, tensor.channels);
}

void writeNpy(const juce::File& file, const std::vector<float>& values)
{
    writeNpy(file, values.data(), values.size(), 1, true);
}

std::vector<float> readVectorField(const juce::var& value)
{
    std::vector<float> output;
    if (const auto* array = value.getArray())
    {
        for (const auto& item : *array)
        {
            output.push_back(static_cast<float>(static_cast<double>(item)));
        }
    }
    return output;
}

struct Voice
{
    DnniReader duration;
    DnniReader pitch;
    DnniReader acoustic;
    DnniReader vocoder;
    std::optional<VoiceInfo> info;
    std::vector<std::string> vocalModes;
};

void readDatabaseModel(VoiceDatabase& database, const juce::MemoryBlock& key, DnniReader& reader, const char* role)
{
    for (const auto& entry : database.getEntries())
    {
        if (entry.key == key)
        {
            juce::MemoryBlock bytes;
            check(database.readEntry(entry, bytes), role);
            check(reader.load(std::move(bytes)), role);
            return;
        }
    }
    fail(juce::String(role) + ": model entry missing");
}

Voice loadVoice(const juce::File& file)
{
    Voice voice;
    VoiceDatabase database;
    check(database.open(file), "voice");
    for (const auto& style : database.getMetadata().timbreStyles)
    {
        voice.vocalModes.push_back(style.toStdString());
    }
    voice.info = database.getInfo();
    if (voice.info.has_value())
    {
        check(voice.duration.load(voice.info->durationModel), "duration");
        check(voice.pitch.load(voice.info->pitchModel), "pitch");
        check(voice.acoustic.load(voice.info->acousticModel), "acoustic");
        check(voice.vocoder.load(voice.info->vocoderModel), "vocoder");
        return voice;
    }
    VoiceConfiguration configuration;
    check(configuration.load(database), "configuration");
    readDatabaseModel(database, configuration.getDurationModel().key, voice.duration, "duration");
    readDatabaseModel(database, configuration.getAcousticModel().key, voice.acoustic, "acoustic");
    readDatabaseModel(database, configuration.getVocoderModel().key, voice.vocoder, "vocoder");
    const auto* pitchEntry = database.findEntry("f0model-dds");
    if (pitchEntry == nullptr)
    {
        fail("voice has no f0model-dds");
    }
    juce::MemoryBlock bytes;
    check(database.readEntry(*pitchEntry, bytes), "pitch");
    check(voice.pitch.load(std::move(bytes)), "pitch");
    return voice;
}

void applyModelOverrides(Voice& voice, const juce::var& request)
{
    for (const auto& [role, reader] : {std::pair<const char*, DnniReader*>{"duration", &voice.duration}, {"pitch", &voice.pitch}, {"acoustic", &voice.acoustic}, {"vocoder", &voice.vocoder}})
    {
        const auto path = request["models"][role].toString();
        if (path.isNotEmpty())
        {
            check(reader->load(juce::File(path)), role);
        }
    }
}

PhonemeTiming loadTiming(const Voice& voice)
{
    PhonemeTiming timing;
    check(timing.load(voice.duration), "duration model");
    if (voice.info.has_value())
    {
        check(timing.setVoiceEmbedding(voice.info->durationVoice), "duration voice");
    }
    return timing;
}

PitchModel loadPitch(const Voice& voice)
{
    PitchModel model;
    check(model.load(voice.pitch), "pitch model");
    if (voice.info.has_value())
    {
        check(model.setSpeaker(voice.info->pitchSpeaker), "pitch speaker");
    }
    return model;
}

AcousticModel loadAcoustic(const Voice& voice)
{
    AcousticModel model;
    check(model.load(voice.acoustic, voice.info.has_value() ? std::span<const std::string>{} : std::span<const std::string>{voice.vocalModes}), "acoustic model");
    if (voice.info.has_value())
    {
        check(model.setVoice(voice.info->acousticSpeaker, voice.info->defaultStyle, voice.info->vocalModeStyles), "acoustic voice");
    }
    return model;
}

std::vector<TimingSyllable> readSyllables(const juce::var& value)
{
    std::vector<TimingSyllable> syllables;
    for (const auto& item : *value.getArray())
    {
        TimingSyllable syllable;
        syllable.language = item["language"].toString().toStdString();
        for (const auto& phoneme : *item["phonemes"].getArray())
        {
            syllable.phonemes.push_back(phoneme.toString().toStdString());
        }
        syllable.durationSeconds = static_cast<double>(item["duration"]);
        syllable.midiPitch = static_cast<int>(item["pitch"]);
        syllable.isContinuation = static_cast<bool>(item.getProperty("continuation", false));
        syllables.push_back(std::move(syllable));
    }
    return syllables;
}

std::vector<PitchNote> readPitchNotes(const juce::var& value)
{
    std::vector<PitchNote> notes;
    const auto syllables = readSyllables(value);
    for (std::size_t index = 0; index < syllables.size(); ++index)
    {
        const auto& item = (*value.getArray())[static_cast<int>(index)];
        PitchNote note;
        note.syllable = syllables[index];
        note.isSilence = static_cast<bool>(item.getProperty("silence", false));
        note.isBreath = static_cast<bool>(item.getProperty("breath", false));
        note.isContinuation = static_cast<bool>(item.getProperty("continuation", false));
        note.isRap = static_cast<bool>(item.getProperty("rap", false));
        note.tone = static_cast<int>(item.getProperty("tone", 0));
        note.vibratoModulation = static_cast<float>(static_cast<double>(item.getProperty("vibrato", 1.0)));
        notes.push_back(std::move(note));
    }
    return notes;
}

std::vector<PhonemeDuration> readPhonemeDurations(const juce::var& value)
{
    std::vector<PhonemeDuration> phonemes;
    for (const auto& item : *value.getArray())
    {
        phonemes.push_back({item["language"].toString().toStdString(), item["symbol"].toString().toStdString(), static_cast<std::size_t>(static_cast<int>(item["syllable"])), static_cast<std::size_t>(static_cast<int>(item.getProperty("interval", 0))), static_cast<double>(item["seconds"])});
    }
    return phonemes;
}

std::vector<TimedPhoneme> readTimedPhonemes(const juce::var& value)
{
    std::vector<TimedPhoneme> phonemes;
    for (const auto& item : *value.getArray())
    {
        phonemes.push_back({item["language"].toString().toStdString(), item["symbol"].toString().toStdString(), static_cast<std::size_t>(static_cast<int>(item["frames"]))});
    }
    return phonemes;
}

const DnniReader& modelReader(const Voice& voice, const juce::String& role)
{
    if (role == "duration") return voice.duration;
    if (role == "pitch") return voice.pitch;
    if (role == "acoustic") return voice.acoustic;
    if (role == "vocoder") return voice.vocoder;
    fail("unknown model role " + role);
}

void writeJson(const juce::File& file, const juce::var& value)
{
    if (!file.replaceWithText(juce::JSON::toString(value, true)))
    {
        fail("cannot write " + file.getFullPathName());
    }
}

void runDnni(const Voice& voice, const juce::var& request, const juce::File& directory, const juce::File& base)
{
    const auto& reader = modelReader(voice, request["model"].toString());
    DnniInference network;
    check(network.load(reader, static_cast<std::size_t>(static_cast<int>(request["node"]))), "network");
    const auto input = readNpy(base.getChildFile(request["input"].toString()));
    DnniTensor condition;
    const auto conditionPath = request["condition"].toString();
    if (conditionPath.isNotEmpty())
    {
        condition = readNpy(base.getChildFile(conditionPath));
    }
    DnniTensor output;
    check(network.run(input, output, conditionPath.isNotEmpty() ? &condition : nullptr), "network run");
    writeNpy(directory.getChildFile("output.npy"), output);
}

void runDuration(const Voice& voice, const juce::var& request, const juce::File& directory)
{
    const auto timing = loadTiming(voice);
    const auto syllables = readSyllables(request["syllables"]);
    std::vector<PhonemeDuration> durations;
    check(timing.predict(syllables, durations), "duration predict");
    juce::Array<juce::var> items;
    for (const auto& duration : durations)
    {
        auto* object = new juce::DynamicObject();
        object->setProperty("language", juce::String(duration.language));
        object->setProperty("symbol", juce::String(duration.symbol));
        object->setProperty("syllable", static_cast<int>(duration.syllableIndex));
        object->setProperty("interval", static_cast<int>(duration.timingIntervalIndex));
        object->setProperty("seconds", duration.durationSeconds);
        items.add(juce::var(object));
    }
    writeJson(directory.getChildFile("durations.json"), items);
}

void runPitch(const Voice& voice, const juce::var& request, const juce::File& directory)
{
    const auto model = loadPitch(voice);
    const auto notes = readPitchNotes(request["notes"]);
    const auto phonemes = readPhonemeDurations(request["phonemes"]);
    const auto envelope = readVectorField(request["vibratoEnvelope"]);
    std::vector<float> midi;
    check(model.run(notes, phonemes, midi, static_cast<std::uint32_t>(static_cast<int>(request.getProperty("seed", 0))), nullptr, {}, nullptr, envelope), "pitch run");
    writeNpy(directory.getChildFile("midi.npy"), midi);
}

void runAcoustic(const Voice& voice, const juce::var& request, const juce::File& directory, const juce::File& base)
{
    const auto model = loadAcoustic(voice);
    const auto phonemes = readTimedPhonemes(request["phonemes"]);
    const auto logF0 = readNpy(base.getChildFile(request["logF0"].toString())).values;
    std::vector<float> weights;
    if (request["vocalModeWeights"].toString().isNotEmpty())
    {
        weights = readNpy(base.getChildFile(request["vocalModeWeights"].toString())).values;
    }
    DnniTensor frames;
    check(model.run(phonemes, logF0, frames, static_cast<std::uint32_t>(static_cast<juce::int64>(request.getProperty("seed", 5489))), nullptr, {}, nullptr, weights), "acoustic run");
    writeNpy(directory.getChildFile("frames.npy"), frames);
}

void runVocoder(const Voice& voice, const juce::var& request, const juce::File& directory, const juce::File& base)
{
    const auto& reader = voice.vocoder;
    const auto& nodes = reader.getNodes();
    const auto& root = nodes[0];
    const auto rootPayload = reader.getPayload(0);
    const auto& configuration = nodes[root.children[0]];
    const auto configurationBytes = reader.getPayload(root.children[0]);
    const auto word = [](std::span<const std::uint8_t> bytes, std::size_t offset)
    {
        std::uint32_t value = 0;
        std::memcpy(&value, bytes.data() + offset, 4);
        return value;
    };
    const bool forceVoiced = configuration.type == "_vocfv2" && word(configurationBytes, 48) != 0;
    const double sampleRate = static_cast<double>(word(configurationBytes, 8));
    const double framePeriod = static_cast<double>(std::bit_cast<float>(word(configurationBytes, 12)));
    const auto hop = static_cast<std::size_t>(std::round(sampleRate * framePeriod));
    const float gain = std::exp(std::bit_cast<float>(word(rootPayload, 4)));
    const float bias = std::bit_cast<float>(word(rootPayload, 14));

    const auto input = readNpy(base.getChildFile(request["frames"].toString()));
    DnniTensor logarithmic = input;
    for (std::size_t frame = 0; frame < input.frames; ++frame)
    {
        if (forceVoiced)
        {
            logarithmic.values[frame * 72] = 1.0f;
        }
        logarithmic.values[frame * 72 + 1] = std::log(input.values[frame * 72 + 1]);
    }
    FeatureNormalizer normalizer;
    check(normalizer.load(reader, configuration.children[0]), "vocoder normalizer");
    DnniTensor normalized;
    check(normalizer.normalize(logarithmic, normalized, true), "vocoder normalize");
    writeNpy(directory.getChildFile("normalized.npy"), normalized);

    const auto runNetwork = [&reader](std::size_t node, const DnniTensor& tensor)
    {
        DnniInference network;
        check(network.load(reader, node), "vocoder network");
        DnniTensor result;
        check(network.run(tensor, result), "vocoder network run");
        return result;
    };
    const auto shape = runNetwork(root.children[1], normalized);
    writeNpy(directory.getChildFile("shape.npy"), shape);
    std::vector<PeriodicExcitationFrame> sourceFrames(input.frames);
    for (std::size_t frame = 0; frame < input.frames; ++frame)
    {
        const float sourceShape = 2.9f / (std::exp(-shape.values[frame]) + 1.0f) + 0.1f;
        sourceFrames[frame] = {input.values[frame * 72 + 1], sourceShape, forceVoiced || input.values[frame * 72] >= 0.5f};
    }
    const auto condition = runNetwork(root.children[2], normalized);
    writeNpy(directory.getChildFile("condition.npy"), condition);
    VocoderSpectralHeads heads;
    check(heads.load(reader, root.children[3]), "spectral heads");
    std::array<DnniTensor, 3> spectra;
    check(heads.run(condition, spectra), "spectral heads run");
    for (std::size_t index = 0; index < 3; ++index)
    {
        writeNpy(directory.getChildFile("spectrum" + juce::String(static_cast<int>(index)) + ".npy"), spectra[index]);
    }
    std::vector<float> ratios;
    check(reader.readFloatVector(root.children[4], ratios), "modulation ratios");
    const auto modulationParameters = runNetwork(root.children[5], condition);
    writeNpy(directory.getChildFile("modulation_parameters.npy"), modulationParameters);

    std::vector<float> excitation;
    std::vector<double> phases;
    check(renderPeriodicExcitation(sourceFrames, hop, framePeriod, gain, excitation, phases), "excitation");
    writeNpy(directory.getChildFile("excitation.npy"), excitation);
    std::vector<float> modulation;
    check(renderPeriodicModulation(sourceFrames, phases, ratios, modulationParameters.values, hop, framePeriod, modulation), "modulation");
    writeNpy(directory.getChildFile("modulation.npy"), modulation);
    for (std::size_t sample = 0; sample < modulation.size(); ++sample)
    {
        modulation[sample] = excitation[sample] * (modulation[sample] + bias);
    }
    VocoderResidual residual;
    check(residual.load(reader, 0), "residual");
    std::vector<float> residualSignal;
    check(residual.render(normalized, excitation, residualSignal), "residual render");
    writeNpy(directory.getChildFile("residual.npy"), residualSignal);
    std::array<std::vector<float>, 3> components;
    const std::array<const std::vector<float>*, 3> sources{&excitation, &residualSignal, &modulation};
    for (std::size_t index = 0; index < 3; ++index)
    {
        VocoderFilterBank filter;
        check(filter.load(reader, root.children[6 + index], hop), "filter");
        check(filter.render(spectra[index], *sources[index], components[index]), "filter render");
        writeNpy(directory.getChildFile("component" + juce::String(static_cast<int>(index)) + ".npy"), components[index]);
    }

    NeuralVocoder vocoder;
    check(vocoder.load(reader), "vocoder");
    NeuralVocoderOutput output;
    check(vocoder.run(input, output), "vocoder run");
    writeNpy(directory.getChildFile("samples.npy"), output.samples);
    std::vector<float> rebuilt(output.samples.size());
    for (std::size_t sample = 0; sample < rebuilt.size(); ++sample)
    {
        rebuilt[sample] = components[0][sample] + components[1][sample] + components[2][sample];
    }
    writeNpy(directory.getChildFile("rebuilt.npy"), rebuilt);
}

void runSynthesize(const Voice& voice, const juce::var& request, const juce::File& directory, const juce::File& base)
{
    const auto acoustic = loadAcoustic(voice);
    const auto phonemes = readTimedPhonemes(request["phonemes"]);
    const auto logF0 = readNpy(base.getChildFile(request["logF0"].toString())).values;
    DnniTensor frames;
    check(acoustic.run(phonemes, logF0, frames, static_cast<std::uint32_t>(static_cast<juce::int64>(request.getProperty("seed", 5489)))), "acoustic run");
    NeuralVocoder vocoder;
    check(vocoder.load(voice.vocoder), "vocoder");
    NeuralVocoderOutput output;
    check(vocoder.run(frames, output), "vocoder run");
    writeNpy(directory.getChildFile("frames.npy"), frames);
    writeNpy(directory.getChildFile("samples.npy"), output.samples);
}
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::fputs("Usage: opensv-parity <request.json> <output-directory>\n", stderr);
        return 1;
    }
    try
    {
        const auto requestFile = juce::File::getCurrentWorkingDirectory().getChildFile(argv[1]);
        const auto directory = juce::File::getCurrentWorkingDirectory().getChildFile(argv[2]);
        if (!directory.createDirectory())
        {
            fail("cannot create " + directory.getFullPathName());
        }
        const auto request = juce::JSON::parse(requestFile);
        const auto base = requestFile.getParentDirectory();
        auto voice = loadVoice(base.getChildFile(request["voice"].toString()));
        applyModelOverrides(voice, request);
        const auto command = request["command"].toString();
        if (command == "dnni") runDnni(voice, request, directory, base);
        else if (command == "duration") runDuration(voice, request, directory);
        else if (command == "pitch") runPitch(voice, request, directory);
        else if (command == "acoustic") runAcoustic(voice, request, directory, base);
        else if (command == "vocoder") runVocoder(voice, request, directory, base);
        else if (command == "synthesize") runSynthesize(voice, request, directory, base);
        else fail("unknown command " + command);
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::fprintf(stderr, "opensv-parity: %s\n", exception.what());
        return 2;
    }
}
