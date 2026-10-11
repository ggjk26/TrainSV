#pragma once

#include "AcousticModel.h"
#include "NeuralVocoder.h"
#include "PhonemeTiming.h"
#include "PitchModel.h"
#include "SynthesisStatistics.h"

#include <juce_core/juce_core.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace sv::synthesis
{
class VoiceSynthesizer
{
public:
    static constexpr double pitchContextSeconds = 0.5;

    struct State
    {
        PitchModel::State pitch;
        AcousticModel::State acoustic;
        NeuralVocoder::State vocoder;

        [[nodiscard]] std::size_t getBytes() const noexcept;
    };

    [[nodiscard]] juce::Result load(const juce::File& voice);
    [[nodiscard]] float getFrameIntervalSeconds() const noexcept;
    [[nodiscard]] float getPitchFrameIntervalSeconds() const noexcept;
    [[nodiscard]] const std::vector<std::string>& getVocalModeNames() const noexcept;
    [[nodiscard]] juce::Result predict(std::span<const TimingSyllable> syllables, std::vector<PhonemeDuration>& output) const;
    [[nodiscard]] juce::Result predictPitch(std::span<const PitchNote> notes, std::vector<float>& midiPitch, const std::function<bool()>& shouldCancel = {}, SynthesisStatistics* statistics = nullptr, State* state = nullptr, std::span<const float> vibratoEnvelope = {}) const;
    [[nodiscard]] juce::Result render(std::span<const TimedPhoneme> phonemes, std::span<const float> logF0, NeuralVocoderOutput& output, std::uint32_t seed = 5489, const std::function<bool()>& shouldCancel = {}, SynthesisStatistics* statistics = nullptr, State* state = nullptr, std::span<const float> vocalModeWeights = {}) const;

private:
    PhonemeTiming timing;
    PitchModel pitch;
    AcousticModel acoustic;
    NeuralVocoder vocoder;
    std::vector<std::string> rapLanguages;
    std::vector<std::string> vocalModeNames;
    float frameIntervalSeconds = 0.0f;
};
}
