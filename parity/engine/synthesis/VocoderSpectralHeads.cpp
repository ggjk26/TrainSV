#include "VocoderSpectralHeads.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <new>
#include <span>
#include <utility>

namespace sv::synthesis
{
namespace
{
constexpr std::size_t maximumElements = 64 * 1024 * 1024;
std::uint32_t readUint32(std::span<const std::uint8_t> payload, std::size_t offset)
{
    return static_cast<std::uint32_t>(payload[offset]) | (static_cast<std::uint32_t>(payload[offset + 1]) << 8) | (static_cast<std::uint32_t>(payload[offset + 2]) << 16) | (static_cast<std::uint32_t>(payload[offset + 3]) << 24);
}

juce::Result modelError(const juce::String& reason)
{
    return juce::Result::fail("Vocoder spectral heads: " + reason);
}
}

juce::Result VocoderSpectralHeads::load(const DnniReader& reader, std::size_t nodeIndex)
{
    const auto& nodes = reader.getNodes();
    if (nodeIndex >= nodes.size() || nodes[nodeIndex].type != "_cepsg1")
    {
        return modelError("the selected node is not the verified spectral-head operator.");
    }
    const auto& node = nodes[nodeIndex];
    const auto payload = reader.getPayload(nodeIndex);
    if (payload.size() < 8 || readUint32(payload, 0) != 3)
    {
        return modelError("the operator must declare three output spectra.");
    }
    const auto bandCount = static_cast<std::size_t>(readUint32(payload, 4));
    if (bandCount == 0 || bandCount > (payload.size() - 8) / 16 || payload.size() != 8 + bandCount * 16)
    {
        return modelError("the band count does not match the serialized band descriptors.");
    }
    if (node.children.size() != 1 || node.children.front() >= nodes.size())
    {
        return modelError("the operator requires one group of band networks.");
    }
    const auto& group = nodes[node.children.front()];
    if (group.type != "cmpg1" || !reader.getPayload(node.children.front()).empty() || group.children.size() != bandCount)
    {
        return modelError("the network group does not match the declared bands.");
    }

    try
    {
        std::vector<Band> replacement;
        replacement.reserve(bandCount);
        std::size_t replacementBins = 0;
        for (std::size_t index = 0; index < bandCount; ++index)
        {
            const auto offset = 8 + index * 16;
            Band band;
            band.firstBin = readUint32(payload, offset);
            const auto endBin = static_cast<std::size_t>(readUint32(payload, offset + 4));
            band.fadeIn = readUint32(payload, offset + 8);
            band.fadeOut = readUint32(payload, offset + 12);
            if (band.firstBin >= endBin || endBin > maximumElements / 3)
            {
                return modelError("a band has an invalid bin range.");
            }
            band.binCount = endBin - band.firstBin;
            if (band.fadeIn == 1 || band.fadeOut == 1 || band.fadeIn > band.binCount || band.fadeOut > band.binCount - band.fadeIn)
            {
                return modelError("band fades must be empty or contain at least two bins, and must not overlap.");
            }
            const auto networkIndex = group.children[index];
            if (networkIndex >= nodes.size() || nodes[networkIndex].type != "modm0")
            {
                return modelError("each band requires a sequential network.");
            }
            if (const auto result = band.network.load(reader, networkIndex); result.failed())
            {
                return result;
            }
            replacementBins = std::max(replacementBins, endBin);
            replacement.push_back(std::move(band));
        }
        bands = std::move(replacement);
        outputBins = replacementBins;
        return juce::Result::ok();
    }
    catch (const std::bad_alloc&)
    {
        return modelError("insufficient memory to load the networks.");
    }
}

std::size_t VocoderSpectralHeads::State::getBytes() const noexcept
{
    std::size_t bytes = bands.capacity() * sizeof(DnniInference::Cache);
    for (const auto& band : bands)
    {
        bytes += band.getBytes();
    }
    return bytes;
}

juce::Result VocoderSpectralHeads::run(const DnniTensor& input, std::array<DnniTensor, 3>& outputs, State* state, const std::function<bool()>& shouldCancel, DnniRunStatistics* statistics) const
{
    if (shouldCancel && shouldCancel())
    {
        return juce::Result::fail("Cancelled");
    }
    if (bands.empty())
    {
        return modelError("no model has been loaded.");
    }
    if (input.frames > maximumElements / outputBins)
    {
        return modelError("assembled spectra exceed the 64 Mi element limit per output.");
    }
    try
    {
        if (state != nullptr)
        {
            state->bands.resize(bands.size());
        }
        std::array<DnniTensor, 3> replacement;
        for (auto& output : replacement)
        {
            output.frames = input.frames;
            output.channels = outputBins;
            output.values.assign(input.frames * outputBins, 0.0f);
        }
        for (std::size_t bandIndex = 0; bandIndex < bands.size(); ++bandIndex)
        {
            const auto& band = bands[bandIndex];
            DnniTensor predicted;
            if (const auto result = band.network.run(input, predicted, nullptr, state != nullptr ? &state->bands[bandIndex] : nullptr, shouldCancel, statistics); result.failed())
            {
                return result;
            }
            if (predicted.frames != input.frames || predicted.channels != 3 * band.binCount)
            {
                return modelError("a band network must preserve frame count and produce three consecutive spectra of its declared width.");
            }
            for (std::size_t frame = 0; frame < input.frames; ++frame)
            {
                for (std::size_t spectrum = 0; spectrum < replacement.size(); ++spectrum)
                {
                    const auto sourceOffset = frame * predicted.channels + spectrum * band.binCount;
                    const auto destinationOffset = frame * outputBins + band.firstBin;
                    for (std::size_t bin = 0; bin < band.binCount; ++bin)
                    {
                        auto value = predicted.values[sourceOffset + bin];
                        if (bin < band.fadeIn)
                        {
                            value = value * static_cast<float>(bin) / static_cast<float>(band.fadeIn - 1);
                        }
                        else if (bin >= band.binCount - band.fadeOut)
                        {
                            value = value * static_cast<float>(band.binCount - 1 - bin) / static_cast<float>(band.fadeOut - 1);
                        }
                        auto& destination = replacement[spectrum].values[destinationOffset + bin];
                        destination += value;
                        if (!std::isfinite(destination))
                        {
                            return modelError("band accumulation produced a non-finite value.");
                        }
                    }
                }
            }
        }
        outputs = std::move(replacement);
        return juce::Result::ok();
    }
    catch (const std::bad_alloc&)
    {
        return modelError("insufficient memory for spectral inference.");
    }
}
}
