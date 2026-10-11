#pragma once

#include "DnniInference.h"
#include "DnniReader.h"
#include "PitchFeatures.h"

#include <juce_core/juce_core.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace sv::synthesis
{
struct PitchDecoderNotes
{
    std::span<const std::size_t> frameCounts;

    std::span<const float> midiPitch;

    std::span<const std::uint8_t> vowelFrames;

    std::span<const float> requestedTilt;

    std::span<const float> requestedShift;

    std::vector<float> predictedTilt;

    std::vector<float> predictedShift;
};

class PitchDecoder
{
public:
    [[nodiscard]] juce::Result load(const DnniReader& reader, std::size_t nodeIndex);

    [[nodiscard]] juce::Result run(
        const DnniTensor& context,
        const DnniTensor& scalarContext,
        std::span<const float> noise,
        const DnniTensor& controls,
        std::span<const float> vibratoControl,
        const PitchFeatures& frontend,
        PitchDecoderNotes& notes,
        DnniTensor& output,
        const std::function<bool()>& shouldCancel = {},
        DnniRunStatistics* statistics = nullptr) const;

private:
    std::array<DnniInference, 9> networks;
    std::size_t embeddingChannels = 0;
    std::size_t frameStride = 0;
    float embeddingScale = 0.0f;
    bool loaded = false;
};
}

