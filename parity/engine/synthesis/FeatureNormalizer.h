#pragma once

#include "DnniInference.h"
#include "DnniReader.h"

#include <juce_core/juce_core.h>

#include <cstddef>
#include <vector>

namespace sv::synthesis
{
class FeatureNormalizer
{
public:
    [[nodiscard]] juce::Result load(const DnniReader& reader, std::size_t nodeIndex);

    [[nodiscard]] std::size_t getChannelCount() const noexcept;

    [[nodiscard]] juce::Result normalize(const DnniTensor& input, DnniTensor& output, bool clamp = true) const;

    [[nodiscard]] juce::Result denormalize(const DnniTensor& input, DnniTensor& output) const;

private:
    [[nodiscard]] juce::Result transform(const DnniTensor& input, DnniTensor& output, bool inverse, bool clamp) const;

    std::vector<float> lower;
    std::vector<float> upper;
    float targetLower = 0.0f;
    float targetUpper = 1.0f;
};

}
