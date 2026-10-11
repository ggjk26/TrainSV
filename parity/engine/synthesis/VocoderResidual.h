#pragma once

#include "DnniInference.h"

#include <juce_core/juce_core.h>

#include <cstddef>
#include <functional>
#include <span>
#include <vector>

namespace sv::synthesis
{
class VocoderResidual
{
public:
    struct State
    {
        DnniInference::Cache network;
        DnniInference::Cache projection;

        [[nodiscard]] std::size_t getBytes() const noexcept;
    };

    [[nodiscard]] juce::Result load(const DnniReader& reader, std::size_t vocoderRoot = 0);
    [[nodiscard]] juce::Result render(const DnniTensor& normalizedFeatures, std::span<const float> excitation, std::vector<float>& output, State* state = nullptr, const std::function<bool()>& shouldCancel = {}, DnniRunStatistics* statistics = nullptr) const;

private:
    DnniInference network;
    DnniInference projection;
    std::vector<std::vector<float>> analysisFilters;
    std::vector<std::vector<float>> synthesisFilters;
    std::size_t hopSamples = 0;
    std::size_t bands = 0;
    std::size_t featureChannels = 0;
    bool gateNoise = false;
    bool loaded = false;
};
}
