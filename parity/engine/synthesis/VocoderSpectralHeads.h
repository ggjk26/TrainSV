#pragma once

#include "DnniInference.h"

#include <juce_core/juce_core.h>

#include <array>
#include <cstddef>
#include <functional>
#include <vector>

namespace sv::synthesis
{
class VocoderSpectralHeads
{
public:
    struct State
    {
        std::vector<DnniInference::Cache> bands;

        [[nodiscard]] std::size_t getBytes() const noexcept;
    };

    [[nodiscard]] juce::Result load(const DnniReader& reader, std::size_t nodeIndex);
    [[nodiscard]] juce::Result run(const DnniTensor& input, std::array<DnniTensor, 3>& outputs, State* state = nullptr, const std::function<bool()>& shouldCancel = {}, DnniRunStatistics* statistics = nullptr) const;

private:
    struct Band
    {
        std::size_t firstBin = 0;
        std::size_t binCount = 0;
        std::size_t fadeIn = 0;
        std::size_t fadeOut = 0;
        DnniInference network;
    };

    std::vector<Band> bands;
    std::size_t outputBins = 0;
};
}
