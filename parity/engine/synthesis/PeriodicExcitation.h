#pragma once

#include <juce_core/juce_core.h>

#include <cstddef>
#include <span>
#include <vector>

namespace sv::synthesis
{
struct PeriodicExcitationFrame
{
    float fundamentalFrequencyHz = 0.0f;
    float shape = 1.0f;
    bool voiced = false;
};

[[nodiscard]] juce::Result renderPeriodicExcitation(std::span<const PeriodicExcitationFrame> frames,
                                                    std::size_t hopSamples,
                                                    double framePeriodSeconds,
                                                    float gain,
                                                    std::vector<float>& samples,
                                                    std::vector<double>& phaseCycles);

[[nodiscard]] juce::Result renderPeriodicModulation(std::span<const PeriodicExcitationFrame> frames,
                                                    std::span<const double> phaseCycles,
                                                    std::span<const float> frequencyRatios,
                                                    std::span<const float> parameters,
                                                    std::size_t hopSamples,
                                                    double framePeriodSeconds,
                                                    std::vector<float>& samples);
}
