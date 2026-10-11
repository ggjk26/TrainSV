#pragma once

#include <juce_core/juce_core.h>

#include <optional>
#include <span>

namespace sv::synthesis
{
struct LipRadiationResponse
{
    double amplitudeGain = 0.0;
    double phaseShiftRadians = 0.0;
};

[[nodiscard]] std::optional<LipRadiationResponse> getLipRadiationResponse(double frequencyHz, double radiusCm) noexcept;

[[nodiscard]] juce::Result applyLipRadiationToHarmonics(std::span<float> amplitudes,
                                                        std::span<float> phasesRadians,
                                                        double fundamentalFrequencyHz,
                                                        double radiusCm,
                                                        bool inverse = false);
}
