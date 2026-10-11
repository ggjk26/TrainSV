#include "LipRadiation.h"

#include <cmath>
#include <functional>
#include <limits>
#include <numbers>

namespace sv::synthesis
{
namespace
{
constexpr double resistance = 128.0 / (9.0 * std::numbers::pi * std::numbers::pi);
constexpr double inductancePerCentimetre = 8.0 / (100.0 * 3.0 * std::numbers::pi * 340.0);

bool isPositiveFinite(double value) noexcept
{
    return std::isfinite(value) && value > 0.0;
}

bool fitsFloat(double value) noexcept
{
    return std::isfinite(value) && std::abs(value) <= static_cast<double>(std::numeric_limits<float>::max());
}
}

std::optional<LipRadiationResponse> getLipRadiationResponse(double frequencyHz, double radiusCm) noexcept
{
    if (!isPositiveFinite(frequencyHz) || !isPositiveFinite(radiusCm))
    {
        return std::nullopt;
    }

    const double reactance = frequencyHz * radiusCm * (2.0 * std::numbers::pi * inductancePerCentimetre);
    if (!isPositiveFinite(reactance))
    {
        return std::nullopt;
    }

    const double amplitudeGain = resistance * (reactance / std::hypot(resistance, reactance));
    if (!isPositiveFinite(amplitudeGain))
    {
        return std::nullopt;
    }

    return LipRadiationResponse{amplitudeGain, std::atan2(resistance, reactance)};
}

juce::Result applyLipRadiationToHarmonics(std::span<float> amplitudes,
                                          std::span<float> phasesRadians,
                                          double fundamentalFrequencyHz,
                                          double radiusCm,
                                          bool inverse)
{
    if (amplitudes.size() != phasesRadians.size())
    {
        return juce::Result::fail("Harmonic amplitudes and phases must have the same length.");
    }

    const std::less<const float*> precedes;
    if (!amplitudes.empty() && precedes(amplitudes.data(), phasesRadians.data() + phasesRadians.size()) && precedes(phasesRadians.data(), amplitudes.data() + amplitudes.size()))
    {
        return juce::Result::fail("Harmonic amplitude and phase arrays must not overlap.");
    }

    if (!getLipRadiationResponse(fundamentalFrequencyHz, radiusCm))
    {
        return juce::Result::fail("Fundamental frequency and lip radius must be finite, positive, and within the numeric range.");
    }

    for (std::size_t index = 0; index < amplitudes.size(); ++index)
    {
        if (!std::isfinite(amplitudes[index]) || amplitudes[index] < 0.0f || !std::isfinite(phasesRadians[index]))
        {
            return juce::Result::fail("Harmonic amplitudes must be finite and non-negative, and phases must be finite.");
        }

        const auto response = getLipRadiationResponse(fundamentalFrequencyHz * static_cast<double>(index + 1), radiusCm);
        if (!response)
        {
            return juce::Result::fail("Harmonic frequency and lip radius exceed the numeric range.");
        }

        const double amplitude = inverse ? amplitudes[index] / response->amplitudeGain : amplitudes[index] * response->amplitudeGain;
        const double phase = phasesRadians[index] + (inverse ? -response->phaseShiftRadians : response->phaseShiftRadians);
        if (!fitsFloat(amplitude) || !fitsFloat(phase))
        {
            return juce::Result::fail("Lip radiation would produce a value outside the float range.");
        }
    }

    for (std::size_t index = 0; index < amplitudes.size(); ++index)
    {
        const auto response = getLipRadiationResponse(fundamentalFrequencyHz * static_cast<double>(index + 1), radiusCm);
        amplitudes[index] = static_cast<float>(inverse ? amplitudes[index] / response->amplitudeGain : amplitudes[index] * response->amplitudeGain);
        phasesRadians[index] = static_cast<float>(phasesRadians[index] + (inverse ? -response->phaseShiftRadians : response->phaseShiftRadians));
    }

    return juce::Result::ok();
}
}
