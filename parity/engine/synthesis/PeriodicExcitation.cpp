#include "PeriodicExcitation.h"

#include <juce_dsp/juce_dsp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <utility>

namespace sv::synthesis
{
namespace
{
constexpr double twoPi = 2.0 * std::numbers::pi;
constexpr double referencePeriodSeconds = 0.005;
constexpr std::size_t parameterKnotCount = 256;
constexpr std::size_t maximumFftSize = 1024 * 1024;

struct LfParameters
{
    double closingTime = 0.0;
    double peakTime = 0.0;
    double returnTime = 0.0;
    double decay = 0.0;
    double growth = 0.0;
};

struct LfKnot
{
    float shape = 0.0f;
    float closingRatio = 0.0f;
    float peakRatio = 0.0f;
    float returnRatio = 0.0f;
    float logDecay = 0.0f;
    float logGrowth = 0.0f;
};

LfKnot makeKnot(std::size_t index)
{
    const double rd = 0.01 * std::pow(600.0, static_cast<double>(index) / 255.0);
    const float shape = static_cast<float>(rd);
    const double returnRatio = rd < 0.21 ? 0.000001 : rd < 2.7 ? (4.8 * rd - 1.0) / 100.0
                                                               : 0.323 / rd;
    double closingRatio = 0.0;
    double peakRatio = 0.0;
    if (index < 209)
    {
        const double rk = (22.4 + 11.8 * rd) / 100.0;
        const double rg = 0.25 * rk / (0.11 * rd / (0.5 + 1.2 * rk) - returnRatio);
        peakRatio = 0.5 / rg;
        closingRatio = peakRatio * (1.0 + rk);
    }
    else
    {
        const double openQuotient = 1.0 - 1.0 / (2.17 * rd);
        const double rg = 0.0093552 + 5.96 / (7.92 - 2.0 * openQuotient);
        const double rk = 2.0 * rg * openQuotient - 0.9572;
        peakRatio = 0.5 / rg;
        closingRatio = peakRatio * (1.0 + rk);
    }

    const double tailLength = 1.0 - closingRatio;
    double lower = 0.0;
    double upper = 1.0 / returnRatio;
    for (int iteration = 0; iteration < 80; ++iteration)
    {
        const double decay = (lower + upper) * 0.5;
        if (-std::expm1(-tailLength * decay) / decay > returnRatio)
        {
            lower = decay;
        }
        else
        {
            upper = decay;
        }
    }
    const double decay = (lower + upper) * 0.5;
    const double tailArea = (-std::expm1(-tailLength * decay) - tailLength * decay * std::exp(-tailLength * decay)) / (decay * decay * returnRatio);
    const double angularFrequency = std::numbers::pi / peakRatio;
    const double closingSine = std::sin(angularFrequency * closingRatio);
    const double closingCosine = std::cos(angularFrequency * closingRatio);
    const auto areaResidual = [tailArea, angularFrequency, closingRatio, closingSine, closingCosine](double growth)
    {
        return tailArea * closingSine * (growth * growth + angularFrequency * angularFrequency) + growth * closingSine - angularFrequency * closingCosine + angularFrequency * std::exp(-growth * closingRatio);
    };
    lower = 0.0;
    upper = 1.0;
    while (areaResidual(upper) > 0.0)
    {
        upper *= 2.0;
    }
    for (int iteration = 0; iteration < 80; ++iteration)
    {
        const double growth = (lower + upper) * 0.5;
        if (areaResidual(growth) > 0.0)
        {
            lower = growth;
        }
        else
        {
            upper = growth;
        }
    }

    return {shape,
            static_cast<float>(closingRatio),
            static_cast<float>(peakRatio),
            static_cast<float>(returnRatio),
            static_cast<float>(std::log(decay / referencePeriodSeconds)),
            static_cast<float>(std::log((lower + upper) * 0.5 / referencePeriodSeconds))};
}

LfParameters interpolateParameters(float shape)
{
    static const auto knots = []
    {
        std::array<LfKnot, parameterKnotCount> values;
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            values[index] = makeKnot(index);
        }
        return values;
    }();

    const auto upper = std::upper_bound(knots.begin(), knots.end(), shape, [](float value, const LfKnot& knot)
                                        { return value < knot.shape; });
    const auto& right = upper == knots.end() ? knots.back() : *upper;
    const auto& left = upper == knots.begin() ? knots.front() : *(upper - 1);
    const double position = left.shape == right.shape ? 0.0 : (static_cast<double>(shape) - left.shape) / (right.shape - left.shape);
    return {std::lerp(left.closingRatio, right.closingRatio, position),
            std::lerp(left.peakRatio, right.peakRatio, position),
            std::lerp(left.returnRatio, right.returnRatio, position),
            std::exp(std::lerp(left.logDecay, right.logDecay, position)) * referencePeriodSeconds,
            std::exp(std::lerp(left.logGrowth, right.logGrowth, position)) * referencePeriodSeconds};
}

std::complex<double> getLfSpectrum(const LfParameters& parameters, std::size_t harmonic)
{
    const double omega = twoPi * static_cast<double>(harmonic);
    const double sourceOmega = std::numbers::pi / parameters.peakTime;
    const double closingAngle = sourceOmega * parameters.closingTime;
    const double closingSine = std::sin(closingAngle);
    const std::complex<double> s(parameters.growth, -omega);
    const auto closingRotation = std::polar(1.0, -omega * parameters.closingTime);
    const auto opening = -(closingRotation * (s * closingSine - sourceOmega * std::cos(closingAngle)) + sourceOmega * std::exp(-parameters.growth * parameters.closingTime)) / (closingSine * (s * s - sourceOmega * sourceOmega));

    const double tailLength = 1.0 - parameters.closingTime;
    const std::complex<double> tailRate(parameters.decay, omega);
    auto returning = -closingRotation / tailRate;
    const double endpoint = 1.0 - parameters.decay * parameters.returnTime;
    if (parameters.decay * endpoint >= referencePeriodSeconds)
    {
        returning += closingRotation * endpoint * (1.0 - std::polar(1.0, -omega * tailLength)) / (std::complex<double>(0.0, omega) * parameters.returnTime * tailRate);
    }
    return opening + returning;
}

juce::Result validateRender(std::span<const PeriodicExcitationFrame> frames, std::size_t hopSamples, double framePeriodSeconds)
{
    if (hopSamples == 0 || hopSamples > std::numeric_limits<std::size_t>::max() / 2 || !std::isfinite(framePeriodSeconds) || framePeriodSeconds <= 0.0)
    {
        return juce::Result::fail("Periodic excitation requires a positive hop and finite positive frame period.");
    }
    if (frames.size() > std::vector<float>().max_size() / hopSamples)
    {
        return juce::Result::fail("Periodic excitation sample count exceeds the container limit.");
    }
    const double sampleRate = static_cast<double>(hopSamples) / framePeriodSeconds;
    if (!std::isfinite(sampleRate) || sampleRate <= 0.0)
    {
        return juce::Result::fail("Periodic excitation sample rate exceeds the numeric range.");
    }
    for (const auto& frame : frames)
    {
        if (!std::isfinite(frame.fundamentalFrequencyHz) || frame.fundamentalFrequencyHz < 0.0f || (frame.voiced && frame.fundamentalFrequencyHz == 0.0f))
        {
            return juce::Result::fail("Periodic excitation requires finite non-negative F0 and positive F0 in voiced frames.");
        }
    }
    return juce::Result::ok();
}

std::vector<double> makeWindow(std::size_t hopSamples, double gain)
{
    std::vector<double> window(hopSamples * 2);
    for (std::size_t index = 0; index < window.size(); ++index)
    {
        window[index] = gain * (0.5 - 0.5 * std::cos(twoPi * static_cast<double>(index) / static_cast<double>(window.size())));
    }
    return window;
}

bool addWindow(std::vector<float>& output, std::size_t frameIndex, std::size_t hopSamples, std::span<const double> signal, std::span<const double> window)
{
    const std::size_t sourceBegin = frameIndex == 0 ? hopSamples : 0;
    const std::size_t outputBegin = frameIndex == 0 ? 0 : (frameIndex - 1) * hopSamples;
    for (std::size_t index = sourceBegin; index < signal.size(); ++index)
    {
        const std::size_t destination = outputBegin + index - sourceBegin;
        const double value = output[destination] + signal[index] * window[index];
        if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
        {
            return false;
        }
        output[destination] = static_cast<float>(value);
    }
    return true;
}
}

juce::Result renderPeriodicExcitation(std::span<const PeriodicExcitationFrame> frames,
                                      std::size_t hopSamples,
                                      double framePeriodSeconds,
                                      float gain,
                                      std::vector<float>& samples,
                                      std::vector<double>& phaseCycles)
{
    if (const auto result = validateRender(frames, hopSamples, framePeriodSeconds); result.failed())
    {
        return result;
    }
    if (!std::isfinite(gain) || gain < 0.0f)
    {
        return juce::Result::fail("Periodic excitation gain must be finite and non-negative.");
    }
    if (hopSamples > maximumFftSize / 4)
    {
        return juce::Result::fail("Periodic excitation hop exceeds the supported FFT size.");
    }
    for (const auto& frame : frames)
    {
        if (frame.voiced && (!std::isfinite(frame.shape) || frame.shape < 0.01f || frame.shape > 6.0f))
        {
            return juce::Result::fail("LF shape must be finite and within the verified parameter range [0.01, 6].");
        }
    }

    const float sampleRate = static_cast<float>(hopSamples) / static_cast<float>(framePeriodSeconds);
    if (!std::isfinite(sampleRate) || sampleRate <= 0.0f)
    {
        return juce::Result::fail("Periodic excitation sample rate exceeds the float range.");
    }
    const auto window = makeWindow(hopSamples, gain);
    std::vector<float> rendered(frames.size() * hopSamples, 0.0f);
    std::vector<double> phases(frames.size());
    std::vector<double> signal(hopSamples * 2);
    const std::size_t fftSize = std::bit_ceil(signal.size() * 2 - 1);
    juce::dsp::FFT fft(static_cast<int>(std::countr_zero(fftSize)));
    std::vector<std::complex<double>> chirp(signal.size());
    std::vector<std::complex<float>> spectrum(fftSize);
    std::vector<std::complex<float>> kernel(fftSize);
    std::vector<std::complex<float>> transformedKernel(fftSize);
    double accumulatedCycles = 0.0;
    for (std::size_t frameIndex = 0; frameIndex < frames.size(); ++frameIndex)
    {
        const auto& frame = frames[frameIndex];
        const double increment = static_cast<double>(frame.fundamentalFrequencyHz * static_cast<float>(framePeriodSeconds));
        accumulatedCycles += increment;
        if (!std::isfinite(accumulatedCycles))
        {
            return juce::Result::fail("Periodic excitation phase exceeds the numeric range.");
        }
        phases[frameIndex] = accumulatedCycles;
        if (!frame.voiced)
        {
            continue;
        }
        const float harmonicLimit = 0.5f * sampleRate / frame.fundamentalFrequencyHz;
        const auto harmonicCount = static_cast<std::size_t>(std::min(static_cast<double>(hopSamples * 2), static_cast<double>(harmonicLimit)));
        if (harmonicCount == 0)
        {
            continue;
        }
        const auto parameters = interpolateParameters(frame.shape);
        const double referenceMagnitude = std::abs(getLfSpectrum(parameters, 2));
        const double referencePhase = std::arg(getLfSpectrum(parameters, 1));
        const double amplitudeScale = 0.1 * sampleRate / (frame.fundamentalFrequencyHz * static_cast<double>(harmonicCount) * referenceMagnitude);
        const double framePhase = twoPi * (std::fmod(accumulatedCycles, 1.0) - increment);
        const std::size_t lastHarmonic = std::min(harmonicCount, hopSamples * 2 - 1);
        const double halfStep = std::numbers::pi * frame.fundamentalFrequencyHz / sampleRate;
        std::fill(spectrum.begin(), spectrum.end(), std::complex<float>{});
        std::fill(kernel.begin(), kernel.end(), std::complex<float>{});
        for (std::size_t index = 0; index < chirp.size(); ++index)
        {
            const double position = static_cast<double>(index);
            chirp[index] = std::polar(1.0, halfStep * position * position);
            kernel[index] = static_cast<std::complex<float>>(std::conj(chirp[index]));
        }
        for (std::size_t harmonic = 1; harmonic <= lastHarmonic; ++harmonic)
        {
            const auto lfSpectrum = getLfSpectrum(parameters, harmonic);
            const double amplitude = std::abs(lfSpectrum);
            const double phase = std::arg(lfSpectrum) + static_cast<double>(harmonic) * (framePhase - referencePhase);
            spectrum[harmonic] = static_cast<std::complex<float>>(std::polar(amplitude, phase) * chirp[harmonic]);
            kernel[fftSize - harmonic] = static_cast<std::complex<float>>(std::conj(chirp[harmonic]));
        }
        fft.perform(kernel.data(), transformedKernel.data(), false);
        fft.perform(spectrum.data(), kernel.data(), false);
        for (std::size_t index = 0; index < fftSize; ++index)
        {
            kernel[index] *= transformedKernel[index];
        }
        fft.perform(kernel.data(), spectrum.data(), true);
        for (std::size_t index = 0; index < signal.size(); ++index)
        {
            signal[index] = (static_cast<std::complex<double>>(spectrum[index]) * chirp[index]).real() * amplitudeScale;
        }
        if (!addWindow(rendered, frameIndex, hopSamples, signal, window))
        {
            return juce::Result::fail("LF excitation produced a sample outside the float range.");
        }
    }
    samples = std::move(rendered);
    phaseCycles = std::move(phases);
    return juce::Result::ok();
}

juce::Result renderPeriodicModulation(std::span<const PeriodicExcitationFrame> frames,
                                      std::span<const double> phaseCycles,
                                      std::span<const float> frequencyRatios,
                                      std::span<const float> parameters,
                                      std::size_t hopSamples,
                                      double framePeriodSeconds,
                                      std::vector<float>& samples)
{
    if (const auto result = validateRender(frames, hopSamples, framePeriodSeconds); result.failed())
    {
        return result;
    }
    if (phaseCycles.size() != frames.size() || frequencyRatios.empty() || frequencyRatios.size() > std::numeric_limits<std::size_t>::max() / 2)
    {
        return juce::Result::fail("Periodic modulation requires one phase per frame and at least one frequency ratio.");
    }
    const std::size_t channels = frequencyRatios.size() * 2;
    if (frames.size() > std::numeric_limits<std::size_t>::max() / channels || parameters.size() != frames.size() * channels)
    {
        return juce::Result::fail("Periodic modulation parameters must contain log amplitudes followed by phase offsets for every frame.");
    }
    for (const auto ratio : frequencyRatios)
    {
        if (!std::isfinite(ratio) || ratio <= 0.0f)
        {
            return juce::Result::fail("Periodic modulation frequency ratios must be finite and positive.");
        }
    }
    for (const auto value : parameters)
    {
        if (!std::isfinite(value))
        {
            return juce::Result::fail("Periodic modulation parameters must be finite.");
        }
    }

    const auto window = makeWindow(hopSamples, 1.0);
    std::vector<float> rendered(frames.size() * hopSamples, 0.0f);
    std::vector<double> signal(hopSamples * 2);
    for (std::size_t frameIndex = 0; frameIndex < frames.size(); ++frameIndex)
    {
        const float frameCycles = static_cast<float>(phaseCycles[frameIndex]);
        if (!std::isfinite(frameCycles))
        {
            return juce::Result::fail("Periodic modulation phase exceeds the float range.");
        }
        std::fill(signal.begin(), signal.end(), 0.0);
        for (std::size_t ratioIndex = 0; ratioIndex < frequencyRatios.size(); ++ratioIndex)
        {
            const double amplitude = std::exp(static_cast<double>(parameters[frameIndex * channels + ratioIndex]));
            const double angularRatio = twoPi * frequencyRatios[ratioIndex];
            const double phase = angularRatio * frameCycles + parameters[frameIndex * channels + frequencyRatios.size() + ratioIndex];
            const double step = angularRatio * frames[frameIndex].fundamentalFrequencyHz * framePeriodSeconds / static_cast<double>(hopSamples);
            for (std::size_t index = 0; index < signal.size(); ++index)
            {
                signal[index] += amplitude * std::cos(phase + (static_cast<double>(index) - static_cast<double>(hopSamples)) * step);
            }
        }
        if (!addWindow(rendered, frameIndex, hopSamples, signal, window))
        {
            return juce::Result::fail("Periodic modulation produced a sample outside the float range.");
        }
    }
    samples = std::move(rendered);
    return juce::Result::ok();
}
}
