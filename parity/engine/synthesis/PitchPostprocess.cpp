#include "PitchPostprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace sv::synthesis
{
juce::Result applyPitchPostprocess(std::span<const PitchNote> notes, float frameIntervalSeconds, std::span<float> midiPitch, const std::function<bool()>& shouldCancel)
{
    if (!std::isfinite(frameIntervalSeconds) || frameIntervalSeconds <= 0.0f || midiPitch.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        return juce::Result::fail("Pitch correction: invalid frame interval or sequence size.");
    }
    if (!std::all_of(midiPitch.begin(), midiPitch.end(), [](float pitch)
                     { return std::isfinite(pitch); }))
    {
        return juce::Result::fail("Pitch correction: non-finite predicted pitch.");
    }
    const double inverseInterval = 1.0 / static_cast<double>(frameIntervalSeconds);
    double seconds = 0.0;
    for (const auto& note : notes)
    {
        if (shouldCancel && shouldCancel())
        {
            return juce::Result::fail("Pitch correction: cancelled.");
        }
        const double duration = note.syllable.durationSeconds;
        const double endSeconds = seconds + duration;
        const double endPosition = endSeconds * inverseInterval;
        if (!std::isfinite(duration) || duration < 0.0 || !std::isfinite(endPosition) || endPosition > static_cast<double>(std::numeric_limits<int>::max()))
        {
            return juce::Result::fail("Pitch correction: note timing is outside the supported range.");
        }
        const auto start = static_cast<std::size_t>(seconds * inverseInterval);
        const auto end = static_cast<std::size_t>(endPosition);
        seconds = endSeconds;
        if (note.isSilence)
        {
            continue;
        }
        const auto middleStart = start + (end - start) / 4;
        const auto middleEnd = std::min(start + (end - start) * 3 / 4, midiPitch.size());
        if (middleStart >= middleEnd)
        {
            continue;
        }
        double weightedPitch = 0.0;
        double weightSum = 0.0;
        for (auto frame = middleStart; frame < middleEnd; ++frame)
        {
            if ((frame & 255) == 0 && shouldCancel && shouldCancel())
            {
                return juce::Result::fail("Pitch correction: cancelled.");
            }
            const double pitch = midiPitch[frame];
            const auto previous = frame < 2 ? frame : frame - 1;
            const auto next = std::min(frame + 1, midiPitch.size() - 1);
            const double weight = 1.0 / (std::abs(static_cast<double>(midiPitch[previous]) - pitch) + 0.1 + std::abs(static_cast<double>(midiPitch[next]) - pitch));
            weightSum += weight;
            weightedPitch += weight * pitch;
        }
        const double offset = static_cast<double>(note.syllable.midiPitch) - weightedPitch / weightSum;
        if (std::abs(offset) <= static_cast<double>(0.2f))
        {
            continue;
        }
        const double correction = offset * std::clamp((duration - static_cast<double>(0.3f)) * 10.0, 0.0, 1.0);
        const auto limit = std::min(end, midiPitch.size());
        for (auto frame = start; frame < limit; ++frame)
        {
            if ((frame & 255) == 0 && shouldCancel && shouldCancel())
            {
                return juce::Result::fail("Pitch correction: cancelled.");
            }
            const double left = std::min(static_cast<double>(frame - start) * 0.1, 1.0);
            const double right = std::min(static_cast<double>(end - frame) * 0.1, 1.0);
            midiPitch[frame] = static_cast<float>(static_cast<double>(midiPitch[frame]) + left * correction * right);
        }
    }
    return juce::Result::ok();
}
}
