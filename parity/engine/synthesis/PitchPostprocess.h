#pragma once

#include "PitchFeatures.h"

#include <juce_core/juce_core.h>

#include <functional>
#include <span>

namespace sv::synthesis
{
[[nodiscard]] juce::Result applyPitchPostprocess(std::span<const PitchNote> notes, float frameIntervalSeconds, std::span<float> midiPitch, const std::function<bool()>& shouldCancel = {});
}
