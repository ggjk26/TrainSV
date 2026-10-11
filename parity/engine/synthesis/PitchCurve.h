#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace sv::synthesis
{
struct PitchCurveNote
{
    double startSeconds = 0.0;

    double endSeconds = 0.0;

    double midiPitch = 60.0;

    double secondsPerQuarter = 0.5;

    bool instantMode = true;

    double tF0Offset = 0.0;

    double tF0Left = 0.1;

    double tF0Right = 0.07;

    double dF0Left = 0.15;

    double dF0Right = 0.15;

    double tF0VbrStart = 0.25;

    double tF0VbrLeft = 0.2;

    double tF0VbrRight = 0.2;

    double dF0Vbr = 1.0;

    double pF0Vbr = 0.0;

    double fF0Vbr = 5.5;

    double rapIntonation = 0.0;
};

[[nodiscard]] std::vector<float> renderPitchTransitions(
    std::span<const PitchCurveNote> notes,
    double firstFrameSeconds,
    double frameIntervalSeconds,
    std::size_t frameCount);

[[nodiscard]] std::vector<float> renderPitchVibrato(
    std::span<const PitchCurveNote> notes,
    double firstFrameSeconds,
    double frameIntervalSeconds,
    std::size_t frameCount);

[[nodiscard]] std::vector<float> renderPitchPredictionBase(
    std::span<const PitchCurveNote> notes,
    double firstFrameSeconds,
    double frameIntervalSeconds,
    std::size_t frameCount);

[[nodiscard]] std::vector<float> renderAutomaticPitchMask(
    std::span<const PitchCurveNote> notes,
    double firstFrameSeconds,
    double frameIntervalSeconds,
    std::size_t frameCount);

}

