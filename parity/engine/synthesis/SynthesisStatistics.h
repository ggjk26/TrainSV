#pragma once

#include <cstddef>

namespace sv::synthesis
{
struct SynthesisStatistics
{
    std::size_t pitchFrames = 0;
    std::size_t acousticFrames = 0;
    std::size_t vocoderFrames = 0;

    std::size_t dnniComputedFrames = 0;
    std::size_t dnniReusedFrames = 0;
    std::size_t dnniContextFrames = 0;

    double pitchMilliseconds = 0.0;
    double acousticMilliseconds = 0.0;
    double vocoderMilliseconds = 0.0;
    double vocoderNetworkMilliseconds = 0.0;
    double vocoderSourceMilliseconds = 0.0;
    double vocoderResidualMilliseconds = 0.0;
    double vocoderFilterMilliseconds = 0.0;
};

}
