#pragma once

#include "DnniInference.h"
#include "DnniReader.h"
#include "FeatureNormalizer.h"
#include "SynthesisStatistics.h"
#include "VocoderFilterBank.h"
#include "VocoderResidual.h"
#include "VocoderSpectralHeads.h"

#include <juce_core/juce_core.h>

#include <array>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

namespace sv::synthesis
{
struct NeuralVocoderOutput
{
    double sampleRate = 0.0;
    std::size_t hopSamples = 0;
    std::vector<float> samples;
    std::vector<float> periodic;
    std::vector<float> aperiodic;
};

class NeuralVocoder
{
public:
    struct State
    {
        DnniInference::Cache shapeNetwork;
        DnniInference::Cache conditionNetwork;
        DnniInference::Cache modulationNetwork;
        VocoderSpectralHeads::State spectralHeads;
        VocoderResidual::State residual;

        [[nodiscard]] std::size_t getBytes() const noexcept;
    };

    [[nodiscard]] juce::Result load(const DnniReader& reader, std::size_t rootNode = 0);

    [[nodiscard]] juce::Result run(const DnniTensor& input, NeuralVocoderOutput& output, std::span<const float> shapeControl = {}, SynthesisStatistics* statistics = nullptr, State* state = nullptr, const std::function<bool()>& shouldCancel = {}, DnniRunStatistics* networkStatistics = nullptr) const;

private:
    FeatureNormalizer inputNormalizer;
    DnniInference shapeNetwork;
    DnniInference conditionNetwork;
    DnniInference modulationNetwork;
    VocoderSpectralHeads spectralHeads;
    std::array<VocoderFilterBank, 3> filters;
    VocoderResidual residual;
    std::vector<float> modulationRatios;
    double sampleRate = 0.0;
    double framePeriodSeconds = 0.0;
    std::size_t hopSamples = 0;
    float excitationGain = 0.0f;
    float modulationBias = 0.0f;
    bool forceVoicedFeatures = false;
    bool loaded = false;
};
}
