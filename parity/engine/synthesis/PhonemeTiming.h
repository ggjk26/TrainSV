#pragma once

#include "AcousticFeatures.h"
#include "DnniInference.h"
#include "FeatureNormalizer.h"
#include "PhoneSet.h"

#include <juce_core/juce_core.h>

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace sv::synthesis
{
struct TimingSyllable
{
    std::string language;
    std::vector<std::string> phonemes;
    double durationSeconds = 0.0;
    int midiPitch = 60;
    bool isContinuation = false;
};

struct PhonemeDuration
{
    std::string language;
    std::string symbol;
    std::size_t syllableIndex = 0;
    std::size_t timingIntervalIndex = 0;
    double durationSeconds = 0.0;
};

[[nodiscard]] juce::Result quantizePhonemeDurations(std::span<const PhonemeDuration> durations, float frameIntervalSeconds, std::vector<TimedPhoneme>& output);

class PhonemeTiming
{
public:
    [[nodiscard]] juce::Result load(const DnniReader& reader, std::size_t rootNode = 0);
    [[nodiscard]] juce::Result setVoiceEmbedding(std::vector<float> replacement);

    [[nodiscard]] juce::Result predict(std::span<const TimingSyllable> syllables, std::vector<PhonemeDuration>& output) const;

private:
    std::vector<PhoneSet> phoneSets;
    PhoneSet unifiedPhoneSet;
    FeatureNormalizer inputNormalization;
    FeatureNormalizer outputNormalization;
    DnniMatrix languageEmbedding;
    DnniMatrix phonemeEmbedding;
    DnniMatrix positionEmbedding;
    std::vector<float> voiceEmbedding;
    DnniInference network;
    int minimumPitch = 0;
    int maximumPitch = 0;
};

}
