#pragma once

#include "DnniInference.h"
#include "DnniReader.h"
#include "FeatureNormalizer.h"
#include "PhoneSet.h"

#include <juce_core/juce_core.h>

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sv::synthesis
{
struct TimedPhoneme
{
    std::string language;
    std::string symbol;
    std::size_t frameCount = 0;
};

struct AcousticFeatureConfig
{
    std::size_t phonemeChannels = 0;
    std::size_t pitchChannels = 0;
    std::size_t acousticChannels = 0;
    float frameIntervalSeconds = 0.0f;
};

class AcousticFeatures
{
public:
    [[nodiscard]] juce::Result load(const DnniReader& reader, std::size_t nodeIndex);
    [[nodiscard]] const AcousticFeatureConfig& getConfig() const noexcept;
    [[nodiscard]] const std::vector<PhoneSet>& getPhoneSets() const noexcept;
    [[nodiscard]] const PhoneSet& getUnifiedPhoneSet() const noexcept;
    [[nodiscard]] juce::Result findLanguageIndex(std::string_view language, std::size_t& index) const;
    [[nodiscard]] juce::Result encodePhonemes(std::span<const TimedPhoneme> phonemes, DnniTensor& output) const;
    [[nodiscard]] juce::Result normalizeLogF0(std::span<const float> logF0, DnniTensor& output) const;
    [[nodiscard]] juce::Result decodeAcoustic71(const DnniTensor& normalized, std::span<const float> logF0, DnniTensor& output) const;

private:
    AcousticFeatureConfig config;
    std::vector<PhoneSet> phoneSets;
    PhoneSet unifiedPhoneSet;
    FeatureNormalizer phonemeNormalization;
    FeatureNormalizer pitchNormalization;
    FeatureNormalizer acousticNormalization;
};
}
