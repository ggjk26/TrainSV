#pragma once

#include <juce_core/juce_core.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace sv::synthesis
{
struct VoiceEntry
{
    juce::MemoryBlock key;
    juce::String name;
    std::uint64_t valueOffset = 0;
    std::uint32_t valueSize = 0;
};

struct VoiceMetadata
{
    juce::String name;
    juce::String vendor;
    int version = -1;
    juce::String language;
    juce::String phoneset;
    juce::String type;
    juce::StringArray languages;
    juce::StringArray timbreStyles;
    juce::StringPairArray properties{false};
};

struct VoiceInfo
{
    juce::File durationModel;
    juce::File pitchModel;
    juce::File acousticModel;
    juce::File vocoderModel;
    std::vector<float> durationVoice;
    std::vector<float> pitchSpeaker;
    std::vector<float> acousticSpeaker;
    std::vector<float> defaultStyle;
    std::vector<std::vector<float>> vocalModeStyles;
};

class VoiceDatabase
{
public:
    [[nodiscard]] juce::Result open(const juce::File& file);

    [[nodiscard]] const std::vector<VoiceEntry>& getEntries() const noexcept;

    [[nodiscard]] const VoiceMetadata& getMetadata() const noexcept;

    [[nodiscard]] const std::optional<VoiceInfo>& getInfo() const noexcept;

    [[nodiscard]] const juce::File& getFile() const noexcept;

    [[nodiscard]] const VoiceEntry* findEntry(juce::StringRef name) const;

    [[nodiscard]] juce::Result readEntry(const VoiceEntry& entry, juce::MemoryBlock& destination);

private:
    [[nodiscard]] juce::Result readMetadata();
    [[nodiscard]] juce::Result openInfo(const juce::File& infoFile);

    juce::File file;
    std::unique_ptr<juce::FileInputStream> stream;
    std::vector<VoiceEntry> entries;
    VoiceMetadata metadata;
    std::optional<VoiceInfo> info;
};

}
