#pragma once

#include "VoiceDatabase.h"

#include <juce_core/juce_core.h>

#include <cstdint>
#include <string>
#include <vector>

namespace sv::synthesis
{
enum class VoiceConfigurationType : std::uint32_t
{
    strings = 0,
    numbers = 1,
    integers = 2
};

struct VoiceConfigurationEntry
{
    VoiceConfigurationType type = VoiceConfigurationType::strings;
    std::uint32_t ordinal = 0;
    juce::MemoryBlock key;
    std::string name;
    std::vector<juce::MemoryBlock> strings;
    std::vector<double> numbers;
    std::vector<std::uint32_t> integers;
};

struct ModelReference
{
    juce::MemoryBlock key;
    std::string architecture;
    std::string name;
};

class VoiceConfiguration
{
public:
    [[nodiscard]] juce::Result load(VoiceDatabase& database);

    [[nodiscard]] const std::vector<VoiceConfigurationEntry>& getEntries() const noexcept;

    [[nodiscard]] const ModelReference& getDurationModel() const noexcept;

    [[nodiscard]] const ModelReference& getAcousticModel() const noexcept;

    [[nodiscard]] const ModelReference& getVocoderModel() const noexcept;

private:
    std::vector<VoiceConfigurationEntry> entries;
    ModelReference duration;
    ModelReference acoustic;
    ModelReference vocoder;
};

}
