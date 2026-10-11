#pragma once

#include <juce_core/juce_core.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace sv::synthesis
{
struct DnniNode
{
    std::uint32_t marker = 0;
    std::uint64_t typeId = 0;
    std::string type;
    std::size_t offset = 0;
    std::size_t payloadOffset = 0;
    std::size_t payloadSize = 0;
    std::vector<std::size_t> children;
};

struct DnniMatrix
{
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::vector<float> values;
};

class DnniReader
{
public:
    [[nodiscard]] juce::Result load(const juce::File& file);

    [[nodiscard]] juce::Result load(juce::MemoryBlock bytes);

    [[nodiscard]] std::uint32_t getVersion() const noexcept;

    [[nodiscard]] const std::vector<DnniNode>& getNodes() const noexcept;

    [[nodiscard]] std::span<const std::uint8_t> getPayload(std::size_t nodeIndex) const;

    [[nodiscard]] juce::Result readFloatVector(std::size_t nodeIndex, std::vector<float>& values) const;

    [[nodiscard]] juce::Result readFloatMatrix(std::size_t nodeIndex, DnniMatrix& matrix) const;

private:
    [[nodiscard]] juce::Result parse();
    [[nodiscard]] juce::Result parseNode(std::size_t& position, unsigned depth, std::size_t& nodeIndex);

    juce::MemoryBlock data;
    std::uint32_t version = 0;
    std::vector<DnniNode> nodes;
};

}
