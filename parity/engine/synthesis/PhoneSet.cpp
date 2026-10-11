#include "PhoneSet.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <utility>

namespace sv::synthesis
{
namespace
{
std::uint32_t readUint32(std::span<const std::uint8_t> bytes, std::size_t offset)
{
    return static_cast<std::uint32_t>(bytes[offset])
         | (static_cast<std::uint32_t>(bytes[offset + 1]) << 8)
         | (static_cast<std::uint32_t>(bytes[offset + 2]) << 16)
         | (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

bool readString(std::span<const std::uint8_t> bytes, std::size_t& position, std::string& value)
{
    if (bytes.size() - position < 4)
    {
        return false;
    }
    const std::size_t length = readUint32(bytes, position);
    position += 4;
    if (length > bytes.size() - position || length > 4096)
    {
        return false;
    }
    const auto* text = reinterpret_cast<const char*>(bytes.data() + position);
    if (!juce::CharPointer_UTF8::isValidString(text, static_cast<int>(length)) || std::find(text, text + length, '\0') != text + length)
    {
        return false;
    }
    value.assign(text, length);
    position += length;
    return true;
}

bool readStringList(std::span<const std::uint8_t> bytes, std::size_t& position, std::vector<std::string>& strings)
{
    if (bytes.size() - position < 4)
    {
        return false;
    }
    const std::size_t count = readUint32(bytes, position);
    position += 4;
    if (count > 4096 || count > (bytes.size() - position) / 4)
    {
        return false;
    }
    strings.resize(count);
    for (auto& string : strings)
    {
        if (!readString(bytes, position, string))
        {
            return false;
        }
    }
    return true;
}

}

juce::Result readPhoneSet(const DnniReader& reader, std::size_t nodeIndex, PhoneSet& output)
{
    const auto& nodes = reader.getNodes();
    if (nodeIndex >= nodes.size() || nodes[nodeIndex].type != "_psv2" || !nodes[nodeIndex].children.empty())
    {
        return juce::Result::fail("Phoneme tables require a leaf _psv2 node.");
    }

    PhoneSet phoneSet;
    const auto bytes = reader.getPayload(nodeIndex);
    std::size_t position = 0;

    if (!readString(bytes, position, phoneSet.name) ||
        !readStringList(bytes, position, phoneSet.symbols) ||
        !readStringList(bytes, position, phoneSet.categories) ||
        !readStringList(bytes, position, phoneSet.unifiedSymbols) ||
        !readStringList(bytes, position, phoneSet.classes) ||
        position != bytes.size())
    {
        return juce::Result::fail("Invalid _psv2 string table.");
    }

    if (phoneSet.name.empty() || phoneSet.symbols.empty() ||
        phoneSet.categories.size() != phoneSet.symbols.size() ||
        phoneSet.unifiedSymbols.size() < phoneSet.symbols.size())
    {
        return juce::Result::fail("Phoneme _psv2 table has inconsistent symbol metadata.");
    }

    std::set<std::string> symbols;
    std::set<std::string> classes;
    for (const auto& category : phoneSet.classes)
    {
        if (category.empty() || !classes.insert(category).second)
        {
            return juce::Result::fail("Phoneme _psv2 class names must be nonempty and unique.");
        }
    }

    for (std::size_t index = 0; index < phoneSet.symbols.size(); ++index)
    {
        if (phoneSet.symbols[index].empty() ||
            !symbols.insert(phoneSet.symbols[index]).second ||
            !classes.contains(phoneSet.categories[index]) ||
            phoneSet.unifiedSymbols[index].empty())
        {
            return juce::Result::fail("Phoneme _psv2 contains an invalid symbol, category, or mapping.");
        }
    }

    output = std::move(phoneSet);
    return juce::Result::ok();
}

}
