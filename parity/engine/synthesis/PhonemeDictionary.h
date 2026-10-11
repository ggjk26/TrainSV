#pragma once

#include <juce_core/juce_core.h>

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sv::synthesis
{
struct PhonemeDefinition
{
    std::string symbol;
    std::string category;
};

class PhonemeDictionary final
{
public:
    [[nodiscard]] juce::Result load(const juce::File& phonesFile, const juce::File& dictionaryFile);

    [[nodiscard]] juce::Result loadJapanese(const juce::File& phonesFile, const juce::File& dictionaryFile, const juce::File& hiraganaFile, const juce::File& katakanaFile, const juce::File& smallKanaFile);

    [[nodiscard]] juce::Result loadMandarin(const juce::File& phonesFile, const juce::File& dictionaryFile, const juce::File& cedictFile);

    [[nodiscard]] juce::Result lookup(std::string_view lyrics, std::vector<std::string>& output) const;

    [[nodiscard]] juce::Result parseExplicitPhonemes(std::string_view text, std::vector<std::string>& output) const;

    [[nodiscard]] bool hasEntry(std::string_view key) const;

    [[nodiscard]] bool isLoaded() const;

    [[nodiscard]] const std::vector<PhonemeDefinition>& getPhonemes() const;

    [[nodiscard]] std::size_t getEntryCount() const;

private:
    std::vector<PhonemeDefinition> phonemes;
    std::unordered_set<std::string> symbols;
    std::unordered_map<std::string, std::vector<std::string>> entries;
    std::unordered_map<std::string, std::string> kanaToRomaji;
    std::unordered_map<std::string, std::string> smallKanaToRomaji;
    bool isMandarin = false;
    bool isJapanese = false;
    bool isEnglishArpabet = false;
};

}
