from opensv_train.filename_labels import parse_filename_phones
from opensv_train.phoneme_maps_sv1112 import (
    SV1112_CANTONESE_XSAMPA_TO_OLD,
    SV1112_CHINESE_XSAMPA_TO_OLD,
    SV1112_VOCALOID_MANDARIN_ALIASES,
)
from pathlib import Path
from trainsv_gui import PAGE


def test_sv1112_chinese_uses_official_xsampa_to_old_labels():
    slash = chr(92)
    grave = chr(96)
    target = {
        "sil", "a", "A", "o", "@", "e", "7", "U", "u", "i",
        "i" + slash, "i" + grave, "AU", "@U", "ia", "iA", "iAU",
        "ie", "iE", "iU", "i@U", "ts", "tsh", "p", "ph", "t", "th",
        "k", "kh", ":n", "N",
    }
    filename = "a A O @ e 7 U u i r" + slash + " au ou p_h t_h k_h ts_h @_n AN @N @" + grave + "_0.wav"
    parsed = parse_filename_phones(filename, target, source_system="sv1112-chinese")
    assert parsed == [
        "sil", "a", "A", "o", "@", "e", "7", "U", "u", "i",
        "i" + slash, "AU", "@U", "ph", "th", "kh", "tsh",
        "@", ":n", "A", "N", "@", "N", "i" + slash, "sil",
    ]
    assert set(parsed) <= target | {"sil"}


def test_sv1112_cantonese_preserves_legacy_consonants_vowels_and_codas():
    source_phones = (
        "ts tsh f h s l m n N w j p ph t th k kh kw kwh "
        "a 6 E e i I O o u U 9 8 y m= N= "
        ":i :y :u :m :n :N :p} :t} :k}"
    )
    target = {
        "sil", "ts", "tsh", "f", "h", "s", "l", "m", "n", "N", "w", "j",
        "p", "ph", "t", "th", "k", "kh", "kw", "kwh",
        "a", "6", "E", "e", "i", "I", "O", "o", "u", "U", "9", "8", "y",
        "m=", "N=", ":i", ":u", ":m", ":n", ":N", ":p_}", ":t_}", ":k_}",
    }
    parsed = parse_filename_phones(
        source_phones + "_0.wav",
        target,
        source_system="sv1112-cantonese",
    )
    expected = source_phones.split()
    expected[expected.index(":y")] = ":i"
    expected[expected.index(":p}")] = ":p_}"
    expected[expected.index(":t}")] = ":t_}"
    expected[expected.index(":k}")] = ":k_}"
    assert parsed == ["sil", *expected, "sil"]
    assert len(SV1112_CANTONESE_XSAMPA_TO_OLD) == 42


def test_mapping_tables_use_old_labels_not_2_3_romanized_labels():
    assert SV1112_CHINESE_XSAMPA_TO_OLD["O"] == ("o",)
    assert len(SV1112_CHINESE_XSAMPA_TO_OLD) == 55
    assert SV1112_CHINESE_XSAMPA_TO_OLD["i^"] == (":\\i",)
    assert SV1112_CHINESE_XSAMPA_TO_OLD["r" + chr(92)] == ("i" + chr(92),)
    assert SV1112_VOCALOID_MANDARIN_ALIASES["p_h"] == (("ph",),)
    assert SV1112_CANTONESE_XSAMPA_TO_OLD["p"] == ("p",)
    assert SV1112_CANTONESE_XSAMPA_TO_OLD["ph"] == ("ph",)
    assert SV1112_CANTONESE_XSAMPA_TO_OLD["kw"] == ("kw",)
    assert SV1112_CANTONESE_XSAMPA_TO_OLD["kwh"] == ("kwh",)
    assert SV1112_CANTONESE_XSAMPA_TO_OLD[":y"] == (":i",)
    assert SV1112_CANTONESE_XSAMPA_TO_OLD[":p}"] == (":p_}",)


def test_sv1112_modes_reject_unknown_symbols_instead_of_guessing():
    import pytest

    with pytest.raises(ValueError, match="no verified"):
        parse_filename_phones(
            "unknown_0.wav",
            {"sil", "u", "n"},
            source_system="sv1112-cantonese",
        )


def test_cli_lists_both_sv1112_mapping_modes():
    cli_source = (Path(__file__).resolve().parents[1] / "opensv_train" / "__main__.py").read_text("utf-8")
    assert '"sv1112-chinese", "sv1112-cantonese"' in cli_source


def test_gui_exposes_translated_sv1112_mapping_modes():
    assert "sv1112-chinese|SynthV 1.11.2 Chinese (Old labels)" in PAGE
    assert "sv1112-cantonese|SynthV 1.11.2 Cantonese (Old labels)" in PAGE
    assert "Synthesizer V 1.11.2 中文（旧版音素标签）" in PAGE
    assert "Synthesizer V 1.11.2 粤语（旧版音素标签）" in PAGE
