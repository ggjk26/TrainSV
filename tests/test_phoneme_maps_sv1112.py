from opensv_train.filename_labels import parse_filename_phones
from opensv_train.phoneme_maps_sv1112 import (
    SV1112_CANTONESE_XSAMPA_TO_OLD,
    SV1112_CHINESE_XSAMPA_TO_OLD,
)
from opensv_train.__main__ import parser
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
    phones = (
        "ts tsh f h s l m n N w j p ph t th k kh kw kwh "
        "a 6 E e i I O o u U 9 8 y m= N= "
        ":i :y :u :m :n :N :p} :t} :k}"
    )
    target = {"sil", *phones.split()}
    parsed = parse_filename_phones(
        phones + "_0.wav",
        target,
        source_system="sv1112-cantonese",
    )
    assert parsed == ["sil", *phones.split(), "sil"]
    assert len(SV1112_CANTONESE_XSAMPA_TO_OLD) == 43


def test_mapping_tables_use_old_labels_not_2_3_romanized_labels():
    assert SV1112_CHINESE_XSAMPA_TO_OLD["O"] == ("o",)
    assert SV1112_CHINESE_XSAMPA_TO_OLD["r" + chr(92)] == ("i" + chr(92),)
    assert SV1112_CHINESE_XSAMPA_TO_OLD["p_h"] == ("ph",)
    assert SV1112_CANTONESE_XSAMPA_TO_OLD["p"] == ("p",)
    assert SV1112_CANTONESE_XSAMPA_TO_OLD["ph"] == ("ph",)
    assert SV1112_CANTONESE_XSAMPA_TO_OLD["kw"] == ("kw",)
    assert SV1112_CANTONESE_XSAMPA_TO_OLD["kwh"] == ("kwh",)


def test_sv1112_modes_reject_unknown_symbols_instead_of_guessing():
    import pytest

    with pytest.raises(ValueError, match="no verified"):
        parse_filename_phones(
            "unknown_0.wav",
            {"sil", "u", "n"},
            source_system="sv1112-cantonese",
        )


def test_cli_accepts_both_sv1112_mapping_modes():
    args = parser().parse_args([
        "import-lab", "--output", "out", "--from-filenames", "--wavs", "wavs",
        "--source-phoneme-system", "sv1112-chinese",
    ])
    assert args.source_phoneme_system == "sv1112-chinese"
    args = parser().parse_args([
        "import-lab", "--output", "out", "--from-filenames", "--wavs", "wavs",
        "--source-phoneme-system", "sv1112-cantonese",
    ])
    assert args.source_phoneme_system == "sv1112-cantonese"


def test_gui_exposes_translated_sv1112_mapping_modes():
    assert "sv1112-chinese|SynthV 1.11.2 Chinese (Old labels)" in PAGE
    assert "sv1112-cantonese|SynthV 1.11.2 Cantonese (Old labels)" in PAGE
    assert "Synthesizer V 1.11.2 中文（旧版音素标签）" in PAGE
    assert "Synthesizer V 1.11.2 粤语（旧版音素标签）" in PAGE
