import json

import numpy as np
import pytest
import soundfile

from opensv_train.filename_labels import (
    filename_recordings,
    labels_from_phones,
    parse_filename_phones,
    read_phone_table,
    write_lab_files,
)


PHONES = {"sil", "7", "@", ":n", "N", "a", "iE", "u@", "y{", "kh", "ts\\"}


def test_order_retake_suffix_and_internal_underscores():
    assert parse_filename_phones("7 @_n_0.wav", PHONES) == ["sil", "7", "@", ":n", "sil"]
    assert parse_filename_phones("@_n @_n_1.wav", PHONES) == ["sil", "@", ":n", "@", ":n", "sil"]


def test_internal_sil_is_kept_and_boundary_sils_are_always_added():
    assert parse_filename_phones("Sil m_0.wav", PHONES | {"m"}) == ["sil", "sil", "m", "sil"]


def test_explicit_composites_and_replacements():
    assert parse_filename_phones("@N_0.wav", PHONES) == ["sil", "@", "N", "sil"]
    assert parse_filename_phones("a_n_0.wav", PHONES) == ["sil", "a", ":n", "sil"]
    assert parse_filename_phones("iE_n_0.wav", PHONES) == ["sil", "iE", ":n", "sil"]
    assert parse_filename_phones("u@N_0.wav", PHONES) == ["sil", "u@", "N", "sil"]
    assert parse_filename_phones("y{_n_0.wav", PHONES) == ["sil", "y{", ":n", "sil"]
    assert parse_filename_phones("k_h_0.wav", PHONES) == ["sil", "kh", "sil"]
    assert parse_filename_phones(r"ts%5c_0.wav", PHONES) == ["sil", "ts\\", "sil"]


def test_known_complete_phone_wins_before_generic_segmentation():
    assert parse_filename_phones("@_n_0.wav", PHONES | {"@_n"}) == ["sil", "@_n", "sil"]


def test_equal_duration_boundaries_are_contiguous():
    entries = labels_from_phones(["sil", "sil", "m", "sil"], 20_000_000)
    assert entries == [
        (0, 2_000_000, "sil"),
        (2_000_000, 10_000_000, "sil"),
        (10_000_000, 18_000_000, "m"),
        (18_000_000, 20_000_000, "sil"),
    ]
    assert entries[0][0] == 0 and entries[-1][1] == 20_000_000
    assert all(a[1] == b[0] for a, b in zip(entries, entries[1:]))


def test_reads_json_phone_table(tmp_path):
    table = tmp_path / "phone_table.json"
    table.write_text(json.dumps({"phones": [{"symbol": "a"}, {"symbol": ":n"}]}), encoding="utf-8")
    assert read_phone_table(table) == {"sil", "a", ":n"}


def test_wav_duration_and_lab_output(tmp_path):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    soundfile.write(str(wav_dir / "Sil m_0.wav"), np.zeros(48000, dtype=np.float32), 48000)
    recordings = filename_recordings(wav_dir, PHONES | {"m"})
    assert len(recordings) == 1
    assert [phone for _, _, phone in recordings[0][2]] == ["sil", "sil", "m", "sil"]
    assert recordings[0][2][-1][1] == 10_000_000
    paths = write_lab_files(recordings, tmp_path / "lab")
    assert paths[0].read_text("utf-8").splitlines() == [
        "0 1000000 sil",
        "1000000 5000000 sil",
        "5000000 9000000 m",
        "9000000 10000000 sil",
    ]


def test_vocaloid_mandarin_maps_to_synthv_simplified_inventory():
    target = {
        "sil", "a", "aa", "o/", "o", "e/", "ee", "e", "u", "i", "v",
        "ao", "ou", "ia", "iaa", "iao", "iae", "io", "iu", "z", "c",
        "b", "p", "d", "t", "g", "k", "gw", "kw", "-n", "ng", "r", "er",
    }
    assert parse_filename_phones("@_n 7_0.wav", target, source_system="vocaloid-mandarin") == [
        "sil", "e/", "-n", "e", "sil"
    ]
    assert parse_filename_phones("p ph t th k kh ts tsh_0.wav", target, source_system="vocaloid-mandarin") == [
        "sil", "b", "p", "d", "t", "g", "k", "z", "c", "sil"
    ]
    assert parse_filename_phones("iE_n iE_r_1.wav", target, source_system="vocaloid-mandarin") == [
        "sil", "iae", "-n", "iae", "sil"
    ]
    assert parse_filename_phones("aI ei_0.wav", target, source_system="vocaloid-mandarin") == [
        "sil", "a", "i", "ee", "i", "sil"
    ]
    assert parse_filename_phones("AN @N iAN iN iUN_0.wav", target, source_system="vocaloid-mandarin") == [
        "sil", "aa", "ng", "e/", "ng", "iaa", "ng", "i", "ng", "io", "ng", "sil"
    ]
    assert parse_filename_phones("7 @`_0.wav", {"sil", "e", "e/", "r"}, source_system="vocaloid-mandarin") == [
        "sil", "e", "e/", "r", "sil"
    ]

def test_user_vocaloid_mandarin_filename_tokens_are_convertible():
    target = {
        "sil", "a", "aa", "o/", "e/", "ee", "e", "o", "u", "i", "iz", "ir", "v",
        "aI", "ei", "ao", "ou", "ia", "iaa", "iao", "ie", "iae", "io", "iu",
        "ua", "uo", "yE_r", "uai", "uei", "f", "er", "-n", "ng",
    }
    source_tokens = (
        "7", "@U", "@_n", "@`", "aI", "AN", "AU", "a", "a_n", "ei", "f",
        "i@U", "iAN", "iAU", "ia", "iE_n", "iE_r", "iN", "iUN", "i",
    )
    for token in source_tokens:
        parsed = parse_filename_phones(f"@_n {token}_0.wav", target, source_system="vocaloid-mandarin")
        assert parsed[0] == parsed[-1] == "sil"
        assert len(parsed) >= 3
        assert set(parsed) <= target | {"sil"}


def test_vocaloid6_underscored_aspirates_map_by_target_inventory():
    simplified = {"sil", "a", "p", "t", "k", "c", "q", "ch", "b", "d", "g", "z", "ng", "e/", "-n"}
    assert parse_filename_phones("p_h t_h k_h ts_h ts\\_h ts`_h_0.wav", simplified, source_system="vocaloid-mandarin") == [
        "sil", "p", "t", "k", "c", "q", "ch", "sil"
    ]
    legacy = {"sil", "p", "ph", "t", "th", "k", "kh", "ts", "tsh", "ts\\h", "ts`h"}
    assert parse_filename_phones("p_h t_h k_h ts_h ts\\_h ts`_h_0.wav", legacy, source_system="vocaloid-mandarin") == [
        "sil", "ph", "th", "kh", "tsh", "ts\\h", "ts`h", "sil"
    ]


def test_vocaloid_mapping_preserves_legacy_xsampa_inventory():
    legacy = {"sil", "a", "e", "7", "@", "A", "N", "p", "ph", "t", "th", "k", "kh", "ts", "tsh", ":n"}
    assert parse_filename_phones("7 p ts a_n @N_0.wav", legacy, source_system="vocaloid-mandarin") == [
        "sil", "7", "p", "ts", "a", ":n", "@", "N", "sil"
    ]

def test_unmatched_wav_files_are_reported_without_aborting_other_files(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    soundfile.write(str(wav_dir / "@_n 7_0.wav"), np.zeros(4800, dtype=np.float32), 48000)
    soundfile.write(str(wav_dir / "badphone_0.wav"), np.zeros(4800, dtype=np.float32), 48000)
    unmatched = []
    recordings = filename_recordings(wav_dir, {"sil", "e/", "-n", "e"}, source_system="vocaloid-mandarin", unmatched=unmatched)
    assert [recording[0] for recording in recordings] == ["@_n 7_0"]
    assert len(unmatched) == 1
    assert unmatched[0]["filename"] == "badphone_0.wav"
    assert "unknown phone" in unmatched[0]["error"]

def test_unknown_phone_fails_loudly():
    with pytest.raises(ValueError, match="unknown phone"):
        parse_filename_phones("definitely_unknown_0.wav", PHONES)


def test_rhoticized_vowel_uses_complete_symbol_when_available():
    assert parse_filename_phones("7 @`_0.wav", PHONES | {"@`"}) == ["sil", "7", "@`", "sil"]


def test_rhoticized_vowel_falls_back_to_available_rhotic_phone():
    assert parse_filename_phones("7 @`_0.wav", PHONES | {"r\\"}) == ["sil", "7", "@", "r\\", "sil"]
    assert parse_filename_phones("7 @`_0.wav", PHONES | {"r"}) == ["sil", "7", "@", "r", "sil"]


def test_rhoticized_vowel_explains_missing_fallback():
    with pytest.raises(ValueError, match="rhoticized vowel"):
        parse_filename_phones("7 @`_0.wav", PHONES)


def test_f0_at_handles_utterance_with_no_voiced_frames():
    from opensv_train.data import Utterance

    utterance = Utterance(
        name="Sil k_0",
        audio=np.zeros(4800, dtype=np.float32),
        phones=[],
        notes=None,
        f0_times=np.array([0.0, 0.005, 0.01]),
        f0_values=np.zeros(3, dtype=np.float32),
    )

    hz, voiced = utterance.f0_at(np.array([0.0025, 0.0075]))

    assert np.allclose(hz, [150.0, 150.0])
    assert not voiced.any()


def test_f0_at_keeps_voicing_mask_for_utterance_with_voiced_frames():
    from opensv_train.data import Utterance

    utterance = Utterance(
        name="vowel_0",
        audio=np.zeros(4800, dtype=np.float32),
        phones=[],
        notes=None,
        f0_times=np.array([0.0, 0.005, 0.01]),
        f0_values=np.array([0.0, 220.0, 220.0], dtype=np.float32),
    )

    hz, voiced = utterance.f0_at(np.array([0.005, 0.01]))

    assert np.allclose(hz, [220.0, 220.0])
    assert voiced.all()


def test_standard_xsampa_maps_to_synthv_mandarin_labels_without_guessing():
    target = {
        "sil", "a", "aa", "o/", "e/", "ee", "e", "o", "u", "i", "iz", "ir", "v",
        "ao", "ou", "ia", "iaa", "iao", "ie", "iae", "io", "iu", "va", "ve",
        "ua", "uaa", "ue/", "ue", "uo", "-i", "-r", "-n", "ng", "b", "p",
        "d", "t", "g", "k", "z", "c", "j", "q", "zh", "ch", "h", "f", "s",
        "sh", "x", "r", "m", "n", "l", "w", "y",
    }
    slash, grave = chr(92), chr(96)
    tokens = [
        "A", "O", "@", "e", "7", "U", "r" + slash, ":r" + slash + grave, "y",
        "au", "ou", "ia", "iA", "iau", "ie", "iE", "iU", "jou", "y{", "yE",
        "ua", "uA", "u@", "ue", "uo", ":" + slash + "i", "r" + slash + grave,
        ":n", "N", "p", "p_h", "t", "t_h", "k", "k_h", "ts", "ts_h",
        "ts" + slash, "ts" + slash + "h", "ts" + grave, "ts" + grave + "h",
        "x", "f", "s", "s" + grave, "s" + slash, "z" + grave, "w", "j",
    ]
    parsed = parse_filename_phones(" ".join(tokens) + "_0.wav", target, source_system="xsampa")
    assert parsed == [
        "sil", "aa", "o/", "e/", "ee", "e", "o", "iz", "ir", "v", "ao", "ou",
        "ia", "iaa", "iao", "ie", "iae", "io", "iu", "va", "ve", "ua", "uaa",
        "ue/", "ue", "uo", "-i", "-r", "-n", "ng", "b", "p", "d", "t", "g",
        "k", "z", "c", "j", "q", "zh", "ch", "h", "f", "s", "sh", "x", "r",
        "w", "y", "sil",
    ]


def test_legacy_synthv_xsampa_spellings_map_to_new_mandarin_inventory():
    target = {
        "sil", "o/", "ao", "ou", "iao", "iu", "iz", "ir", "-i", "p", "t", "k",
        "c", "q", "ch",
    }
    slash, grave = chr(92), chr(96)
    tokens = [
        "o", "AU", "@U", "iAU", "i@U", "i" + slash, "i" + grave,
        ":" + slash + "i", "ph", "th", "kh", "tsh",
        "ts" + slash + "h", "ts" + grave + "h",
    ]
    assert parse_filename_phones(
        " ".join(tokens) + "_0.wav", target, source_system="synthv-xsampa"
    ) == [
        "sil", "o/", "ao", "ou", "iao", "iu", "iz", "ir", "-i", "p", "t", "k",
        "c", "q", "ch", "sil",
    ]


def test_explicit_xsampa_mode_does_not_split_unknown_token_into_wrong_phones():
    with pytest.raises(ValueError, match="unknown phone"):
        parse_filename_phones("xyz_0.wav", {"sil", "x", "y", "z"}, source_system="xsampa")


def test_unreadable_wav_is_reported_without_dropping_other_valid_recordings(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    soundfile.write(str(wav_dir / "Sil Sil k Sil_0.wav"), np.zeros(4800, dtype=np.float32), 48000)
    (wav_dir / "k_0.wav").write_bytes(b"this is not a WAV")
    unmatched = []
    recordings = filename_recordings(wav_dir, {"sil", "k"}, unmatched=unmatched)
    assert [name for name, _, _ in recordings] == ["Sil Sil k Sil_0"]
    assert [phone for _, _, entries in recordings for _, _, phone in entries] == [
        "sil", "sil", "sil", "k", "sil", "sil",
    ]
    assert len(unmatched) == 1
    assert unmatched[0]["filename"] == "k_0.wav"
    assert "unreadable or damaged WAV" in unmatched[0]["error"]


def test_xsampa_compound_mandarin_finals_use_explicit_mappings_not_character_splitting():
    target = {"sil", "a", "i", "ai", "ei", "ao", "ou", "iao", "iu", "er"}
    tokens = "aI ei AU @U iAU i@U @" + chr(96)
    assert parse_filename_phones(
        f"{tokens}_0.wav", target, source_system="xsampa"
    ) == ["sil", "ai", "ei", "ao", "ou", "iao", "iu", "er", "sil"]


def test_uppercase_wav_extension_and_silence_k_sequence_are_kept(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    soundfile.write(
        str(wav_dir / "Sil Sil k Sil_0.WAV"),
        np.zeros(4800, dtype=np.float32),
        48000,
    )
    unmatched = []
    recordings = filename_recordings(wav_dir, {"sil", "k"}, unmatched=unmatched)
    assert len(recordings) == 1
    assert not unmatched
    assert [phone for _, _, phone in recordings[0][2]] == [
        "sil", "sil", "sil", "k", "sil", "sil",
    ]


def test_supported_pcm_wav_falls_back_when_primary_decoder_fails(tmp_path, monkeypatch):
    from opensv_train import audio_io

    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    soundfile.write(
        str(wav_dir / "Sil Sil k Sil_0.wav"),
        np.zeros(4800, dtype=np.float32),
        48000,
        subtype="PCM_16",
    )

    def reject_with_primary_reader(*_args, **_kwargs):
        raise RuntimeError("simulated libsndfile subtype limitation")

    monkeypatch.setattr(audio_io.soundfile, "read", reject_with_primary_reader)
    unmatched = []
    recordings = filename_recordings(wav_dir, {"sil", "k"}, unmatched=unmatched)
    assert len(recordings) == 1
    assert unmatched == []
    assert recordings[0][0] == "Sil Sil k Sil_0"


def test_unmatched_report_classifies_phone_mapping_and_wav_decode_failures(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    soundfile.write(str(wav_dir / "badphone_0.wav"), np.zeros(4800, dtype=np.float32), 48000)
    (wav_dir / "Sil Sil k Sil_0.wav").write_bytes(b"not a wav")
    unmatched = []
    recordings = filename_recordings(wav_dir, {"sil", "k"}, unmatched=unmatched)
    assert not recordings
    categories = {item["filename"]: item["category"] for item in unmatched}
    assert categories == {
        "badphone_0.wav": "phone_mapping",
        "Sil Sil k Sil_0.wav": "wav_decode",
    }


def test_standard_xsampa_aspirates_map_to_legacy_synthv_phone_symbols():
    legacy = {
        "sil", "p", "ph", "t", "th", "k", "kh", "ts", "tsh",
        r"ts\h", "ts" + chr(96) + "h",
    }
    tokens = ["p_h", "t_h", "k_h", "ts_h", r"ts\_h", "ts" + chr(96) + "_h"]
    parsed = parse_filename_phones(
        " ".join(tokens) + "_0.wav", legacy, source_system="xsampa"
    )
    assert parsed == [
        "sil", "ph", "th", "kh", "tsh", r"ts\h", "ts" + chr(96) + "h", "sil",
    ]


def test_synthv_legacy_xsampa_aliases_still_map_to_new_phone_symbols():
    simple = {
        "sil", "aa", "ee", "e/", "o/", "ao", "ou", "iz", "ir", "v",
        "ng", "-n", "b", "p", "d", "t", "g", "k", "z", "c", "q", "ch",
    }
    tokens = ["ph", "th", "kh", "tsh", r"ts\h", "ts" + chr(96) + "h"]
    parsed = parse_filename_phones(
        " ".join(tokens) + "_0.wav", simple, source_system="synthv-xsampa"
    )
    assert parsed == ["sil", "p", "t", "k", "c", "q", "ch", "sil"]
