import csv
from pathlib import Path

import numpy as np
import soundfile

from opensv_train.audit import audit_dataset


def _write_dataset(root, rows):
    wavs = root / "wavs"
    wavs.mkdir(parents=True)
    with (root / "transcriptions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "ph_seq", "ph_dur"])
        writer.writeheader()
        writer.writerows(rows)
    return wavs


def _write_silence(path, seconds=1.0):
    soundfile.write(str(path), np.zeros(round(seconds * 48000), dtype=np.float32), 48000)


def test_audit_finds_missing_wavs_rare_phones_transitions_and_short_phones(tmp_path):
    wavs = _write_dataset(tmp_path, [
        {"name": "one", "ph_seq": "SP k a t i SP", "ph_dur": "0.1 0.01 0.39 0.015 0.385 0.1"},
        {"name": "two", "ph_seq": "SP k a t i SP", "ph_dur": "0.1 0.01 0.39 0.015 0.385 0.1"},
        {"name": "three", "ph_seq": "SP p u SP", "ph_dur": "0.1 0.4 0.4 0.1"},
        {"name": "missing", "ph_seq": "SP z a SP", "ph_dur": "0.1 0.4 0.4 0.1"},
    ])
    for name in ("one", "two", "three"):
        _write_silence(wavs / f"{name}.wav")

    report = audit_dataset(
        tmp_path,
        min_phone_recordings=2,
        min_transition_recordings=2,
        short_phone_ms=20,
        check_audio=True,
    )

    assert report["fatal_errors"] == []
    assert report["summary"]["rows"] == 4
    assert report["summary"]["trainable_rows"] == 3
    assert report["summary"]["missing_wavs"] == 1
    assert report["summary"]["invalid_rows"] == 0
    assert report["summary"]["short_phone_occurrences"] == 4
    assert report["summary"]["unreadable_wavs"] == 0
    assert report["duration_mismatches"] == []

    missing = report["missing_wavs"]
    assert missing == [{"name": "missing", "rows": [5]}]
    rare_phones = {item["phone"]: item["recordings"] for item in report["rare_phones"]}
    assert rare_phones == {"p": 1, "u": 1}
    rare_pairs = {(item["from"], item["to"]): item["recordings"] for item in report["rare_transitions"]}
    assert rare_pairs == {("p", "u"): 1}
    assert {item["phone"] for item in report["short_phone_examples"]} == {"k", "t"}


def test_audit_flags_duplicate_names_malformed_labels_and_unreadable_audio(tmp_path):
    wavs = _write_dataset(tmp_path, [
        {"name": "same", "ph_seq": "SP m SP", "ph_dur": "0.2 0.6 0.2"},
        {"name": "same", "ph_seq": "SP m SP", "ph_dur": "0.2 0.6 0.2"},
        {"name": "bad", "ph_seq": "k a", "ph_dur": "0.8"},
        {"name": "corrupt", "ph_seq": "SP n SP", "ph_dur": "0.2 0.6 0.2"},
        {"name": "../outside", "ph_seq": "SP z SP", "ph_dur": "0.2 0.6 0.2"},
    ])
    _write_silence(wavs / "same.wav")
    _write_silence(wavs / "bad.wav")
    (wavs / "corrupt.wav").write_bytes(b"not a WAV file")

    report = audit_dataset(tmp_path, check_audio=True)

    assert report["summary"]["duplicate_names"] == 1
    assert report["duplicate_names"] == ["same"]
    assert report["summary"]["invalid_rows"] == 2
    invalid_by_name = {item["name"]: item["errors"] for item in report["invalid_rows"]}
    assert any("ph_seq has 2 phones" in error for error in invalid_by_name["bad"])
    assert any("must be a file stem" in error for error in invalid_by_name["../outside"])
    assert report["summary"]["unreadable_wavs"] == 1
    assert report["unreadable_wavs"][0]["name"] == "corrupt"
    assert report["summary"]["missing_wavs"] == 1


def test_audit_reports_required_column_errors_without_modifying_dataset(tmp_path):
    (tmp_path / "wavs").mkdir()
    source = tmp_path / "transcriptions.csv"
    source.write_text("name,ph_seq\nsong,a\n", encoding="utf-8")

    report = audit_dataset(tmp_path)

    assert any("ph_dur" in error for error in report["fatal_errors"])
    assert source.read_text("utf-8") == "name,ph_seq\nsong,a\n"


def test_audit_rejects_invalid_thresholds(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="min_phone_recordings"):
        audit_dataset(tmp_path, min_phone_recordings=0)
    with pytest.raises(ValueError, match="short_phone_ms"):
        audit_dataset(tmp_path, short_phone_ms=-1)



def test_audit_data_command_is_documented_in_cli():
    cli = (Path(__file__).resolve().parents[1] / "opensv_train" / "__main__.py").read_text("utf-8")
    assert 'commands.add_parser("audit-data"' in cli
    assert 'command.set_defaults(handler=audit_data_command)' in cli
    assert 'command.add_argument("--strict"' in cli



def test_transition_frame_weights_emphasize_only_speech_handoffs():
    from types import SimpleNamespace

    from opensv_train.train import transition_frame_weights

    phones = [
        SimpleNamespace(symbol="sil"),
        SimpleNamespace(symbol="k"),
        SimpleNamespace(symbol="a"),
        SimpleNamespace(symbol="t"),
        SimpleNamespace(symbol="sil"),
    ]
    frames = [3, 4, 5, 4, 3]

    baseline = transition_frame_weights(phones, frames)
    assert np.all(baseline == 1.0)

    weighted = transition_frame_weights(phones, frames, weight=1.5, radius=2)
    assert weighted.shape == (sum(frames),)
    assert np.all(weighted[:3] == 1.0)  # silence -> consonant is excluded
    assert np.all(weighted[5:9] == 1.5)  # k -> a handoff
    assert np.all(weighted[10:14] == 1.5)  # a -> t handoff
    assert np.all(weighted[13:] == 1.0)  # consonant -> silence is excluded


def test_transition_frame_weights_validate_settings_and_phone_alignment():
    import pytest
    from types import SimpleNamespace

    from opensv_train.train import transition_frame_weights

    phones = [SimpleNamespace(symbol="k"), SimpleNamespace(symbol="a")]
    with pytest.raises(ValueError, match="at least 1.0"):
        transition_frame_weights(phones, [2, 2], weight=0.5)
    with pytest.raises(ValueError, match="non-negative"):
        transition_frame_weights(phones, [2, 2], radius=-1)
    with pytest.raises(ValueError, match="one positive duration"):
        transition_frame_weights(phones, [2])
    with pytest.raises(ValueError, match="one positive duration"):
        transition_frame_weights(phones, [2, 0])
