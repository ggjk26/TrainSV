import numpy as np
import pytest

from opensv_train.audio_to_midi import _extract_note_events, _midi_bytes


def test_extracts_quantized_monophonic_notes_and_transposes():
    times = np.arange(60, dtype=np.float64) * 0.01
    frequencies = np.array([440.0] * 30 + [523.251] * 30)
    notes = _extract_note_events(
        times,
        frequencies,
        min_note_ms=50,
        median_window=1,
        transpose_semitones=2,
    )
    assert [note[2] for note in notes] == [71, 74]
    assert notes[0][0] == 0
    assert notes[0][1] <= notes[1][0] + 0.02


def test_rejects_audio_without_stable_voiced_notes():
    with pytest.raises(ValueError, match="No stable pitched notes"):
        _extract_note_events(
            np.arange(40, dtype=np.float64) * 0.01,
            np.zeros(40, dtype=np.float64),
            median_window=1,
        )


def test_writes_standard_single_track_midi_with_note_events():
    midi = _midi_bytes([(0.0, 0.5, 69), (0.5, 1.0, 71)], tempo_bpm=120)
    assert midi[:4] == b"MThd"
    assert midi[4:14] == b"\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0"
    assert midi[14:18] == b"MTrk"
    assert b"\x90\x45\x60" in midi
    assert b"\x80\x45\x00" in midi
    assert b"\x90\x47\x60" in midi
    assert midi.endswith(b"\x00\xff\x2f\x00")


@pytest.mark.parametrize("tempo", [0, 19, 301, float("nan")])
def test_rejects_invalid_midi_tempo(tempo):
    with pytest.raises(ValueError, match="tempo must be between"):
        _midi_bytes([(0.0, 0.5, 69)], tempo_bpm=tempo)
