import numpy as np
import pytest

from opensv_train.audio_to_midi import _extract_note_events, _midi_bytes


def _times(count):
    return np.arange(count, dtype=np.float64) * 0.01


def test_extracts_quantized_monophonic_notes_and_transposes():
    times = _times(60)
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
    assert notes[0][1] == pytest.approx(notes[1][0], abs=0.01)


def test_repairs_a_short_isolated_octave_tracking_spike():
    times = _times(50)
    frequencies = np.array([440.0] * 20 + [880.0] * 3 + [440.0] * 27)
    diagnostics = {}
    notes = _extract_note_events(
        times,
        frequencies,
        min_note_ms=20,
        median_window=1,
        diagnostics=diagnostics,
    )
    assert [note[2] for note in notes] == [69]
    assert diagnostics["octave_spikes_corrected"] == 1


def test_preserves_a_sustained_octave_change():
    times = _times(50)
    frequencies = np.array([440.0] * 20 + [880.0] * 30)
    notes = _extract_note_events(
        times, frequencies, min_note_ms=50, median_window=1
    )
    assert [note[2] for note in notes] == [69, 81]


def test_short_pitch_flicker_is_merged_into_the_surrounding_note():
    times = _times(50)
    frequencies = np.array([440.0] * 20 + [466.1638] + [440.0] * 29)
    notes = _extract_note_events(
        times, frequencies, min_note_ms=50, median_window=1
    )
    assert [note[2] for note in notes] == [69]


def test_tiny_unvoiced_gap_is_filled_only_when_pitch_matches_on_both_sides():
    times = _times(40)
    frequencies = np.array([440.0] * 20 + [0.0] * 2 + [440.0] * 18)
    diagnostics = {}
    notes = _extract_note_events(
        times,
        frequencies,
        min_note_ms=50,
        median_window=1,
        diagnostics=diagnostics,
    )
    assert [note[2] for note in notes] == [69]
    assert diagnostics["voiced_frames_percent"] == 95.0


def test_does_not_bridge_a_real_rest_between_different_notes():
    times = _times(70)
    frequencies = np.array([440.0] * 25 + [0.0] * 15 + [523.251] * 30)
    notes = _extract_note_events(
        times, frequencies, min_note_ms=50, median_window=1
    )
    assert [note[2] for note in notes] == [69, 72]
    assert notes[1][0] - notes[0][1] >= 0.10


def test_octave_correction_can_be_disabled_for_intentional_short_octave_notes():
    times = _times(50)
    frequencies = np.array([440.0] * 20 + [880.0] * 3 + [440.0] * 27)
    notes = _extract_note_events(
        times,
        frequencies,
        min_note_ms=20,
        median_window=1,
        octave_correction=False,
    )
    assert [note[2] for note in notes] == [69, 81, 69]


def test_rejects_audio_without_stable_voiced_notes():
    with pytest.raises(ValueError, match="No stable pitched notes"):
        _extract_note_events(
            _times(40),
            np.zeros(40, dtype=np.float64),
            median_window=1,
        )


def test_rejects_strength_contours_with_wrong_length():
    with pytest.raises(ValueError, match="pitch strengths must match"):
        _extract_note_events(
            _times(30),
            np.full(30, 440.0),
            strengths=np.ones(29),
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
