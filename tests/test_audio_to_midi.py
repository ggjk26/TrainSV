from __future__ import annotations

import struct

import numpy as np
import soundfile as sf

from opensv_train.audio_to_midi import MidiNote, convert_audio_to_midi, write_midi


def test_write_midi_creates_format_zero_file_with_note_events(tmp_path):
    output = tmp_path / "test.mid"
    write_midi([MidiNote(0.0, 0.5, 69)], output)

    data = output.read_bytes()
    assert data[:4] == b"MThd"
    header_length, fmt, tracks, ppq = struct.unpack(">IHHH", data[4:14])
    assert (header_length, fmt, tracks, ppq) == (6, 0, 1, 480)
    assert data[14:18] == b"MTrk"
    track_length = struct.unpack(">I", data[18:22])[0]
    track = data[22:22 + track_length]
    assert b"\x90\x45\x60" in track
    assert b"\x80\x45\x00" in track
    assert track.endswith(b"\x00\xff\x2f\x00")


def test_convert_synthetic_singing_tone_to_midi(tmp_path):
    sample_rate = 22050
    seconds = 1.2
    time = np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate
    audio = (0.35 * np.sin(2 * np.pi * 220.0 * time)).astype(np.float32)
    source = tmp_path / "sung-a3.wav"
    output = tmp_path / "midi" / "sung-a3.mid"
    sf.write(source, audio, sample_rate)

    report = convert_audio_to_midi(
        source,
        output,
        pitch_floor=80,
        pitch_ceiling=500,
        voicing_threshold=0.35,
        min_note_ms=40,
        bridge_gap_ms=20,
        smoothing_frames=3,
    )

    assert report["output"] == str(output)
    assert report["notes"] == 1
    assert report["lowest_note"] == 57  # 220 Hz is A3 (MIDI 57).
    assert report["highest_note"] == 57
    assert output.read_bytes().startswith(b"MThd")


def test_silent_audio_fails_with_actionable_message(tmp_path):
    source = tmp_path / "silence.wav"
    sf.write(source, np.zeros(22050, dtype=np.float32), 22050)

    try:
        convert_audio_to_midi(source, tmp_path / "silent.mid")
    except ValueError as error:
        assert "silent" in str(error).lower()
    else:
        raise AssertionError("silent audio should not create a MIDI file")


def test_write_midi_rejects_non_midi_extension(tmp_path):
    try:
        write_midi([MidiNote(0.0, 0.5, 60)], tmp_path / "notes.txt")
    except ValueError as error:
        assert "must end in .mid" in str(error)
    else:
        raise AssertionError("invalid extension should fail")
