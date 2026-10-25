"""Monophonic audio-to-MIDI conversion for voicebank pitch checks.

This is a practical melody extractor, not polyphonic transcription: it tracks the
dominant voiced fundamental with Praat autocorrelation and writes quantized notes
to a standard MIDI type-0 file that can be imported into Synthesizer V.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import parselmouth
from scipy.ndimage import median_filter


def _vlq(value: int) -> bytes:
    """Encode a non-negative MIDI variable-length quantity."""
    if value < 0:
        raise ValueError("MIDI delta time cannot be negative")
    value = int(value)
    buffer = value & 0x7F
    while (value := value >> 7):
        buffer <<= 8
        buffer |= (value & 0x7F) | 0x80
    out = bytearray()
    while True:
        out.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(out)


def _midi_bytes(notes: list[tuple[float, float, int]], *, tempo_bpm: float = 120.0) -> bytes:
    """Build a standards-compliant single-track MIDI file from (start, end, pitch)."""
    if not math.isfinite(tempo_bpm) or not 20.0 <= tempo_bpm <= 300.0:
        raise ValueError("tempo must be between 20 and 300 BPM")
    ppq = 480
    micros_per_quarter = round(60_000_000 / tempo_bpm)
    events: list[tuple[int, int, bytes]] = [
        (0, 0, b"\xff\x51\x03" + micros_per_quarter.to_bytes(3, "big")),
        (0, 1, b"\xff\x58\x04\x04\x02\x18\x08"),
    ]
    for start, end, pitch in notes:
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            continue
        if not 0 <= int(pitch) <= 127:
            continue
        start_tick = max(0, round(start * tempo_bpm * ppq / 60.0))
        end_tick = max(start_tick + 1, round(end * tempo_bpm * ppq / 60.0))
        # At a shared tick, note-offs precede note-ons to avoid stuck/overlapping notes.
        events.append((start_tick, 3, bytes((0x90, int(pitch), 96))))
        events.append((end_tick, 2, bytes((0x80, int(pitch), 0))))
    events.sort(key=lambda item: (item[0], item[1]))
    track = bytearray()
    previous_tick = 0
    for tick, _order, payload in events:
        track.extend(_vlq(tick - previous_tick))
        track.extend(payload)
        previous_tick = tick
    track.extend(b"\x00\xff\x2f\x00")
    return b"MThd" + struct.pack(">IHHH", 6, 0, 1, ppq) + b"MTrk" + struct.pack(">I", len(track)) + track


def _extract_note_events(
    times: np.ndarray,
    frequencies: np.ndarray,
    *,
    a4_hz: float = 440.0,
    min_note_ms: float = 70.0,
    median_window: int = 5,
    transpose_semitones: int = 0,
) -> list[tuple[float, float, int]]:
    if not math.isfinite(a4_hz) or not 400.0 <= a4_hz <= 480.0:
        raise ValueError("A4 reference must be between 400 and 480 Hz")
    if not math.isfinite(min_note_ms) or not 20.0 <= min_note_ms <= 1000.0:
        raise ValueError("minimum note duration must be between 20 and 1000 ms")
    if median_window < 1 or median_window > 21 or median_window % 2 == 0:
        raise ValueError("median window must be an odd number from 1 to 21")
    if not -24 <= transpose_semitones <= 24:
        raise ValueError("transpose must be between -24 and +24 semitones")
    if len(times) != len(frequencies) or len(times) < 2:
        raise ValueError("audio is too short to estimate a melody")

    times = np.asarray(times, dtype=np.float64)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    dt = float(np.median(np.diff(times)))
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("pitch tracker returned invalid frame times")
    voiced = np.isfinite(frequencies) & (frequencies > 0)
    midi_float = np.zeros(len(frequencies), dtype=np.float64)
    midi_float[voiced] = 69.0 + 12.0 * np.log2(frequencies[voiced] / a4_hz)
    midi = np.zeros(len(frequencies), dtype=np.int16)
    midi[voiced] = np.clip(np.rint(midi_float[voiced]), 0, 127).astype(np.int16)

    # Smooth short tracking flicker without letting unvoiced frames become notes.
    if median_window > 1:
        filtered = median_filter(midi.astype(np.float64), size=median_window, mode="nearest")
        midi[voiced] = np.rint(filtered[voiced]).astype(np.int16)

    # Fill tiny gaps only when the same note is present on both sides.
    max_gap = max(1, round(0.03 / dt))
    i = 0
    while i < len(midi):
        if midi[i] != 0:
            i += 1
            continue
        j = i
        while j < len(midi) and midi[j] == 0:
            j += 1
        if i > 0 and j < len(midi) and j - i <= max_gap and midi[i - 1] == midi[j]:
            midi[i:j] = midi[i - 1]
        i = j

    events: list[tuple[float, float, int]] = []
    start = 0
    while start < len(midi):
        pitch = int(midi[start])
        if pitch == 0:
            start += 1
            continue
        end = start + 1
        while end < len(midi) and int(midi[end]) == pitch:
            end += 1
        start_time = max(0.0, float(times[start] - dt / 2))
        end_time = float(times[end - 1] + dt / 2)
        if (end_time - start_time) * 1000.0 >= min_note_ms:
            final_pitch = pitch + int(transpose_semitones)
            if 0 <= final_pitch <= 127:
                events.append((start_time, end_time, final_pitch))
        start = end

    if not events:
        raise ValueError(
            "No stable pitched notes were detected. Try a clearer monophonic recording, "
            "adjust the pitch range, or lower the minimum note duration."
        )
    return events


def convert_audio_to_midi(
    source: str | Path,
    output: str | Path,
    *,
    pitch_floor: float = 65.0,
    pitch_ceiling: float = 1100.0,
    voicing_threshold: float = 0.45,
    min_note_ms: float = 70.0,
    median_window: int = 5,
    a4_hz: float = 440.0,
    transpose_semitones: int = 0,
    tempo_bpm: float = 120.0,
) -> dict[str, float | int | str]:
    """Extract a monophonic F0 line and save it as a standard MIDI file."""
    source_path = Path(source).expanduser()
    output_path = Path(output).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"audio input does not exist: {source_path}")
    if output_path.suffix.lower() not in {".mid", ".midi"}:
        raise ValueError("output file must use the .mid or .midi extension")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("MIDI output must not overwrite the source audio")
    if not (math.isfinite(pitch_floor) and math.isfinite(pitch_ceiling) and 20 <= pitch_floor < pitch_ceiling <= 5000):
        raise ValueError("pitch range must satisfy 20 <= floor < ceiling <= 5000 Hz")
    if not math.isfinite(voicing_threshold) or not 0.05 <= voicing_threshold <= 1.0:
        raise ValueError("voicing threshold must be between 0.05 and 1.0")

    sound = parselmouth.Sound(str(source_path))
    if sound.get_total_duration() < 0.08:
        raise ValueError("audio is too short; provide at least 80 ms of audio")
    pitch = sound.to_pitch_ac(
        time_step=0.01,
        pitch_floor=float(pitch_floor),
        pitch_ceiling=float(pitch_ceiling),
        silence_threshold=0.03,
        voicing_threshold=float(voicing_threshold),
        octave_cost=0.01,
        octave_jump_cost=0.35,
        voiced_unvoiced_cost=0.14,
    )
    frequencies = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    times = np.asarray(pitch.xs(), dtype=np.float64)
    events = _extract_note_events(
        times,
        frequencies,
        a4_hz=a4_hz,
        min_note_ms=min_note_ms,
        median_window=median_window,
        transpose_semitones=transpose_semitones,
    )
    midi = _midi_bytes(events, tempo_bpm=tempo_bpm)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(midi)
    pitches = [event[2] for event in events]
    return {
        "source": str(source_path),
        "output": str(output_path),
        "notes": len(events),
        "duration_seconds": round(max(end for _, end, _ in events), 3),
        "lowest_midi_note": min(pitches),
        "highest_midi_note": max(pitches),
        "tempo_bpm": float(tempo_bpm),
    }
