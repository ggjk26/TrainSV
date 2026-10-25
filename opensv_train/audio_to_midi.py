"""Monophonic singing-audio to MIDI conversion for voicebank testing.

Uses Praat's autocorrelation pitch tracker (already required by TrainSV) and
writes a standard MIDI file without requiring an external MIDI package.
The conversion estimates a single melodic line; it is not polyphonic
transcription or score-alignment software.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct

import numpy as np
import parselmouth
import soundfile as sf


@dataclass(frozen=True)
class MidiNote:
    start: float
    end: float
    pitch: int
    velocity: int = 96


def _variable_length(value: int) -> bytes:
    """Encode one non-negative MIDI variable-length quantity."""
    if value < 0:
        raise ValueError("MIDI delta time cannot be negative")
    value = int(value)
    encoded = [value & 0x7F]
    value >>= 7
    while value:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(encoded))


def write_midi(notes: list[MidiNote], output: str | Path, tempo_bpm: float = 120.0) -> Path:
    """Write a format-0 Standard MIDI File with a single monophonic track."""
    if not math.isfinite(tempo_bpm) or not 20.0 <= tempo_bpm <= 300.0:
        raise ValueError("tempo_bpm must be between 20 and 300")
    if not notes:
        raise ValueError("No pitched notes are available to write")

    ppq = 480
    microseconds_per_quarter = round(60_000_000 / tempo_bpm)
    track_name = b"TrainSV pitch test"
    events: list[tuple[int, int, bytes]] = [
        (0, -2, b"\xFF\x03" + bytes([len(track_name)]) + track_name),
        (0, -1, b"\xFF\x51\x03" + microseconds_per_quarter.to_bytes(3, "big")),
    ]
    ticks_per_second = ppq * tempo_bpm / 60.0
    for note in notes:
        if not (0 <= int(note.pitch) <= 127):
            raise ValueError(f"MIDI pitch is out of range: {note.pitch}")
        if not (0 <= note.start < note.end and math.isfinite(note.end)):
            raise ValueError(f"Invalid MIDI note interval: {note.start}–{note.end}")
        start_tick = max(0, round(note.start * ticks_per_second))
        end_tick = max(start_tick + 1, round(note.end * ticks_per_second))
        events.append((start_tick, 1, bytes((0x90, int(note.pitch), int(note.velocity)))))
        events.append((end_tick, 0, bytes((0x80, int(note.pitch), 0))))

    events.sort(key=lambda event: (event[0], event[1]))
    track = bytearray()
    previous_tick = 0
    for tick, _order, event_data in events:
        track.extend(_variable_length(tick - previous_tick))
        track.extend(event_data)
        previous_tick = tick
    track.extend(b"\x00\xFF\x2F\x00")

    midi = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ppq)
    midi += b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
    target = Path(output).expanduser()
    if target.suffix.lower() not in {".mid", ".midi"}:
        raise ValueError("MIDI output path must end in .mid or .midi")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(midi)
    return target


def _median_smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or values.size == 0:
        return values.copy()
    if width % 2 == 0:
        width += 1
    radius = width // 2
    result = values.copy()
    for index in range(values.size):
        left, right = max(0, index - radius), min(values.size, index + radius + 1)
        window = values[left:right]
        valid = window[np.isfinite(window)]
        if valid.size:
            result[index] = float(np.median(valid))
    return result


def _segments(pitches: np.ndarray) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    start = 0
    current = int(pitches[0]) if pitches.size and pitches[0] >= 0 else -1
    for index in range(1, pitches.size + 1):
        value = int(pitches[index]) if index < pitches.size and pitches[index] >= 0 else -1
        if value != current:
            if current >= 0:
                result.append((start, index, current))
            start, current = index, value
    return result


def _clean_note_frames(
    note_frames: np.ndarray,
    max_gap_frames: int,
    min_note_frames: int,
) -> np.ndarray:
    """Bridge short same-note gaps and merge tiny tracker glitches."""
    frames = note_frames.copy()
    if max_gap_frames > 0:
        index = 0
        while index < frames.size:
            if frames[index] >= 0:
                index += 1
                continue
            gap_start = index
            while index < frames.size and frames[index] < 0:
                index += 1
            if (gap_start > 0 and index < frames.size
                    and index - gap_start <= max_gap_frames
                    and frames[gap_start - 1] == frames[index]):
                frames[gap_start:index] = frames[gap_start - 1]

    if min_note_frames > 1:
        # A brief one-frame tracker glitch should not become a separate MIDI note.
        for _ in range(3):
            segments = _segments(frames)
            changed = False
            for position, (start, end, pitch) in enumerate(segments):
                if end - start >= min_note_frames:
                    continue
                left = segments[position - 1] if position > 0 else None
                right = segments[position + 1] if position + 1 < len(segments) else None
                if left is None and right is None:
                    continue
                if left is None:
                    replacement = right[2]
                elif right is None:
                    replacement = left[2]
                elif left[2] == right[2]:
                    replacement = left[2]
                else:
                    replacement = left[2] if abs(pitch - left[2]) <= abs(pitch - right[2]) else right[2]
                frames[start:end] = replacement
                changed = True
            if not changed:
                break
    return frames


def convert_audio_to_midi(
    audio_path: str | Path,
    output_path: str | Path,
    *,
    pitch_floor: float = 65.0,
    pitch_ceiling: float = 1100.0,
    voicing_threshold: float = 0.45,
    min_note_ms: float = 65.0,
    bridge_gap_ms: float = 40.0,
    smoothing_frames: int = 5,
    transpose_semitones: float = 0.0,
    tempo_bpm: float = 120.0,
) -> dict:
    """Extract a sung melody and save quantized note events to a MIDI file."""
    source = Path(audio_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {source}")
    if not (math.isfinite(pitch_floor) and math.isfinite(pitch_ceiling)
            and 30.0 <= pitch_floor < pitch_ceiling <= 5000.0):
        raise ValueError("Pitch range must satisfy 30 <= floor < ceiling <= 5000 Hz")
    if not math.isfinite(voicing_threshold) or not 0.0 < voicing_threshold <= 1.0:
        raise ValueError("Voicing threshold must be in the range (0, 1]")
    if not math.isfinite(min_note_ms) or not 0.0 <= min_note_ms <= 2000.0:
        raise ValueError("Minimum note length must be between 0 and 2000 ms")
    if not math.isfinite(bridge_gap_ms) or not 0.0 <= bridge_gap_ms <= 500.0:
        raise ValueError("Gap bridging must be between 0 and 500 ms")
    if not isinstance(smoothing_frames, int) or not 1 <= smoothing_frames <= 31:
        raise ValueError("Smoothing width must be an integer from 1 to 31 frames")
    if not math.isfinite(transpose_semitones) or not -24.0 <= transpose_semitones <= 24.0:
        raise ValueError("Transpose must be between -24 and +24 semitones")

    try:
        audio, sample_rate = sf.read(str(source), dtype="float32", always_2d=True)
    except Exception as error:  # decoder errors vary by libsndfile version
        raise ValueError(f"Could not decode the audio file (try WAV or FLAC): {error}") from error
    if audio.size == 0 or sample_rate <= 0:
        raise ValueError("The selected audio file is empty")
    mono = np.mean(audio, axis=1, dtype=np.float64).astype(np.float64)
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.any(np.abs(mono) > 1e-7):
        raise ValueError("The audio is silent; no melody can be extracted")

    sound = parselmouth.Sound(mono, sampling_frequency=float(sample_rate))
    pitch = sound.to_pitch_ac(
        time_step=0.01,
        pitch_floor=float(pitch_floor),
        pitch_ceiling=float(pitch_ceiling),
        voicing_threshold=float(voicing_threshold),
        silence_threshold=0.03,
    )
    frequencies = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    times = np.asarray(pitch.xs(), dtype=np.float64)
    frame_step = float(pitch.dx)
    if frequencies.size == 0 or times.size != frequencies.size:
        raise ValueError("The audio was too short for pitch tracking")
    voiced = np.isfinite(frequencies) & (frequencies >= pitch_floor) & (frequencies <= pitch_ceiling)
    semitones = np.full(frequencies.shape, np.nan, dtype=np.float64)
    semitones[voiced] = 69.0 + 12.0 * np.log2(frequencies[voiced] / 440.0)
    semitones = _median_smooth(semitones, smoothing_frames)

    note_frames = np.full(semitones.shape, -1, dtype=np.int16)
    valid = np.isfinite(semitones)
    rounded = np.rint(semitones[valid] + transpose_semitones).astype(np.int64)
    note_frames[valid] = np.clip(rounded, 0, 127).astype(np.int16)
    gap_frames = max(0, round((bridge_gap_ms / 1000.0) / frame_step))
    minimum_frames = max(1, math.ceil((min_note_ms / 1000.0) / frame_step))
    note_frames = _clean_note_frames(note_frames, gap_frames, minimum_frames)
    segments = _segments(note_frames)
    audio_duration = len(mono) / float(sample_rate)
    notes: list[MidiNote] = []
    for start_frame, end_frame, midi_pitch in segments:
        start = max(0.0, float(times[start_frame] - frame_step / 2))
        end = min(audio_duration, float(times[end_frame - 1] + frame_step / 2))
        if end - start < 0.015:
            continue
        notes.append(MidiNote(start, end, midi_pitch))
    if not notes:
        raise ValueError(
            "No stable voiced melody was detected. Try a lower pitch floor, a wider pitch ceiling, "
            "or a less noisy/isolated vocal recording."
        )
    target = write_midi(notes, output_path, tempo_bpm=tempo_bpm)
    midi_values = [note.pitch for note in notes]
    return {
        "output": str(target),
        "notes": len(notes),
        "duration_seconds": audio_duration,
        "lowest_note": min(midi_values),
        "highest_note": max(midi_values),
        "pitch_floor": float(pitch_floor),
        "pitch_ceiling": float(pitch_ceiling),
        "tempo_bpm": float(tempo_bpm),
    }
