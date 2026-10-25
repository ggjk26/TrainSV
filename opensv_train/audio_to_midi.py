"""Monophonic audio-to-MIDI conversion for voicebank pitch checks.

The extractor keeps F0 in continuous semitone space until after voiced-run
smoothing and short octave-error repair. It is still melody transcription, not
waveform reconstruction or polyphonic transcription.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import parselmouth


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


def _true_runs(mask: np.ndarray):
    """Yield [start, end) spans where a boolean mask is true."""
    mask = np.asarray(mask, dtype=bool)
    if not len(mask):
        return
    padded = np.concatenate(([False], mask, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return zip(starts.tolist(), ends.tolist())


def _median_filter_voiced(values: np.ndarray, voiced: np.ndarray, width: int) -> np.ndarray:
    """Median-filter each voiced island separately; never smear pitch over silence."""
    if width <= 1:
        return values.copy()
    result = values.copy()
    pad = width // 2
    for start, end in _true_runs(voiced):
        segment = values[start:end]
        if len(segment) <= 1:
            continue
        padded = np.pad(segment, (pad, pad), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, width)
        result[start:end] = np.median(windows, axis=-1)
    return result


def _repair_octave_spikes(
    values: np.ndarray,
    voiced: np.ndarray,
    strengths: np.ndarray | None,
    frame_step: float,
) -> tuple[np.ndarray, int]:
    """Repair only short, isolated ~12-semitone excursions that return to baseline."""
    source = values.copy()
    result = values.copy()
    corrected = 0
    max_frames = max(1, int(round(0.06 / frame_step)))

    for seg_start, seg_end in _true_runs(voiced):
        i = seg_start + 1
        while i < seg_end - 1:
            left = source[max(seg_start, i - 3):i]
            if len(left) < 2:
                i += 1
                continue
            left_base = float(np.median(left))
            delta = float(source[i] - left_base)
            if abs(abs(delta) - 12.0) > 1.5:
                i += 1
                continue
            direction = 1 if delta > 0 else -1

            def is_octave_outlier(value: float) -> bool:
                offset = value - left_base
                return (
                    (1 if offset > 0 else -1) == direction
                    and abs(abs(offset) - 12.0) <= 1.8
                )

            j = i
            while j < seg_end and is_octave_outlier(float(source[j])):
                j += 1
            run_length = j - i
            right = source[j:min(seg_end, j + 3)] if j < seg_end else np.array([])
            if run_length > max_frames or len(right) < 2:
                i = max(j, i + 1)
                continue

            right_base = float(np.median(right))
            if abs(right_base - left_base) > 2.5:
                i = j
                continue

            # A markedly stronger periodicity estimate may be a real short note,
            # so confidence is used as a conservative veto, not as a hard voicing gate.
            if strengths is not None:
                left_strength = strengths[max(seg_start, i - 3):i]
                spike_strength = strengths[i:j]
                right_strength = strengths[j:min(seg_end, j + 3)]
                left_strength = left_strength[np.isfinite(left_strength)]
                spike_strength = spike_strength[np.isfinite(spike_strength)]
                right_strength = right_strength[np.isfinite(right_strength)]
                if len(spike_strength) and len(left_strength) and len(right_strength):
                    neighbours = float(np.mean(np.concatenate((left_strength, right_strength))))
                    if float(np.mean(spike_strength)) > neighbours + 0.25:
                        i = j
                        continue

            result[i:j] = np.linspace(left_base, right_base, run_length + 2)[1:-1]
            corrected += 1
            i = j
    return result, corrected


def _runs_from_labels(labels: np.ndarray) -> list[list[int]]:
    """Return mutable [start, end, label] runs for a label contour."""
    runs: list[list[int]] = []
    i = 0
    while i < len(labels):
        j = i + 1
        while j < len(labels) and int(labels[j]) == int(labels[i]):
            j += 1
        runs.append([i, j, int(labels[i])])
        i = j
    return runs


def _merge_short_note_runs(
    labels: np.ndarray,
    continuous_pitch: np.ndarray,
    times: np.ndarray,
    frame_step: float,
    min_note_ms: float,
) -> np.ndarray:
    """Absorb too-short quantization flicker into a plausible neighbour instead of making rests."""
    runs = _runs_from_labels(labels)
    minimum_seconds = min_note_ms / 1000.0
    i = 0
    while i < len(runs):
        start, end, pitch = runs[i]
        if pitch < 0 or (float(times[end - 1] - times[start] + frame_step) >= minimum_seconds):
            i += 1
            continue

        previous = runs[i - 1] if i > 0 else None
        following = runs[i + 1] if i + 1 < len(runs) else None
        prev_pitch = previous[2] if previous is not None and previous[2] >= 0 else -1
        next_pitch = following[2] if following is not None and following[2] >= 0 else -1

        if prev_pitch >= 0 and next_pitch >= 0:
            if prev_pitch == next_pitch:
                replacement = prev_pitch
            else:
                samples = continuous_pitch[start:end]
                samples = samples[np.isfinite(samples)]
                if len(samples):
                    left_cost = float(np.mean(np.abs(samples - prev_pitch)))
                    right_cost = float(np.mean(np.abs(samples - next_pitch)))
                else:
                    left_cost = right_cost = math.inf
                if abs(left_cost - right_cost) <= 0.15:
                    left_length = previous[1] - previous[0]
                    right_length = following[1] - following[0]
                    replacement = prev_pitch if left_length >= right_length else next_pitch
                else:
                    replacement = prev_pitch if left_cost < right_cost else next_pitch
        elif prev_pitch >= 0:
            replacement = prev_pitch
        elif next_pitch >= 0:
            replacement = next_pitch
        else:
            replacement = -1

        runs[i][2] = replacement
        # Coalesce adjacent equal labels immediately so the next duration check sees
        # the actual merged note, not two pieces of the same note.
        if i > 0 and runs[i - 1][2] == runs[i][2]:
            runs[i - 1][1] = runs[i][1]
            runs.pop(i)
            i = max(0, i - 1)
            continue
        if i + 1 < len(runs) and runs[i + 1][2] == runs[i][2]:
            runs[i][1] = runs[i + 1][1]
            runs.pop(i + 1)
            i = max(0, i - 1)
            continue
        i += 1

    result = np.full(len(labels), -1, dtype=np.int16)
    for start, end, pitch in runs:
        result[start:end] = pitch
    return result


def _stabilize_note_labels(
    continuous_pitch: np.ndarray,
    voiced: np.ndarray,
    frame_step: float,
    *,
    note_change_ms: float = 120.0,
    switch_margin: float = 0.72,
) -> tuple[np.ndarray, int]:
    """Convert an F0 contour to stable MIDI pitches, suppressing short expressive excursions.

    A candidate pitch must move at least switch_margin semitones away from the
    current note's center and remain there for note_change_ms before becoming a
    new MIDI note. Once confirmed, the new note is assigned from the first frame of
    the sustained excursion so confirmation latency does not shift the note onset.
    During a slow glide, the pending target follows the contour, avoiding a cascade
    of short intermediate semitone notes.
    """
    if not math.isfinite(note_change_ms) or not 20.0 <= note_change_ms <= 500.0:
        raise ValueError("pitch-change confirmation must be between 20 and 500 ms")
    if not math.isfinite(frame_step) or frame_step <= 0:
        raise ValueError("pitch tracker returned invalid frame times")

    values = np.asarray(continuous_pitch, dtype=np.float64)
    voiced = np.asarray(voiced, dtype=bool)
    if len(values) != len(voiced):
        raise ValueError("pitch contour and voiced mask must have the same length")

    labels = np.full(len(values), -1, dtype=np.int16)
    confirm_frames = max(1, int(math.ceil((note_change_ms / 1000.0) / frame_step - 1e-9)))
    suppressed_excursions = 0

    for start, end in _true_runs(voiced):
        valid_start = start
        while valid_start < end and not math.isfinite(float(values[valid_start])):
            valid_start += 1
        if valid_start >= end:
            continue

        current = int(np.clip(np.rint(values[valid_start]), 0, 127))
        labels[start:valid_start] = current
        labels[valid_start:end] = current
        pending_start = -1
        pending_direction = 0
        pending_target = current

        for index in range(valid_start + 1, end):
            value = float(values[index])
            if not math.isfinite(value):
                if pending_start >= 0:
                    suppressed_excursions += 1
                pending_start = -1
                pending_direction = 0
                continue

            candidate = int(np.clip(np.rint(value), 0, 127))
            direction = 0
            if candidate > current and value >= current + switch_margin:
                direction = 1
            elif candidate < current and value <= current - switch_margin:
                direction = -1

            if direction == 0:
                if pending_start >= 0:
                    suppressed_excursions += 1
                pending_start = -1
                pending_direction = 0
                pending_target = current
                continue

            if pending_start < 0 or direction != pending_direction:
                if pending_start >= 0:
                    suppressed_excursions += 1
                pending_start = index
                pending_direction = direction
                pending_target = candidate
            elif direction > 0:
                pending_target = max(pending_target, candidate)
            else:
                pending_target = min(pending_target, candidate)

            if index - pending_start + 1 >= confirm_frames:
                confirmed = int(np.clip(pending_target, 0, 127))
                if confirmed != current:
                    # Preserve the onset of the confirmed change, not the later
                    # frame on which its duration becomes long enough.
                    labels[pending_start:index + 1] = confirmed
                    current = confirmed
                else:
                    suppressed_excursions += 1
                pending_start = -1
                pending_direction = 0
                pending_target = current

        if pending_start >= 0:
            suppressed_excursions += 1

    return labels, suppressed_excursions


def _extract_note_events(
    times: np.ndarray,
    frequencies: np.ndarray,
    *,
    a4_hz: float = 440.0,
    min_note_ms: float = 70.0,
    median_window: int = 5,
    note_change_ms: float = 120.0,
    transpose_semitones: int = 0,
    strengths: np.ndarray | None = None,
    octave_correction: bool = True,
    diagnostics: dict[str, float | int] | None = None,
) -> list[tuple[float, float, int]]:
    if not math.isfinite(a4_hz) or not 400.0 <= a4_hz <= 480.0:
        raise ValueError("A4 reference must be between 400 and 480 Hz")
    if not math.isfinite(min_note_ms) or not 20.0 <= min_note_ms <= 1000.0:
        raise ValueError("minimum note duration must be between 20 and 1000 ms")
    if median_window < 1 or median_window > 21 or median_window % 2 == 0:
        raise ValueError("median window must be an odd number from 1 to 21")
    if not math.isfinite(note_change_ms) or not 20.0 <= note_change_ms <= 500.0:
        raise ValueError("pitch-change confirmation must be between 20 and 500 ms")
    if not -24 <= transpose_semitones <= 24:
        raise ValueError("transpose must be between -24 and +24 semitones")
    if len(times) != len(frequencies) or len(times) < 2:
        raise ValueError("audio is too short to estimate a melody")

    times = np.asarray(times, dtype=np.float64)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("pitch tracker returned invalid frame times")
    frame_step = float(np.median(np.diff(times)))
    if not math.isfinite(frame_step) or frame_step <= 0:
        raise ValueError("pitch tracker returned invalid frame times")

    voiced = np.isfinite(frequencies) & (frequencies > 0)
    strength_values: np.ndarray | None = None
    if strengths is not None:
        strength_values = np.asarray(strengths, dtype=np.float64)
        if len(strength_values) != len(frequencies):
            raise ValueError("pitch strengths must match the frequency contour")

    continuous = np.full(len(frequencies), np.nan, dtype=np.float64)
    continuous[voiced] = 69.0 + 12.0 * np.log2(frequencies[voiced] / a4_hz)

    octave_spikes = 0
    if octave_correction:
        continuous, octave_spikes = _repair_octave_spikes(
            continuous, voiced, strength_values, frame_step
        )
    continuous = _median_filter_voiced(continuous, voiced, median_window)
    continuous_transposed = continuous + float(transpose_semitones)

    # Stabilize note identity before building MIDI events. This avoids making
    # every vibrato crossing a semitone threshold into a separate MIDI note.
    labels, suppressed_excursions = _stabilize_note_labels(
        continuous_transposed,
        voiced,
        frame_step,
        note_change_ms=note_change_ms,
        switch_margin=0.72,
    )

    # Fill only tiny gaps when the same note is present on both sides (for example,
    # a brief consonant closure). Do not bridge actual rests or different notes.
    max_gap = max(1, round(0.03 / frame_step))
    i = 0
    while i < len(labels):
        if labels[i] >= 0:
            i += 1
            continue
        j = i
        while j < len(labels) and labels[j] < 0:
            j += 1
        if i > 0 and j < len(labels) and j - i <= max_gap and labels[i - 1] == labels[j] >= 0:
            labels[i:j] = labels[i - 1]
        i = j

    labels = _merge_short_note_runs(
        labels, continuous_transposed, times, frame_step, min_note_ms
    )
    if diagnostics is not None:
        diagnostics["octave_spikes_corrected"] = octave_spikes
        diagnostics["short_pitch_excursions_suppressed"] = suppressed_excursions
        diagnostics["voiced_frames_percent"] = round(float(np.mean(voiced) * 100.0), 1)

    def boundary(index: int) -> float:
        if index <= 0:
            return max(0.0, float(times[0] - frame_step / 2))
        if index >= len(labels):
            return max(0.0, float(times[-1] + frame_step / 2))
        default = max(0.0, float(times[index] - frame_step / 2))
        previous_pitch = int(labels[index - 1])
        next_pitch = int(labels[index])
        if previous_pitch < 0 or next_pitch < 0 or previous_pitch == next_pitch:
            return default
        left_value = float(continuous_transposed[index - 1])
        right_value = float(continuous_transposed[index])
        if not (math.isfinite(left_value) and math.isfinite(right_value)) or right_value == left_value:
            return default
        threshold = previous_pitch + (0.72 if next_pitch > previous_pitch else -0.72)
        if (left_value - threshold) * (right_value - threshold) > 0:
            return default
        fraction = (threshold - left_value) / (right_value - left_value)
        if not 0.0 <= fraction <= 1.0:
            return default
        return max(0.0, float(times[index - 1] + fraction * (times[index] - times[index - 1])))

    events: list[tuple[float, float, int]] = []
    i = 0
    while i < len(labels):
        pitch = int(labels[i])
        if pitch < 0:
            i += 1
            continue
        j = i + 1
        while j < len(labels) and int(labels[j]) == pitch:
            j += 1
        start_time = boundary(i)
        end_time = boundary(j)
        if end_time > start_time and (end_time - start_time) * 1000.0 >= min_note_ms:
            events.append((start_time, end_time, pitch))
        i = j

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
    note_change_ms: float = 120.0,
    a4_hz: float = 440.0,
    transpose_semitones: int = 0,
    tempo_bpm: float = 120.0,
    octave_correction: bool = True,
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
    if not math.isfinite(tempo_bpm) or not 20.0 <= tempo_bpm <= 300.0:
        raise ValueError("tempo must be between 20 and 300 BPM")

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
    selected = pitch.selected_array
    frequencies = np.asarray(selected["frequency"], dtype=np.float64)
    try:
        strengths = np.asarray(selected["strength"], dtype=np.float64)
    except (KeyError, IndexError, TypeError, ValueError):
        strengths = None

    diagnostics: dict[str, float | int] = {}
    events = _extract_note_events(
        np.asarray(pitch.xs(), dtype=np.float64),
        frequencies,
        a4_hz=a4_hz,
        min_note_ms=min_note_ms,
        median_window=median_window,
        note_change_ms=note_change_ms,
        transpose_semitones=transpose_semitones,
        strengths=strengths,
        octave_correction=octave_correction,
        diagnostics=diagnostics,
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
        **diagnostics,
    }
