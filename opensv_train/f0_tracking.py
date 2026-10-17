"""F0 extraction with conservative recovery for weakly voiced synthesized audio.

A valid waveform may be silent or contain only voiceless consonants; those
samples get zero F0 with no voicing mask and remain available for acoustic
training. A low-threshold retry is used only when normal tracking yields no
reliable voiced run.
"""
from __future__ import annotations

import math

import numpy as np

FRAME_SECONDS = 0.005
SAMPLE_RATE = 48000


def _validate_f0_settings(floor: float, ceiling: float, voicing_threshold: float) -> None:
    if not np.isfinite(floor) or not np.isfinite(ceiling) or floor <= 0 or ceiling <= floor:
        raise ValueError("F0 range must satisfy 0 < floor < ceiling (Hz)")
    if not np.isfinite(voicing_threshold) or not 0.0 <= voicing_threshold <= 1.0:
        raise ValueError("F0 voicing threshold must be between 0 and 1")


def _empty_f0_track(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(audio).reshape(-1).size
    frame_count = max(1, int(math.ceil(samples / (SAMPLE_RATE * FRAME_SECONDS))))
    return (
        np.arange(frame_count, dtype=np.float64) * FRAME_SECONDS,
        np.zeros(frame_count, dtype=np.float32),
    )


def _stable_f0_mask(values: np.ndarray, minimum_run: int = 3) -> np.ndarray:
    """Mark stable local F0 runs, allowing real glides across a longer segment."""
    source = np.asarray(values, dtype=np.float32).reshape(-1)
    valid = np.isfinite(source) & (source > 0)
    mask = np.zeros(source.shape, dtype=bool)
    changes = np.diff(np.pad(valid.astype(np.int8), (1, 1)))
    starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    for start, stop in zip(starts, stops):
        if stop - start < minimum_run:
            continue
        semitones = 12.0 * np.log2(source[start:stop].astype(np.float64))
        # A real sung line may move by more than four semitones over a whole
        # syllable. Detect implausible frame-to-frame jumps rather than rejecting
        # every long glide or note change across that run.
        if semitones.size <= 1 or float(np.max(np.abs(np.diff(semitones)))) <= 8.0:
            mask[start:stop] = True
    return mask


def extract_f0(
    audio: np.ndarray,
    floor: float = 65.0,
    ceiling: float = 1100.0,
    *,
    voicing_threshold: float = 0.60,
) -> tuple[np.ndarray, np.ndarray]:
    import parselmouth

    _validate_f0_settings(floor, ceiling, voicing_threshold)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("cannot extract F0 from empty or non-finite decoded audio")
    if float(np.max(np.abs(audio))) <= 1e-8:
        return _empty_f0_track(audio)

    def track(track_floor: float, track_ceiling: float, threshold: float):
        pitch = parselmouth.Sound(audio.astype(np.float64), SAMPLE_RATE).to_pitch_ac(
            time_step=FRAME_SECONDS,
            voicing_threshold=threshold,
            pitch_floor=track_floor,
            pitch_ceiling=track_ceiling,
        )
        times = np.asarray(pitch.xs(), dtype=np.float64).reshape(-1)
        values = np.nan_to_num(
            np.asarray(pitch.selected_array["frequency"], dtype=np.float32).reshape(-1),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        if not times.size or not values.size or times.size != values.size:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float32)
        values[values < 0] = 0
        return times, values

    try:
        times, values = track(floor, ceiling, voicing_threshold)
    except (parselmouth.PraatError, RuntimeError, ValueError, FloatingPointError):
        return _empty_f0_track(audio)
    if not times.size:
        return _empty_f0_track(audio)

    # Drop isolated/unreliable tracker blips, but preserve real note transitions.
    # If reliable coverage is sparse, retry with a broader range and lower
    # voicing threshold; fill only missing frames with stable retry estimates.
    stable = _stable_f0_mask(values)
    coverage = float(np.count_nonzero(stable)) / max(len(stable), 1)
    values = np.where(stable, values, 0.0).astype(np.float32, copy=False)
    if coverage < 0.15:
        retry_floor = max(25.0, min(float(floor), 35.0))
        retry_ceiling = min(2000.0, max(float(ceiling), 1800.0))
        retry_threshold = max(0.10, min(0.30, float(voicing_threshold) * 0.5))
        try:
            retry_times, retry_values = track(
                retry_floor, retry_ceiling, retry_threshold
            )
        except (parselmouth.PraatError, RuntimeError, ValueError, FloatingPointError):
            retry_times = np.array([], dtype=np.float64)
            retry_values = np.array([], dtype=np.float32)
        retry_stable = _stable_f0_mask(retry_values) if retry_times.size else np.array([], dtype=bool)
        if retry_stable.any():
            if retry_times.shape == times.shape and np.allclose(retry_times, times, atol=1e-7):
                fill = retry_stable & (values <= 0)
                values[fill] = retry_values[fill]
            else:
                nearest = np.clip(np.searchsorted(retry_times, times), 0, len(retry_times) - 1)
                close = np.abs(retry_times[nearest] - times) <= FRAME_SECONDS * 0.5
                recovered = retry_stable[nearest] & close & (values <= 0)
                values[recovered] = retry_values[nearest[recovered]]
    return times, values
