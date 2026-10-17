"""F0 extraction presets for training and evaluation WAVs.

These settings affect only how the input audio's fundamental frequency is
measured. They do not replace note labels or automatically train the Pitch
model.
"""
from __future__ import annotations

import math

# profile -> (minimum Hz, maximum Hz, Praat voicing threshold)
F0_PROFILES: dict[str, tuple[float, float, float]] = {
    "standard": (65.0, 1100.0, 0.60),
    "vocaloid": (45.0, 1400.0, 0.45),
    "low": (45.0, 650.0, 0.60),
    "high": (100.0, 1400.0, 0.60),
    "wide": (40.0, 1600.0, 0.60),
}


def resolve_f0_profile(
    profile: str = "standard",
    *,
    floor: float | None = None,
    ceiling: float | None = None,
    voicing_threshold: float | None = None,
) -> tuple[float, float, float]:
    """Resolve a preset and validate numeric overrides."""
    if profile not in F0_PROFILES:
        raise ValueError(f"unknown F0 profile {profile!r}; choose one of {', '.join(F0_PROFILES)}")
    default_floor, default_ceiling, default_threshold = F0_PROFILES[profile]
    resolved_floor = default_floor if floor is None else float(floor)
    resolved_ceiling = default_ceiling if ceiling is None else float(ceiling)
    resolved_threshold = default_threshold if voicing_threshold is None else float(voicing_threshold)
    if not math.isfinite(resolved_floor) or resolved_floor <= 0:
        raise ValueError("F0 floor must be a positive finite frequency in Hz")
    if not math.isfinite(resolved_ceiling) or resolved_ceiling <= resolved_floor:
        raise ValueError("F0 ceiling must be a finite frequency greater than the F0 floor")
    if not math.isfinite(resolved_threshold) or not 0 < resolved_threshold <= 1:
        raise ValueError("F0 voicing threshold must be greater than 0 and at most 1")
    return resolved_floor, resolved_ceiling, resolved_threshold


def f0_cache_tag(floor: float, ceiling: float, voicing_threshold: float) -> str:
    """Stable cache suffix so F0 settings cannot reuse an incompatible contour."""
    floor, ceiling, voicing_threshold = resolve_f0_profile(
        "standard", floor=floor, ceiling=ceiling, voicing_threshold=voicing_threshold
    )
    return f"f0-{floor:g}-{ceiling:g}-v{voicing_threshold:g}"

F0_SMOOTHING_OPTIONS = (0, 3, 5, 7)

# profile -> (median-filter width in 5 ms frames, isolated-octave correction,
#             smooth-melody cutoff Hz, melody-loss weight, fast-motion-loss weight)
# The VOCALOID preset intentionally combines broader tracking with conservative
# track cleanup and a bias toward the smooth contour. This only conditions the
# F0 target used in training; it does not post-process the final waveform.
F0_PROCESSING_PROFILES: dict[str, tuple[int, bool, float, float, float]] = {
    "standard": (0, False, 3.0, 1.0, 1.0),
    "vocaloid": (5, True, 2.5, 1.2, 0.8),
    "low": (0, False, 3.0, 1.0, 1.0),
    "high": (0, False, 3.0, 1.0, 1.0),
    "wide": (0, False, 3.0, 1.0, 1.0),
}


def resolve_f0_processing(
    profile: str = "standard",
    *,
    smoothing_frames: int | None = None,
    correct_octave: bool | None = None,
    pitch_smoothing_hz: float | None = None,
    pitch_melody_weight: float | None = None,
    pitch_motion_weight: float | None = None,
) -> tuple[int, bool, float, float, float]:
    """Resolve the F0 cleanup and Pitch-loss defaults for a tracking preset.

    Explicit values override preset defaults. The VOCALOID profile applies a
    25 ms median filter, fixes isolated octave tracker spikes, and slightly
    emphasizes the smooth melody contour over fast pitch movement.
    """
    if profile not in F0_PROCESSING_PROFILES:
        raise ValueError(f"unknown F0 processing profile {profile!r}")
    defaults = F0_PROCESSING_PROFILES[profile]
    smoothing = defaults[0] if smoothing_frames is None else smoothing_frames
    octave = defaults[1] if correct_octave is None else correct_octave
    cutoff = defaults[2] if pitch_smoothing_hz is None else float(pitch_smoothing_hz)
    melody = defaults[3] if pitch_melody_weight is None else float(pitch_melody_weight)
    motion = defaults[4] if pitch_motion_weight is None else float(pitch_motion_weight)
    if smoothing not in F0_SMOOTHING_OPTIONS:
        raise ValueError(
            f"F0 smoothing must be one of {', '.join(map(str, F0_SMOOTHING_OPTIONS))} frames"
        )
    if not math.isfinite(cutoff) or not 0.0 < cutoff < 100.0:
        raise ValueError("pitch smoothing cutoff must be finite and between 0 and 100 Hz")
    if not math.isfinite(melody) or not math.isfinite(motion) or melody < 0 or motion < 0:
        raise ValueError("Pitch loss weights must be finite, non-negative numbers")
    if melody == 0 and motion == 0:
        raise ValueError("at least one Pitch loss weight must be greater than zero")
    return int(smoothing), bool(octave), cutoff, melody, motion


def process_f0_track(
    values,
    *,
    smoothing_frames: int = 0,
    correct_octave: bool = False,
    shift_semitones: float = 0.0,
):
    """Conservatively clean a tracked F0 contour in log-frequency space.

    smoothing_frames is a median-filter width in 5 ms frames (0 disables
    smoothing). Octave correction only changes isolated one/two-octave errors
    whose voiced neighbours agree closely; it intentionally does not flatten
    sustained octave changes. A semitone shift adjusts the training target,
    not the note labels or the synthesizer's output globally.
    """
    import numpy as np
    from scipy.ndimage import median_filter

    if smoothing_frames not in F0_SMOOTHING_OPTIONS:
        raise ValueError(
            f"F0 smoothing must be one of {', '.join(map(str, F0_SMOOTHING_OPTIONS))} frames"
        )
    shift = float(shift_semitones)
    if not math.isfinite(shift) or not -24.0 <= shift <= 24.0:
        raise ValueError("F0 shift must be finite and between -24 and +24 semitones")

    source = np.asarray(values, dtype=np.float32).reshape(-1)
    result = np.zeros(source.shape, dtype=np.float32)
    voiced = np.isfinite(source) & (source > 0)
    if not voiced.any():
        return result

    # Work on runs of voiced frames separately so a median filter never bleeds
    # across unvoiced consonants, breath, or silence.
    changes = np.diff(np.pad(voiced.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    shift_octaves = shift / 12.0
    for start, stop in zip(starts, stops):
        log_f0 = np.log2(source[start:stop].astype(np.float64))
        if correct_octave and len(log_f0) >= 3:
            original = log_f0.copy()
            for index in range(1, len(original) - 1):
                neighbours = (original[index - 1], original[index + 1])
                # Don't reinterpret real note transitions as octave errors.
                if abs(neighbours[0] - neighbours[1]) > 0.20:
                    continue
                delta = original[index] - (neighbours[0] + neighbours[1]) / 2.0
                octave = int(round(delta))
                if octave and abs(octave) <= 2 and abs(delta - octave) <= 0.16:
                    log_f0[index] = original[index] - octave
        if smoothing_frames:
            log_f0 = median_filter(log_f0, size=smoothing_frames, mode="nearest")
        result[start:stop] = np.exp2(log_f0 + shift_octaves).astype(np.float32)
    return result
