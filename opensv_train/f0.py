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
