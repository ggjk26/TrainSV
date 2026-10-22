"""Phone-boundary weighting and transition-focused audio sampling helpers."""
from __future__ import annotations

import math

import numpy as np


_SILENCE = {"sil", "pau", "SP", "br", "AP"}


def transition_boundary_frames(phones, phone_frames: list[int]) -> list[int]:
    """Return frame offsets of adjacent non-silence phone boundaries."""
    if len(phones) != len(phone_frames) or any(int(count) < 1 for count in phone_frames):
        raise ValueError("phone frames must contain one positive duration per phone")
    boundaries = []
    offset = 0
    for index, (phone, count) in enumerate(zip(phones, phone_frames)):
        offset += int(count)
        if index + 1 >= len(phones):
            continue
        next_phone = phones[index + 1]
        if phone.symbol not in _SILENCE and next_phone.symbol not in _SILENCE:
            boundaries.append(offset)
    return boundaries


def sample_transition_segment(
    frames: int,
    width: int,
    boundaries: list[int],
    rng,
    *,
    probability: float = 0.5,
    context_frames: int = 8,
) -> tuple[int, int | None]:
    """Choose a random segment or center one on a speech transition.

    Returns (segment_start_frame, absolute_focused_boundary). Focused crops keep
    at least context_frames on both sides of their selected boundary. If the
    phrase is too short or has no usable boundary, sampling falls back to
    uniform random windows.
    """
    if frames < 1 or width < 1:
        raise ValueError("frames and segment width must be positive")
    width = min(int(width), int(frames))
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("transition sampling probability must be between 0 and 1")
    if context_frames < 1:
        raise ValueError("transition sampling context must be positive")
    max_start = frames - width
    can_focus = width >= 2 * context_frames and probability > 0
    candidates = [
        int(boundary) for boundary in boundaries
        if context_frames <= int(boundary) <= frames - context_frames
    ] if can_focus else []
    if candidates and rng.random() < probability:
        boundary = rng.choice(candidates)
        offset = rng.randint(context_frames, width - context_frames)
        start = max(0, min(max_start, boundary - offset))
        if start + context_frames <= boundary <= start + width - context_frames:
            return start, boundary
    return rng.randint(0, max_start), None


def transition_frame_weights(
    phones, phone_frames: list[int], *, weight: float = 1.0, radius: int = 2
) -> np.ndarray:
    """Return frame weights that optionally emphasize speech-phone handoffs.

    Only non-silence/non-breath boundaries are considered. A weight of 1.0 is
    an exact no-op; this weights the training loss rather than smoothing output.
    """
    if not math.isfinite(weight) or weight < 1.0:
        raise ValueError("transition weight must be finite and at least 1.0")
    if radius < 0:
        raise ValueError("transition radius must be non-negative")
    boundaries = transition_boundary_frames(phones, phone_frames)
    result = np.ones(sum(phone_frames), dtype=np.float32)
    if weight == 1.0 or radius == 0:
        return result
    for boundary in boundaries:
        begin, end = max(0, boundary - radius), min(len(result), boundary + radius)
        result[begin:end] = np.maximum(result[begin:end], np.float32(weight))
    return result
