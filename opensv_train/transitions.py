"""Phone-boundary weighting helpers for acoustic training."""
from __future__ import annotations

import math

import numpy as np


def transition_frame_weights(
    phones, phone_frames: list[int], *, weight: float = 1.0, radius: int = 2
) -> np.ndarray:
    """Return frame weights that optionally emphasize speech-phone handoffs.

    Only non-silence/non-breath boundaries are considered. A weight of 1.0 is
    an exact no-op; this weights training targets rather than smoothing
    generated audio, so plosives and fricative releases remain sharp.
    """
    if not math.isfinite(weight) or weight < 1.0:
        raise ValueError("transition weight must be finite and at least 1.0")
    if radius < 0:
        raise ValueError("transition radius must be non-negative")
    if len(phones) != len(phone_frames) or any(int(count) < 1 for count in phone_frames):
        raise ValueError("phone frames must contain one positive duration per phone")
    result = np.ones(sum(phone_frames), dtype=np.float32)
    if weight == 1.0 or radius == 0 or len(phones) < 2:
        return result
    boundary = 0
    silent = {"sil", "pau", "SP", "br", "AP"}
    for index, (phone, count) in enumerate(zip(phones, phone_frames)):
        boundary += int(count)
        if index + 1 >= len(phones):
            continue
        next_phone = phones[index + 1]
        if phone.symbol in silent or next_phone.symbol in silent:
            continue
        begin, end = max(0, boundary - radius), min(len(result), boundary + radius)
        result[begin:end] = np.maximum(result[begin:end], np.float32(weight))
    return result
