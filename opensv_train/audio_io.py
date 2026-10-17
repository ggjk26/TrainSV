"""Robust WAV decoding shared by LAB generation and training.

Use libsndfile first, then fall back to SciPy for PCM/float WAV variants that
the installed libsndfile build may not support. Valid silence/unvoiced audio
is accepted; F0 availability is evaluated separately.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile


def _to_float32(samples: np.ndarray) -> np.ndarray:
    """Normalize SciPy's integer/float WAV arrays to float32."""
    samples = np.asarray(samples)
    if samples.dtype.kind == "f":
        return samples.astype(np.float32, copy=False)
    if samples.dtype.kind == "u":
        info = np.iinfo(samples.dtype)
        midpoint = float(info.max + 1) / 2.0
        return ((samples.astype(np.float64) - midpoint) / midpoint).astype(np.float32)
    if samples.dtype.kind == "i":
        scale = float(1 << (np.iinfo(samples.dtype).bits - 1))
        return (samples.astype(np.float64) / scale).astype(np.float32)
    raise ValueError(f"unsupported decoded WAV sample type: {samples.dtype}")


def read_wav_frames(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode a WAV to float32 [frames, channels] plus its sample rate."""
    path = Path(path)
    primary_error: Exception | None = None
    try:
        samples, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        samples = np.asarray(samples, dtype=np.float32)
    except (OSError, RuntimeError, ValueError) as error:
        primary_error = error
        try:
            from scipy.io import wavfile

            rate, samples = wavfile.read(str(path), mmap=False)
            samples = _to_float32(np.asarray(samples))
            if samples.ndim == 1:
                samples = samples[:, None]
        except Exception as fallback_error:
            raise ValueError(
                f"WAV decoding failed (libsndfile: {primary_error}; "
                f"SciPy WAV reader: {fallback_error})"
            ) from fallback_error

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or samples.shape[1] <= 0:
        raise ValueError(f"decoded WAV has invalid channel dimensions: {samples.shape}")
    if not isinstance(rate, (int, np.integer)) or int(rate) <= 0:
        raise ValueError(f"decoded WAV has invalid sample rate: {rate}")
    if samples.shape[0] <= 0:
        raise ValueError("WAV contains no decoded audio samples")
    if not np.isfinite(samples).all():
        raise ValueError("decoded WAV contains non-finite audio samples")
    return samples, int(rate)
