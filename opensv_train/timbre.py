"""Reference-relative timbre diagnostics for rendered singing audio.

These measurements describe spectral balance; they are clues for investigating
dark/muffled or nasal-sounding renders, not an objective nasality classifier.
"""
from __future__ import annotations

import numpy as np

PROFILE_KEYS = ("spectral_centroid_hz", "low_mid_ratio_db", "high_mid_ratio_db")
CAVEAT = (
    "Spectral balance is only a heuristic. It cannot determine nasality on its own; "
    "inspect matching vowels and phoneme boundaries by listening."
)


def analyze_timbre(audio, sample_rate: int = 48000) -> dict[str, float]:
    """Measure spectral centroid and low/mid/high band balance in a waveform."""
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim > 1:
        signal = signal.mean(axis=-1)
    signal = signal.reshape(-1)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if signal.size == 0:
        return {key: 0.0 for key in PROFILE_KEYS}

    n_fft, hop = 2048, 512
    if signal.size < n_fft:
        signal = np.pad(signal, (0, n_fft - signal.size))
    padded = np.pad(signal, (n_fft // 2, n_fft // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
    window = np.hanning(n_fft)
    magnitude = np.abs(np.fft.rfft(frames * window[None, :], axis=1))
    mean_magnitude = magnitude.mean(axis=0)
    power = np.mean(magnitude ** 2, axis=0)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    def band_power(low: float, high: float) -> float:
        mask = (frequencies >= low) & (frequencies < high)
        return float(power[mask].sum())

    eps = 1e-12
    low = band_power(250.0, 1000.0)
    mid = band_power(1000.0, 3000.0)
    high = band_power(3000.0, 8000.0)
    total_magnitude = float(mean_magnitude.sum())
    centroid = (
        float((frequencies * mean_magnitude).sum() / total_magnitude)
        if total_magnitude > eps else 0.0
    )
    return {
        "spectral_centroid_hz": centroid,
        "low_mid_ratio_db": float(10.0 * np.log10((low + eps) / (mid + eps))),
        "high_mid_ratio_db": float(10.0 * np.log10((high + eps) / (mid + eps))),
    }


def summarize_timbre(
    reference_audio, rendered_audio, sample_rate: int = 48000,
    phrase_names: list[str] | None = None,
) -> dict:
    """Summarize global and per-phrase spectral balance against matching WAVs."""
    if not reference_audio or not rendered_audio:
        raise ValueError("at least one reference and rendered phrase are required")
    if len(reference_audio) != len(rendered_audio):
        raise ValueError("reference and rendered phrase counts must match")
    if phrase_names is not None and len(phrase_names) != len(reference_audio):
        raise ValueError("phrase names must match the number of audio pairs")
    refs = [analyze_timbre(x, sample_rate) for x in reference_audio]
    outputs = [analyze_timbre(x, sample_rate) for x in rendered_audio]

    def average(profiles):
        return {
            key: float(np.mean([profile[key] for profile in profiles]))
            for key in PROFILE_KEYS
        }

    reference = average(refs)
    rendered = average(outputs)
    delta = {key: float(rendered[key] - reference[key]) for key in PROFILE_KEYS}
    per_phrase = []
    for index, (ref, out) in enumerate(zip(refs, outputs)):
        per_phrase.append({
            "name": phrase_names[index] if phrase_names is not None else f"phrase-{index + 1}",
            "reference": ref,
            "rendered": out,
            "delta": {key: float(out[key] - ref[key]) for key in PROFILE_KEYS},
        })
    if delta["high_mid_ratio_db"] <= -2.0 and delta["spectral_centroid_hz"] <= -250.0:
        interpretation = (
            "The render is globally darker than the reference: upper-band balance and "
            "spectral centroid are both lower. This may sound more muffled, but does not "
            "prove that nasality is the cause."
        )
    elif delta["low_mid_ratio_db"] >= 2.0 and delta["high_mid_ratio_db"] <= -1.0:
        interpretation = (
            "The render has relatively more low/mid energy and less upper-band energy than "
            "the reference. This can be perceived as covered or nasal, but is not a direct "
            "nasality measurement."
        )
    elif delta["high_mid_ratio_db"] <= -2.0:
        interpretation = (
            "The render's 3–8 kHz balance is lower than the reference. Inspect clarity and "
            "consonant detail; this alone does not identify the cause."
        )
    else:
        interpretation = (
            "No large global darkening signal was detected by these three summaries. "
            "If the sound still seems nasal, compare individual vowels and transitions."
        )
    return {
        "phrases": len(reference_audio),
        "reference": reference,
        "rendered": rendered,
        "delta": delta,
        "per_phrase": per_phrase,
        "interpretation": interpretation,
        "caveat": CAVEAT,
    }
