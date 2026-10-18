from pathlib import Path

import numpy as np

from opensv_train.timbre import analyze_timbre, summarize_timbre


def _tone(bright: bool, sample_rate: int = 48000) -> np.ndarray:
    time = np.arange(sample_rate // 2, dtype=np.float64) / sample_rate
    signal = (
        0.55 * np.sin(2 * np.pi * 440 * time)
        + 0.25 * np.sin(2 * np.pi * 1100 * time)
        + 0.12 * np.sin(2 * np.pi * 2200 * time)
    )
    if bright:
        signal = signal + 0.28 * np.sin(2 * np.pi * 4500 * time) + 0.18 * np.sin(2 * np.pi * 6500 * time)
    return signal.astype(np.float32)


def test_timbre_profile_is_finite_for_silence_and_empty_input():
    silence = analyze_timbre(np.zeros(2048, dtype=np.float32))
    empty = analyze_timbre(np.zeros(0, dtype=np.float32))
    assert all(np.isfinite(value) for value in silence.values())
    assert all(np.isfinite(value) for value in empty.values())


def test_high_band_diagnostic_detects_spectral_balance_change():
    dark = _tone(False)
    bright = _tone(True)
    p_dark = analyze_timbre(dark)
    p_bright = analyze_timbre(bright)
    assert p_bright["high_mid_ratio_db"] > p_dark["high_mid_ratio_db"]
    assert p_bright["spectral_centroid_hz"] > p_dark["spectral_centroid_hz"]


def test_timbre_summary_compares_target_and_render_and_disclaims_nasality():
    result = summarize_timbre([_tone(False)], [_tone(True)])
    assert result["phrases"] == 1
    assert result["delta"]["high_mid_ratio_db"] > 0
    assert "not" in result["caveat"].lower()
    assert result["reference"]["spectral_centroid_hz"] < result["rendered"]["spectral_centroid_hz"]


def test_brightness_and_preset_controls_are_wired_into_training_and_webui():
    root = Path(__file__).resolve().parents[1]
    cli = (root / "opensv_train" / "__main__.py").read_text("utf-8")
    trainer = (root / "opensv_train" / "train.py").read_text("utf-8")
    loss = (root / "opensv_train" / "losses.py").read_text("utf-8")
    gui = (root / "trainsv_gui.py").read_text("utf-8")

    assert 'command.add_argument("--brightness-weight"' in cli
    assert "brightness_weight=arguments.brightness_weight" in cli
    assert "brightness_weight=config.brightness_weight" in trainer
    assert "self.brightness_weight * brightness" in loss
    assert 'data-id="t_timbre_preset"' in gui
    assert 'data-id="t_brightness_weight"' in gui
    assert "function applyTimbrePreset(name)" in gui
    assert "timbre_diagnostics.json" in cli
    assert '"timbre_diagnostics": timbre_diagnostics' in gui
