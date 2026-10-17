from __future__ import annotations

import pytest

from opensv_train.f0 import F0_PROFILES, f0_cache_tag, resolve_f0_profile, resolve_f0_processing, resolve_pitch_loss_options


def test_f0_profiles_have_expected_ranges():
    assert tuple(F0_PROFILES) == ("standard", "vocaloid", "low", "high", "wide")
    assert resolve_f0_profile("standard") == (65.0, 1100.0, 0.60)
    assert resolve_f0_profile("vocaloid") == (45.0, 1400.0, 0.45)
    assert resolve_f0_profile("low") == (45.0, 650.0, 0.60)
    assert resolve_f0_profile("high") == (100.0, 1400.0, 0.60)
    assert resolve_f0_profile("wide") == (40.0, 1600.0, 0.60)


def test_f0_presets_only_control_f0_cleanup():
    assert resolve_f0_processing("standard") == (0, False)
    assert resolve_f0_processing("vocaloid") == (5, True)
    assert resolve_f0_processing("low") == (0, False)


def test_pitch_loss_defaults_and_overrides_are_independent_of_f0_profile():
    assert resolve_pitch_loss_options() == (3.0, 1.0, 1.0)
    assert resolve_pitch_loss_options(
        pitch_smoothing_hz=3.5,
        pitch_melody_weight=1.2,
        pitch_motion_weight=0.8,
    ) == (3.5, 1.2, 0.8)
    assert resolve_f0_processing(
        "vocaloid", smoothing_frames=0, correct_octave=False
    ) == (0, False)


def test_cli_wires_f0_profile_defaults_to_finetune_and_evaluate():
    from pathlib import Path

    cli_source = (Path(__file__).resolve().parents[1] / "opensv_train" / "__main__.py").read_text("utf-8")

    assert "arguments.f0_smoothing," in cli_source
    assert "arguments.f0_octave_correction," in cli_source
    assert "arguments.pitch_smoothing_hz," in cli_source
    assert "arguments.pitch_melody_weight," in cli_source
    assert "arguments.pitch_motion_weight," in cli_source
    assert "resolve_f0_processing(" in cli_source
    assert "resolve_pitch_loss_options(" in cli_source
    assert cli_source.count('action=argparse.BooleanOptionalAction, default=None') == 2
    assert cli_source.count('type=int, choices=F0_SMOOTHING_OPTIONS, default=None') == 2
    assert cli_source.count('type=float, default=None, help="override the default low-pass cutoff') == 2


def test_f0_profile_numeric_overrides_and_validation():
    assert resolve_f0_profile(
        "vocaloid", floor=55, ceiling=1700, voicing_threshold=0.5
    ) == (55.0, 1700.0, 0.5)

    with pytest.raises(ValueError, match="unknown F0 profile"):
        resolve_f0_profile("guess")
    with pytest.raises(ValueError, match="greater than the F0 floor"):
        resolve_f0_profile("standard", floor=100, ceiling=90)
    with pytest.raises(ValueError, match="voicing threshold"):
        resolve_f0_profile("standard", voicing_threshold=0)
    with pytest.raises(ValueError, match="positive finite"):
        resolve_f0_profile("standard", floor=float("nan"))


def test_f0_cache_tag_changes_with_analysis_settings():
    assert f0_cache_tag(45, 1400, 0.45) == "f0-45-1400-v0.45"
    assert f0_cache_tag(65, 1100, 0.6) == "f0-65-1100-v0.6"
    assert f0_cache_tag(45, 1400, 0.45) != f0_cache_tag(65, 1100, 0.6)
