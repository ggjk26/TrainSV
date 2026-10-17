from __future__ import annotations

import pytest

from opensv_train.f0 import F0_PROFILES, f0_cache_tag, resolve_f0_profile


def test_f0_profiles_have_expected_ranges():
    assert tuple(F0_PROFILES) == ("standard", "vocaloid")
    assert resolve_f0_profile("standard") == (65.0, 1100.0, 0.60)
    assert resolve_f0_profile("vocaloid") == (45.0, 1400.0, 0.45)


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
