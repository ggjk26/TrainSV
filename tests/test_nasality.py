import math

import numpy as np
import pytest

from opensv_train.nasality import oral_vowel_frames


def test_vowel_between_stop_and_silence_is_oral():
    mask = oral_vowel_frames(["silence", "stop", "vowel", "silence"], [2, 1, 3, 2])
    assert mask.tolist() == [0, 0, 0, 1, 1, 1, 0, 0]


def test_vowel_next_to_a_nasal_is_excluded_on_either_side():
    onset = oral_vowel_frames(["nasal", "vowel", "silence"], [2, 3, 1])
    coda = oral_vowel_frames(["stop", "vowel", "nasal"], [1, 3, 2])
    assert onset.sum() == 0 and coda.sum() == 0


def test_diphthong_counts_and_the_next_syllable_nasal_still_excludes_it():
    assert oral_vowel_frames(["diphthong"], [4]).tolist() == [1, 1, 1, 1]
    assert oral_vowel_frames(["diphthong", "nasal", "vowel"], [2, 1, 2]).tolist() == [0, 0, 0, 0, 0]


def test_length_matches_total_frames_and_inputs_are_validated():
    assert oral_vowel_frames(["vowel", "stop"], [3, 4]).shape == (7,)
    with pytest.raises(ValueError):
        oral_vowel_frames(["vowel"], [1, 2])
    with pytest.raises(ValueError):
        oral_vowel_frames(["vowel"], [0])


torch = pytest.importorskip("torch")
from opensv_train.losses import SpectralLoss  # noqa: E402

RATE = 48000


def _signal(low_gain: float) -> "torch.Tensor":
    time = torch.arange(RATE // 2, dtype=torch.float32) / RATE
    return (
        0.05 * low_gain * torch.sin(2 * math.pi * 300 * time)
        + 0.3 * torch.sin(2 * math.pi * 700 * time)
        + 0.2 * torch.sin(2 * math.pi * 1500 * time)
    )


def _mask(value: float = 1.0) -> "torch.Tensor":
    return torch.full((RATE // 2,), value)


def test_nasal_term_is_absent_without_a_mask():
    assert "nasal" not in SpectralLoss()(_signal(1.0), _signal(1.0))


def test_nasal_term_is_zero_when_prediction_matches_the_target():
    terms = SpectralLoss(nasal_weight=1.0)(_signal(1.0), _signal(1.0), nasal_mask=_mask())
    assert float(terms["nasal"]) < 1e-6
    assert abs(float(terms["nasal_delta_db"])) < 1e-4


def test_excess_low_band_energy_is_penalised_but_a_lighter_low_band_is_not():
    loss = SpectralLoss(nasal_weight=1.0)
    heavy = loss(_signal(8.0), _signal(1.0), nasal_mask=_mask())
    light = loss(_signal(0.0), _signal(1.0), nasal_mask=_mask())
    assert float(heavy["nasal"]) > 0.05 and float(heavy["nasal_delta_db"]) > 0
    assert float(light["nasal"]) == 0.0 and float(light["nasal_delta_db"]) < 0


def test_margin_asks_for_less_low_band_energy_than_the_target():
    terms = SpectralLoss(nasal_weight=1.0, nasal_margin_db=6.0)(_signal(1.0), _signal(1.0), nasal_mask=_mask())
    assert float(terms["nasal"]) == pytest.approx(0.6, abs=1e-3)


def test_empty_mask_gives_a_finite_zero_term_and_the_weight_scales_the_total():
    empty = SpectralLoss(nasal_weight=1.0)(_signal(8.0), _signal(1.0), nasal_mask=_mask(0.0))
    assert float(empty["nasal"]) == 0.0 and torch.isfinite(empty["total"])
    base = SpectralLoss(nasal_weight=0.0)(_signal(8.0), _signal(1.0), nasal_mask=_mask())
    scaled = SpectralLoss(nasal_weight=2.0)(_signal(8.0), _signal(1.0), nasal_mask=_mask())
    assert float(scaled["total"] - base["total"]) == pytest.approx(2 * float(scaled["nasal"]), rel=1e-4)
