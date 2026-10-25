import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from opensv_train.transitions import (
    sample_transition_segment,
    transition_boundary_frames,
    transition_crop_samples,
    transition_frame_weights,
)


def _phones(*symbols):
    return [SimpleNamespace(symbol=symbol) for symbol in symbols]


def test_boundary_collection_excludes_silence_and_preserves_frame_offsets():
    phones = _phones("sil", "n", "y", "a", "sil")
    assert transition_boundary_frames(phones, [10, 12, 8, 30, 10]) == [22, 30]
    weights = transition_frame_weights(phones, [10, 12, 8, 30, 10], weight=1.5, radius=2)
    assert np.all(weights[28:32] == 1.5)
    assert np.all(weights[20:24] == 1.5)
    assert np.all(weights[:10] == 1.0)
    assert np.all(weights[60:] == 1.0)


def test_transition_sampling_centers_the_window_and_keeps_spectral_context():
    rng = random.Random(3)
    frames, width, context = 300, 80, 8
    boundaries = [70, 130, 220]
    for _ in range(50):
        start, boundary = sample_transition_segment(
            frames, width, boundaries, rng, probability=1.0, context_frames=context
        )
        assert boundary in boundaries
        assert start <= boundary - context
        assert boundary + context <= start + width
        assert 0 <= start <= frames - width


def test_transition_sampling_zero_probability_matches_uniform_fallback():
    seed = 202
    actual = sample_transition_segment(100, 25, [45, 68], random.Random(seed), probability=0.0)
    expected = (random.Random(seed).randint(0, 75), None)
    assert actual == expected


def test_transition_crop_samples_include_context_and_fall_back_at_edges():
    assert transition_crop_samples(80, 160, 480, 8) == (34560, 42240)
    assert transition_crop_samples(5, 160, 480, 8) is None
    assert transition_crop_samples(155, 160, 480, 8) is None


def test_transition_features_are_wired_into_training_cli_and_webui():
    root = Path(__file__).resolve().parents[1]
    trainer = (root / "opensv_train" / "train.py").read_text("utf-8")
    cli = (root / "opensv_train" / "__main__.py").read_text("utf-8")
    gui = (root / "trainsv_gui.py").read_text("utf-8")
    assert "sample_transition_segment(" in trainer
    assert "transition_crop_samples(" in trainer
    assert 'command.add_argument("--transition-sampling-probability"' in cli
    assert 'command.add_argument("--transition-local-weight"' in cli
    assert 'data-id="t_transition_sampling"' in gui
    assert 'data-id="t_transition_local_weight"' in gui
