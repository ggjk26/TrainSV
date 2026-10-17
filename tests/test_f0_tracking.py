from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from opensv_train.f0_tracking import _stable_f0_mask, extract_f0


def test_stable_f0_mask_rejects_single_frame_and_fast_jumps():
    assert not _stable_f0_mask(np.array([0, 220, 0], dtype=np.float32)).any()
    assert not _stable_f0_mask(np.array([100, 100, 400, 100], dtype=np.float32)).any()
    assert _stable_f0_mask(np.array([0, 220, 220, 220, 0], dtype=np.float32)).tolist() == [
        False, True, True, True, False
    ]


def test_extract_f0_retries_sensitive_settings_for_stable_weak_voicing(monkeypatch):
    calls = []

    class FakePitch:
        def __init__(self, threshold):
            self.selected_array = {
                "frequency": np.zeros(8, dtype=np.float32)
                if threshold > 0.4
                else np.array([0, 0, 220, 220, 220, 220, 0, 0], dtype=np.float32)
            }

        def xs(self):
            return np.arange(8, dtype=np.float64) * 0.005

    class FakeSound:
        def __init__(self, _audio, _rate):
            pass

        def to_pitch_ac(self, **kwargs):
            calls.append(kwargs)
            return FakePitch(kwargs["voicing_threshold"])

    monkeypatch.setitem(
        sys.modules, "parselmouth",
        SimpleNamespace(Sound=FakeSound, PraatError=RuntimeError),
    )
    audio = np.sin(2 * np.pi * 220 * np.arange(4800) / 48000).astype(np.float32)
    times, values = extract_f0(audio, 65.0, 1100.0, voicing_threshold=0.60)
    assert len(calls) == 2
    assert calls[1]["pitch_floor"] == 35.0
    assert calls[1]["pitch_ceiling"] == 1800.0
    assert calls[1]["voicing_threshold"] == 0.3
    assert values.tolist() == [0, 0, 220, 220, 220, 220, 0, 0]
    assert times.size == values.size


def test_extract_f0_does_not_promote_isolated_fallback_spikes(monkeypatch):
    class FakePitch:
        def __init__(self, threshold):
            self.selected_array = {
                "frequency": np.zeros(8, dtype=np.float32)
                if threshold > 0.4
                else np.array([0, 0, 220, 0, 0, 0, 0, 0], dtype=np.float32)
            }

        def xs(self):
            return np.arange(8, dtype=np.float64) * 0.005

    class FakeSound:
        def __init__(self, _audio, _rate):
            pass

        def to_pitch_ac(self, **kwargs):
            return FakePitch(kwargs["voicing_threshold"])

    monkeypatch.setitem(
        sys.modules, "parselmouth",
        SimpleNamespace(Sound=FakeSound, PraatError=RuntimeError),
    )
    audio = np.full(4800, 1e-4, dtype=np.float32)
    _times, values = extract_f0(audio, 65.0, 1100.0, voicing_threshold=0.60)
    assert np.all(values == 0)


def test_silence_returns_valid_zero_f0_track(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "parselmouth",
        SimpleNamespace(Sound=object, PraatError=RuntimeError),
    )
    times, values = extract_f0(np.zeros(4800, dtype=np.float32))
    assert times.size > 0
    assert values.shape == times.shape
    assert np.all(values == 0)
