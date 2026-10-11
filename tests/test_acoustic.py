import numpy as np
import torch

import parity
from opensv_train.acoustic import AcousticModel, TimedPhoneme, midi_to_log_f0
from opensv_train.duration import quantize_durations
from opensv_train.synthetic import random_phrase
from test_duration import duration_model


def acoustic_model(voice):
    if voice.info is None:
        return AcousticModel(voice.models["acoustic"], voice.vocal_modes)
    info = voice.info
    return AcousticModel(voice.models["acoustic"], list(info.vocal_modes), info.acoustic_speaker, info.default_style, list(info.vocal_modes.values()))


def random_timed_phrase(voice, rng, language="japanese", notes=10):
    timing = duration_model(voice)
    syllables = random_phrase(timing.phone_sets, language, rng, notes=notes)
    durations = timing.predict(syllables)
    counts = quantize_durations(durations, 0.005)
    phonemes = [TimedPhoneme(d.language, d.symbol, count) for d, count in zip(durations, counts)]
    frames = sum(counts)
    midi = np.repeat([s.pitch for s in syllables], np.maximum(1, np.round(np.array([s.seconds for s in syllables]) / 0.005).astype(int)))
    midi = np.interp(np.linspace(0, len(midi) - 1, frames), np.arange(len(midi)), midi)
    midi = midi + 0.3 * np.sin(np.arange(frames) * 2 * np.pi * 5.5 * 0.005)
    return phonemes, midi_to_log_f0(midi).astype(np.float32)


def phoneme_json(phonemes):
    return [{"language": p.language, "symbol": p.symbol, "frames": p.frames} for p in phonemes]


def test_acoustic_matches_engine(voice_path):
    from opensv_train.voice import load_voice

    voice = load_voice(voice_path)
    model = acoustic_model(voice)
    rng = np.random.default_rng(2)
    for language in ("japanese", "english"):
        phonemes, log_f0 = random_timed_phrase(voice, rng, language)
        arrays = {"logF0": log_f0}
        request = {"voice": voice_path, "command": "acoustic", "phonemes": phoneme_json(phonemes), "seed": 5489}
        if model.vocal_modes:
            weights = rng.uniform(0, 1, (len(log_f0), len(model.vocal_modes))).astype(np.float32)
            arrays["vocalModeWeights"] = weights
        expected = parity.run(request, arrays)["frames"]
        with torch.no_grad():
            actual = model(phonemes, torch.from_numpy(log_f0), 5489, torch.from_numpy(arrays["vocalModeWeights"]) if model.vocal_modes else None).frames.numpy()
        assert actual.shape == expected.shape
        voiced_mismatch = np.mean(actual[:, 0] != expected[:, 0])
        assert voiced_mismatch < 0.002
        np.testing.assert_allclose(actual[:, 1:], expected[:, 1:], atol=2e-3, rtol=1e-3)
