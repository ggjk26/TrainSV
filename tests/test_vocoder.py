import numpy as np
import torch

import parity
from opensv_train.vocoder import Vocoder
from opensv_train.voice import load_voice
from test_acoustic import acoustic_model, random_timed_phrase


def relative_error(actual, expected):
    return float(np.abs(actual - expected).max() / max(np.abs(expected).max(), 1e-9))


def test_vocoder_stages_match_engine(voice_path):
    voice = load_voice(voice_path)
    phonemes, log_f0 = random_timed_phrase(voice, np.random.default_rng(3), notes=6)
    with torch.no_grad():
        frames = acoustic_model(voice)(phonemes, torch.from_numpy(log_f0)).frames
        expected = parity.run({"voice": voice_path, "command": "vocoder"}, {"frames": frames.numpy()})
        output = Vocoder(voice.models["vocoder"])(frames)
    stages = output.stages
    np.testing.assert_allclose(expected["rebuilt"], expected["samples"], atol=1e-6)
    checks = {
        "normalized": stages["normalized"], "shape": stages["shape"].unsqueeze(1), "condition": stages["condition"],
        "spectrum0": stages["spectra"][0], "spectrum1": stages["spectra"][1], "spectrum2": stages["spectra"][2],
        "modulation_parameters": stages["modulation_parameters"], "excitation": stages["excitation"],
        "modulation": stages["modulation"], "residual": stages["residual"],
        "component0": stages["components"][0], "component1": stages["components"][1], "component2": stages["components"][2],
        "samples": output.samples,
    }
    errors = {name: relative_error(value.numpy(), expected[name]) for name, value in checks.items()}
    assert all(error < 1e-4 for error in errors.values()), errors
