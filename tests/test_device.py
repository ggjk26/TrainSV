import pytest
import torch

from opensv_train.data import load_dataset, make_phrases
from opensv_train.synthetic import render_dataset
from opensv_train.train import TrainConfig, Trainer
from opensv_train.voice import Voice


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_scores_match_cpu(voice_path, tmp_path):
    render_dataset(Voice.load(voice_path), tmp_path, phrases=2, seed=1, log=lambda message: None)
    phrases = [phrase for utterance in load_dataset(tmp_path, "japanese") for phrase in make_phrases(utterance)]
    scores = {}
    for device in ("cpu", "cuda"):
        config = TrainConfig(models=("acoustic", "duration", "pitch"), device=device)
        scores[device] = Trainer(Voice.load(voice_path), [], phrases, config, log=lambda message: None).evaluate()
    for key in ("mel", "voicing", "duration", "pitch_semitones"):
        assert scores["cuda"][key] == pytest.approx(scores["cpu"][key], rel=1e-3, abs=1e-4)
