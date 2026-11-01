from types import SimpleNamespace

import pytest
import torch
from torch import nn

from opensv_train.train import TrainConfig, Trainer


class _LanguageEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 2))


class _FakeAcoustic(nn.Module):
    def __init__(self):
        super().__init__()
        self.phone_sets = [
            SimpleNamespace(name="Mandarin"),
            SimpleNamespace(name="Japanese"),
        ]
        self.language_embedding = _LanguageEmbedding()
        self.speaker = nn.Parameter(torch.ones(2))
        self.default_style = nn.Parameter(torch.ones(2))
        self.mode_styles = nn.ParameterList()
        self.shared_layer = nn.Linear(2, 2)


class _FakeVoice(nn.Module):
    def __init__(self):
        super().__init__()
        self.acoustic = _FakeAcoustic()
        self.vocoder = nn.Linear(2, 2)

    @property
    def models(self):
        return {"acoustic": self.acoustic, "vocoder": self.vocoder}


def _phrase(language):
    return SimpleNamespace(
        phones=[SimpleNamespace(language=language, symbol="a")]
    )


def _trainer(train, validation=None, **overrides):
    config_values = {
        "models": ("acoustic",),
        "params": "all",
        "device": "cpu",
        "cross_language_preservation_weight": 1.0,
    }
    config_values.update(overrides)
    config = TrainConfig(**config_values)
    voice = _FakeVoice()
    messages = []
    trainer = Trainer(
        voice,
        train,
        validation if validation is not None else [_phrase("mandarin")],
        config,
        log=messages.append,
    )
    return trainer, voice, messages


def test_cross_language_preservation_freezes_non_source_language_embedding():
    trainer, voice, messages = _trainer([_phrase("mandarin")])
    embedding = voice.acoustic.language_embedding.weight
    embedding.sum().backward()

    assert torch.equal(embedding.grad[:, 0], torch.ones(4))
    assert torch.equal(embedding.grad[:, 1], torch.zeros(4))
    assert trainer._cross_language_anchors
    assert any("monolingual mandarin" in message for message in messages)


def test_cross_language_preservation_penalizes_drift_in_shared_acoustic_weights():
    trainer, voice, _ = _trainer([_phrase("mandarin")])
    initial = trainer.cross_language_preservation_penalty().item()
    with torch.no_grad():
        voice.acoustic.shared_layer.weight.add_(0.25)
    changed = trainer.cross_language_preservation_penalty().item()

    assert initial == 0.0
    assert changed > initial


def test_cross_language_preservation_rejects_mixed_language_training_data():
    with pytest.raises(ValueError, match="monolingual training set"):
        _trainer([_phrase("mandarin"), _phrase("japanese")])


def test_cross_language_preservation_rejects_foreign_language_validation_data():
    with pytest.raises(ValueError, match="same language"):
        _trainer([_phrase("mandarin")], validation=[_phrase("japanese")])


def test_cross_language_preservation_requires_full_acoustic_training():
    with pytest.raises(ValueError, match="--params all"):
        _trainer([_phrase("mandarin")], params="embeddings")
