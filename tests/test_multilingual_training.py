from argparse import Namespace
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from opensv_train.__main__ import _resolve_dataset_phoneme_maps, _resolve_language_weights, _resolve_training_datasets, parser
from opensv_train.train import TrainConfig, Trainer


class _FakeAcoustic(nn.Module):
    def __init__(self):
        super().__init__()
        self.phone_sets = [
            SimpleNamespace(name="Mandarin"),
            SimpleNamespace(name="Japanese"),
        ]
        self.language_embedding = nn.Embedding(2, 4)
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


def _phrase(language, frames=10):
    return SimpleNamespace(
        phones=[SimpleNamespace(language=language, symbol="a")],
        frames=frames,
    )


def _trainer(train, validation=None, weights=None):
    return Trainer(
        _FakeVoice(),
        train,
        validation or [],
        TrainConfig(
            models=("acoustic",),
            params="all",
            device="cpu",
            language_weights=weights or {"mandarin": 1.0, "japanese": 1.0},
        ),
        log=lambda _: None,
    )


def test_joint_dataset_cli_accepts_separate_language_directories():
    args = parser().parse_args([
        "finetune",
        "--voice", "base.nofs",
        "--output", "trained.nofs",
        "--dataset", "zh=C:/datasets/mandarin",
        "--dataset", "ja=C:/datasets/japanese",
    ])
    datasets = _resolve_training_datasets(args)
    assert datasets == [
        ("mandarin", "C:/datasets/mandarin"),
        ("japanese", "C:/datasets/japanese"),
    ]
    assert _resolve_language_weights(args, datasets) == {
        "mandarin": 1.0,
        "japanese": 1.0,
    }


def test_joint_dataset_cli_accepts_custom_language_weights():
    args = Namespace(
        dataset=["mandarin=zh-data", "japanese=ja-data"],
        data=None,
        language=None,
        language_weight=["mandarin=1", "japanese=3"],
    )
    datasets = _resolve_training_datasets(args)
    assert _resolve_language_weights(args, datasets) == {
        "mandarin": 1.0,
        "japanese": 3.0,
    }


def test_single_dataset_cli_remains_compatible_with_existing_mode():
    args = Namespace(
        dataset=[],
        data="one-dataset",
        language="ja",
        language_weight=[],
    )
    assert _resolve_training_datasets(args) == [("japanese", "one-dataset")]
    assert _resolve_language_weights(args, [("japanese", "one-dataset")]) == {}


@pytest.mark.parametrize("specs", [
    ["mandarin=one"],
    ["mandarin=one", "zh=two"],
])
def test_joint_training_requires_distinct_languages(specs):
    args = Namespace(dataset=specs, data=None, language=None, language_weight=[])
    with pytest.raises(SystemExit, match="at least two datasets|one --dataset per language"):
        _resolve_training_datasets(args)


def test_language_weights_must_cover_each_dataset_once():
    args = Namespace(
        dataset=["mandarin=zh-data", "japanese=ja-data"],
        data=None,
        language=None,
        language_weight=["mandarin=1"],
    )
    datasets = _resolve_training_datasets(args)
    with pytest.raises(SystemExit, match="exactly one positive weight"):
        _resolve_language_weights(args, datasets)


def test_language_weights_must_be_positive_and_finite():
    args = Namespace(
        dataset=["mandarin=zh-data", "japanese=ja-data"],
        data=None,
        language=None,
        language_weight=["mandarin=0", "japanese=1"],
    )
    datasets = _resolve_training_datasets(args)
    with pytest.raises(SystemExit, match="greater than zero"):
        _resolve_language_weights(args, datasets)


def test_trainer_rejects_a_phrase_that_contains_multiple_languages():
    phrase = SimpleNamespace(phones=[
        SimpleNamespace(language="mandarin", symbol="a"),
        SimpleNamespace(language="japanese", symbol="a"),
    ])
    with pytest.raises(ValueError, match="exactly one language"):
        _trainer([phrase])


def test_joint_sampler_balances_languages_independently_of_dataset_size():
    trainer = _trainer(
        [_phrase("mandarin")] * 20 + [_phrase("japanese")],
        weights={"mandarin": 1.0, "japanese": 1.0},
    )
    groups = {
        "mandarin": [_phrase("mandarin")] * 20,
        "japanese": [_phrase("japanese")],
    }
    counts = Counter()
    for _ in range(4000):
        phrase = trainer._sample_language_phrase("acoustic", groups, {
            "mandarin": [10] * 20,
            "japanese": [10],
        })
        counts[phrase.phones[0].language] += 1
    share = counts["mandarin"] / sum(counts.values())
    assert 0.46 < share < 0.54


def test_joint_sampler_applies_weights_between_languages():
    trainer = _trainer(
        [_phrase("mandarin"), _phrase("japanese")],
        weights={"mandarin": 1.0, "japanese": 3.0},
    )
    groups = {"mandarin": [_phrase("mandarin")], "japanese": [_phrase("japanese")]}
    counts = Counter()
    for _ in range(4000):
        phrase = trainer._sample_language_phrase("duration", groups, {})
        counts[phrase.phones[0].language] += 1
    japanese_share = counts["japanese"] / sum(counts.values())
    assert 0.72 < japanese_share < 0.78


def test_audio_sampling_remains_frame_weighted_inside_each_language():
    trainer = _trainer([_phrase("mandarin"), _phrase("japanese")])
    short, long = _phrase("mandarin", 10), _phrase("mandarin", 90)
    groups = {"mandarin": [short, long], "japanese": [_phrase("japanese")]}
    cumulative = {"mandarin": [10, 100], "japanese": [10]}
    counts = Counter()
    for _ in range(4000):
        phrase = trainer._sample_language_phrase("acoustic", groups, cumulative)
        if phrase is short:
            counts["short"] += 1
        elif phrase is long:
            counts["long"] += 1
    assert 0.87 < counts["long"] / (counts["short"] + counts["long"]) < 0.93


def test_joint_validation_reports_each_language_and_uses_macro_weighting():
    trainer = _trainer(
        [_phrase("mandarin"), _phrase("japanese")],
        validation=[_phrase("mandarin"), _phrase("japanese")],
        weights={"mandarin": 1.0, "japanese": 3.0},
    )

    def fake_evaluate(phrases=None):
        language = phrases[0].phones[0].language
        return {"acoustic": 0.2 if language == "mandarin" else 0.6}

    trainer.evaluate = fake_evaluate
    aggregate, per_language = trainer._validation_report()
    assert per_language == {
        "mandarin": {"acoustic": 0.2},
        "japanese": {"acoustic": 0.6},
    }
    assert aggregate["acoustic"] == pytest.approx(0.5)

def test_joint_training_accepts_separate_phoneme_maps_per_language():
    args = Namespace(
        dataset=["mandarin=zh-data", "japanese=ja-data"],
        dataset_phoneme_map=["mandarin=zh-map.json", "ja=ja-map.json"],
        data=None,
        language=None,
        language_weight=[],
    )
    datasets = _resolve_training_datasets(args)
    assert _resolve_dataset_phoneme_maps(args, datasets) == {
        "mandarin": "zh-map.json",
        "japanese": "ja-map.json",
    }


def test_language_specific_phoneme_map_cannot_reference_an_absent_language():
    args = Namespace(
        dataset=["mandarin=zh-data", "japanese=ja-data"],
        dataset_phoneme_map=["english=en-map.json"],
        data=None,
        language=None,
        language_weight=[],
    )
    datasets = _resolve_training_datasets(args)
    with pytest.raises(SystemExit, match="has no --dataset"):
        _resolve_dataset_phoneme_maps(args, datasets)


def test_language_specific_phoneme_map_cannot_be_repeated():
    args = Namespace(
        dataset=["mandarin=zh-data", "japanese=ja-data"],
        dataset_phoneme_map=["mandarin=one.json", "zh=two.json"],
        data=None,
        language=None,
        language_weight=[],
    )
    datasets = _resolve_training_datasets(args)
    with pytest.raises(SystemExit, match="duplicate --dataset-phoneme-map"):
        _resolve_dataset_phoneme_maps(args, datasets)

