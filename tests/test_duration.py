import dataclasses

import numpy as np

import parity
from opensv_train.duration import DurationModel
from opensv_train.synthetic import random_phrase
from opensv_train.voice import load_voice


def syllable_json(syllables):
    return [{"language": s.language, "phonemes": s.phonemes, "duration": s.seconds, "pitch": s.pitch, "continuation": s.continuation} for s in syllables]


def duration_model(voice):
    return DurationModel(voice.models["duration"], None if voice.info is None else voice.info.duration_voice)


def test_duration_matches_engine(voice_path):
    voice = load_voice(voice_path)
    model = duration_model(voice)
    rng = np.random.default_rng(1)
    for language in ("japanese", "english", "mandarin"):
        for _ in range(3):
            syllables = random_phrase(model.phone_sets, language, rng, notes=10)
            expected = parity.run({"voice": voice_path, "command": "duration", "syllables": syllable_json(syllables)})["durations"]
            actual = [dataclasses.asdict(d) for d in model.predict(syllables)]
            assert [(d["symbol"], d["syllable"], d["interval"]) for d in actual] == [(d["symbol"], d["syllable"], d["interval"]) for d in expected]
            np.testing.assert_allclose([d["seconds"] for d in actual], [d["seconds"] for d in expected], rtol=1e-4, atol=1e-6)
