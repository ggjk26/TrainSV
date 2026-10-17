import dataclasses

import numpy as np
import torch

import parity
from opensv_train.pitch import PitchModel, PitchNote, pad_pitch_notes
from opensv_train.train import _low_pass
from opensv_train.synthetic import random_phrase
from opensv_train.voice import Voice, initialize_pitch_from_file, load_voice
from test_duration import duration_model


def pitch_model(voice):
    return PitchModel(voice.models["pitch"], None if voice.info is None else voice.info.pitch_speaker)


def random_pitch_notes(phone_sets, language, rng, notes=8):
    syllables = random_phrase(phone_sets, language, rng, notes=notes)[1:-1]
    result = []
    for syllable in syllables:
        silent = syllable.phonemes == ["sil"]
        result.append(PitchNote(syllable, silence=silent, continuation=syllable.continuation, rap=bool(rng.random() < 0.2), tone=int(rng.integers(0, 5)), vibrato=float(rng.uniform(0.5, 1.5))))
    return pad_pitch_notes(result)


def note_json(notes):
    return [{
        "language": n.syllable.language, "phonemes": n.syllable.phonemes, "duration": n.syllable.seconds, "pitch": n.syllable.pitch,
        "continuation": n.continuation, "silence": n.silence, "breath": n.breath, "rap": n.rap, "tone": n.tone, "vibrato": n.vibrato,
    } for n in notes]


def test_pitch_matches_engine(voice_path):
    voice = load_voice(voice_path)
    timing = duration_model(voice)
    model = pitch_model(voice)
    rng = np.random.default_rng(4)
    for language in ("japanese", "mandarin", "english"):
        notes = random_pitch_notes(model.phone_sets, language, rng)
        if language == "english":
            for note in notes[1:-1]:
                note.syllable.pitch += 30
        phonemes = timing.predict([note.syllable for note in notes])
        envelope = rng.uniform(0.5, 1.2, 20000).astype(np.float32)
        expected = parity.run({
            "voice": voice_path, "command": "pitch", "notes": note_json(notes),
            "phonemes": [dataclasses.asdict(p) for p in phonemes], "vibratoEnvelope": envelope.tolist(),
        })["midi"]
        actual = model.predict(notes, phonemes, vibrato_envelope=envelope)
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=5e-3)



def test_pitch_import_preserves_target_speaker_and_accepts_identical_model(voice_path, tmp_path):
    voice = Voice.load(voice_path)
    source = tmp_path / "pitch.dnni"
    source.write_bytes(voice.files.models["pitch"].serialize())
    speaker_before = voice.pitch.speaker_vector().detach().clone()
    parameters_before = {
        key: value.detach().clone()
        for key, value in voice.pitch.named_parameters()
        if not key.startswith("speaker")
    }

    count = initialize_pitch_from_file(voice, source)

    assert count == len(parameters_before)
    torch.testing.assert_close(voice.pitch.speaker_vector(), speaker_before, rtol=0, atol=0)
    for key, expected in parameters_before.items():
        torch.testing.assert_close(dict(voice.pitch.named_parameters())[key], expected, rtol=0, atol=0)


def test_pitch_smoothing_cutoff_is_configurable():
    sample_rate = 200.0
    time = torch.arange(0, 2.0, 1.0 / sample_rate)
    curve = torch.sin(2 * np.pi * 8.0 * time)
    smooth = _low_pass(curve, cutoff_hz=3.0)
    detailed = _low_pass(curve, cutoff_hz=12.0)
    assert smooth.shape == detailed.shape == curve.shape
    assert smooth.square().mean() < detailed.square().mean()


def test_pitch_smoothing_rejects_invalid_cutoff():
    import pytest
    import torch

    with pytest.raises(ValueError, match="cutoff"):
        _low_pass(torch.ones(20), cutoff_hz=0.0)
    with pytest.raises(ValueError, match="cutoff"):
        _low_pass(torch.ones(20), cutoff_hz=100.0)
