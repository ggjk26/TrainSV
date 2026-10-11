import dataclasses
import json
import os

import numpy as np
import torch

import parity
from opensv_train.nofs import NofsFile
from opensv_train.voice import Voice
from test_acoustic import phoneme_json, random_timed_phrase
from test_duration import syllable_json
from test_pitch import note_json, random_pitch_notes


def perturb(module, generator, scale=0.05):
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * parameter.abs().mean() * scale)


def test_saved_voice_runs_identically_in_engine(voice_path, tmp_path):
    voice = Voice.load(voice_path)
    generator = torch.Generator().manual_seed(0)
    for model in voice.models.values():
        perturb(model, generator)
    if voice.files.info is None:
        saved = voice.save(tmp_path / "voice.nofs")
        assert os.path.getsize(saved) == os.path.getsize(voice_path)
        renamed = NofsFile.load(voice.save(tmp_path / "renamed.nofs", name="A longer perturbed singer name"))
        assert sorted(renamed.index_offsets()) == sorted(renamed.record_offsets())
        assert renamed.metadata()[".name"] == "A longer perturbed singer name"
    else:
        saved = voice.save(tmp_path / "databases" / "Perturbed", name="Perturbed", version="101")
    perturbed = voice
    reloaded = Voice.load(saved)
    saved = str(saved)

    rng = np.random.default_rng(7)
    syllables = [note.syllable for note in random_pitch_notes(reloaded.pitch.phone_sets, "japanese", rng)]
    durations = parity.run({"voice": saved, "command": "duration", "syllables": syllable_json(syllables)})["durations"]
    np.testing.assert_allclose([d.seconds for d in reloaded.duration.predict(syllables)], [d["seconds"] for d in durations], rtol=1e-4)
    np.testing.assert_allclose([d.seconds for d in perturbed.duration.predict(syllables)], [d["seconds"] for d in durations], rtol=1e-3)

    notes = random_pitch_notes(reloaded.pitch.phone_sets, "japanese", rng)
    phonemes = reloaded.duration.predict([note.syllable for note in notes])
    midi = parity.run({"voice": saved, "command": "pitch", "notes": note_json(notes), "phonemes": [dataclasses.asdict(p) for p in phonemes]})["midi"]
    np.testing.assert_allclose(reloaded.pitch.predict(notes, phonemes), midi, atol=5e-3)

    phonemes, log_f0 = random_timed_phrase(reloaded.files, rng, notes=6)
    expected = parity.run({"voice": saved, "command": "synthesize", "phonemes": phoneme_json(phonemes)}, {"logF0": log_f0})
    with torch.no_grad():
        frames = reloaded.acoustic(phonemes, torch.from_numpy(log_f0)).frames
        samples = reloaded.vocoder(frames).samples.numpy()
        unrounded = perturbed.vocoder(perturbed.acoustic(phonemes, torch.from_numpy(log_f0)).frames).samples.numpy()
    np.testing.assert_allclose(frames.numpy()[:, 1:], expected["frames"][:, 1:], atol=2e-3, rtol=1e-3)
    assert np.abs(samples - expected["samples"]).max() < 1e-4 * np.abs(expected["samples"]).max()
    assert np.abs(unrounded - expected["samples"]).max() < 2e-2 * np.abs(expected["samples"]).max()


def test_unchanged_info_voice_keeps_model_references(voice_path, tmp_path):
    voice = Voice.load(voice_path)
    if voice.files.info is None:
        return
    saved = json.loads(voice.save(tmp_path / "databases" / "Copy", version="101").read_text("utf-8"))
    for kind in ("duration", "f0", "acoustic", "vocoder"):
        assert saved[kind] == voice.files.info.info[kind]
