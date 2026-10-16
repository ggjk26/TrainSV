import numpy as np
import pytest
import soundfile

from opensv_train import data as data_module
from opensv_train.data import import_labels, load_dataset, make_phrases, parse_note, resolve_f0_profile
from opensv_train.synthetic import random_phrase, render_dataset
from opensv_train.voice import Voice


def test_note_names():
    assert [parse_note(n) for n in ("C4", "A4", "D#5", "Eb3", "A4+50", "rest")] == [60, 69, 75, 51, 69.5, None]


def test_rendered_dataset_reconstructs_the_teachers_inputs(voice_path, tmp_path):
    teacher = Voice.load(voice_path)
    render_dataset(teacher, tmp_path, phrases=3, seed=5, log=lambda message: None)
    rng = np.random.default_rng(5)
    originals = [random_phrase(teacher.duration.phone_sets, "japanese", rng) for _ in range(3)]
    for utterance, original in zip(load_dataset(tmp_path, "japanese"), originals):
        (phrase,) = make_phrases(utterance)
        assert phrase.problem is None
        rebuilt, expected = teacher.duration.prepare(phrase.syllables), teacher.duration.prepare(original)
        assert [(first, end) for first, end, _ in rebuilt.intervals] == [(first, end) for first, end, _ in expected.intervals]
        assert [d.symbol for d in rebuilt.phonemes] == [d.symbol for d in expected.phonemes]
        teacher_seconds = [d.seconds for d in teacher.duration.predict(original)]
        assert np.abs(np.array(phrase.syllable_targets) - teacher_seconds).max() <= 0.0051
        assert len(phrase.audio()) == phrase.frames * 240
        notes, phonemes, midi, mask = phrase.pitch_example()
        assert len(notes) == len(original)
        assert abs(sum(p.seconds for p in phonemes) - len(midi) * 0.005) < 0.01
        assert 0.3 < mask.mean() < 1.0


def test_lab_import(tmp_path):
    (tmp_path / "lab").mkdir()
    (tmp_path / "wav").mkdir()
    (tmp_path / "lab" / "song.lab").write_text("0 3000000 pau\n3000000 4000000 k\n4000000 6500000 a\n6500000 7300000 vf\n7300000 9000000 i\n9000000 12000000 br\n")
    soundfile.write(str(tmp_path / "wav" / "song.wav"), np.zeros(int(1.2 * 44100), dtype=np.float32), 44100)
    import_labels(tmp_path / "lab", tmp_path / "wav", ["song"], tmp_path / "dataset", {"vf": "cl"})
    rows = (tmp_path / "dataset" / "transcriptions.csv").read_text().splitlines()
    assert rows[1].startswith("song,SP k a cl i AP,0.3000000 0.1000000 0.2500000 0.0800000 0.1700000 0.3000000")

    (tmp_path / "lab" / "gap.lab").write_text("0 1000 a\n2000 3000 i\n")
    soundfile.write(str(tmp_path / "wav" / "gap.wav"), np.zeros(100, dtype=np.float32), 44100)
    with pytest.raises(ValueError):
        import_labels(tmp_path / "lab", tmp_path / "wav", ["gap"], tmp_path / "other", {})


def test_word_spans_follow_diffsinger():
    from opensv_train.data import word_spans

    assert word_spans("n i h ao SP AP".split(), {"i", "ao"}) == [1, 2, 1, 1, 1]
    assert word_spans("SP k a N t o cl AP".split(), {"a", "o", "N"}) == [2, 1, 2, 2, 1]


def test_f0_tracking_profiles_and_overrides():
    assert resolve_f0_profile("standard") == (65.0, 1100.0, 0.60)
    assert resolve_f0_profile("vocaloid") == (45.0, 1400.0, 0.45)
    assert resolve_f0_profile(
        "vocaloid", floor=55, ceiling=1700, voicing_threshold=0.5
    ) == (55.0, 1700.0, 0.5)

    with pytest.raises(ValueError, match="unknown F0 profile"):
        resolve_f0_profile("guess")
    with pytest.raises(ValueError, match="greater than the F0 floor"):
        resolve_f0_profile("standard", floor=100, ceiling=90)
    with pytest.raises(ValueError, match="voicing threshold"):
        resolve_f0_profile("standard", voicing_threshold=0)


def test_f0_profile_parameters_are_passed_to_tracker_and_cache_is_separate(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    wavs = dataset / "wavs"
    wavs.mkdir(parents=True)
    (wavs / "sample.wav").write_bytes(b"test wav placeholder")
    (dataset / "transcriptions.csv").write_text(
        "name,ph_seq,ph_dur\nsample,a,0.1\n", encoding="utf-8"
    )
    calls = []

    def fake_read_audio(_path):
        return np.zeros(4800, dtype=np.float32)

    def fake_extract_f0(audio, floor, ceiling, *, voicing_threshold):
        calls.append((floor, ceiling, voicing_threshold))
        return np.array([0.0, 0.005, 0.01]), np.array([220.0, 0.0, 330.0])

    monkeypatch.setattr(data_module, "read_audio", fake_read_audio)
    monkeypatch.setattr(data_module, "extract_f0", fake_extract_f0)
    cache = dataset / ".cache"

    synthetic_voice = load_dataset(
        dataset, "japanese", cache=cache,
        f0_range=(45.0, 1400.0), f0_voicing_threshold=0.45,
    )
    assert calls == [(45.0, 1400.0, 0.45)]
    assert synthetic_voice[0].f0_values.tolist() == [220.0, 0.0, 330.0]
    assert (cache / "sample.f0-45-1400-v0.45.npz").is_file()

    # Same settings reuse the matching cache rather than re-extracting F0.
    load_dataset(
        dataset, "japanese", cache=cache,
        f0_range=(45.0, 1400.0), f0_voicing_threshold=0.45,
    )
    assert len(calls) == 1

    # Standard settings have a distinct cache key and trigger a new analysis.
    load_dataset(dataset, "japanese", cache=cache)
    assert calls[-1] == (65.0, 1100.0, 0.60)
    assert len(calls) == 2
