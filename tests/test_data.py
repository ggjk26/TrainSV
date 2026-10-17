import numpy as np
import pytest
import soundfile

from opensv_train.data import import_labels, load_dataset, make_phrases, parse_note
from opensv_train.synthetic import random_phrase, render_dataset
from opensv_train.voice import Voice


def test_f0_tracking_settings_are_validated():
    from opensv_train.data import _validate_f0_settings

    _validate_f0_settings(45.0, 650.0, 0.55)
    with pytest.raises(ValueError, match="F0 range"):
        _validate_f0_settings(0.0, 650.0, 0.55)
    with pytest.raises(ValueError, match="F0 range"):
        _validate_f0_settings(700.0, 650.0, 0.55)
    with pytest.raises(ValueError, match="threshold"):
        _validate_f0_settings(45.0, 650.0, 1.5)


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
