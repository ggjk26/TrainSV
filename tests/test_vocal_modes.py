import csv
import json

import torch

from opensv_train.data import load_dataset, make_phrases
from opensv_train.synthetic import render_dataset
from opensv_train.train import TrainConfig, Trainer, mode_levels
from opensv_train.voice import Voice


def test_new_vocal_mode_learns_a_style_from_labelled_recordings(voice_path, tmp_path):
    student = Voice.load(voice_path)
    if student.files.info is None or not student.acoustic.vocal_modes:
        return
    target = student.acoustic.mode_style_vectors()[0].detach().clone()
    teacher = Voice.load(voice_path)
    with torch.no_grad():
        teacher.acoustic.default_style.copy_(target)
    render_dataset(teacher, tmp_path / "data", phrases=4, seed=3, log=lambda message: None)
    table = tmp_path / "data" / "transcriptions.csv"
    rows = list(csv.DictReader(table.open(encoding="utf-8")))
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*rows[0], "style"])
        writer.writeheader()
        writer.writerows({**row, "style": "Copy"} for row in rows)

    student.set_vocal_modes(["Copy"])
    default = student.acoustic.default_style_vector().detach().clone()
    start = torch.dist(student.acoustic.mode_style_vectors()[0], target).item()
    phrases = [phrase for utterance in load_dataset(tmp_path / "data", "japanese") for phrase in make_phrases(utterance)]
    Trainer(student, phrases, [], TrainConfig(params="embeddings", steps=40, anchor=0.0), log=lambda message: None).run()

    assert torch.dist(student.acoustic.mode_style_vectors()[0], target).item() < 0.9 * start
    assert torch.equal(student.acoustic.default_style_vector(), default)

    student.files.info.mode_extras = mode_levels(student, phrases, torch.device("cpu"))
    saved = json.loads(student.save(tmp_path / "databases" / "Modes", version="101").read_text("utf-8"))
    assert [style["name"] for style in saved["styles"]] == ["(base)", "(default)", "Copy"]
    assert "extra" in saved["styles"][2]
