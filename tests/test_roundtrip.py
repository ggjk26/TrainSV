import json
import numpy as np
import pytest
import torch

from opensv_train.dnni import DnniFile, _read_quantization, read_matrix, write_matrix
from opensv_train.layers import MatrixParameter, VectorParameter
from opensv_train.nofs import is_info_voice
from opensv_train.voice import INFO_KINDS, ROLES, Voice, load_voice

def tied_parameters(root):
    for node in root.walk():
        if node.type in ("prim0", "prim2", "prim3", "prim4", "prim5"):
            yield MatrixParameter(node)
        elif node.type == "prim1":
            yield VectorParameter(node)


def test_unmodified_export_is_byte_identical(voice_path):
    voice = load_voice(voice_path)
    for role in ROLES:
        original = voice.models[role].serialize()
        copy = DnniFile.parse(original)
        for parameter in tied_parameters(copy.root):
            parameter.export()
        assert copy.serialize() == original, role


def test_requantized_matrix_tracks_changes(voice_path):
    voice = load_voice(voice_path)
    node = next(node for node in voice.models["acoustic"].root.walk() if node.type == "prim4")
    matrix = read_matrix(node)
    changed = matrix + np.random.default_rng(0).normal(0, np.abs(matrix).mean() * 0.1, matrix.shape)
    changed[0] *= 3.0
    write_matrix(node, changed)
    decoded = read_matrix(node)
    step = _read_quantization(node.payload).scales[:, None] / 32768
    assert step[0, 0] * 32768 >= np.abs(changed[0]).max() * (1 - 1e-6)
    assert np.all(np.abs(decoded - changed) <= step * 1.01)


def test_parameters_receive_gradients(voice_path):
    voice = load_voice(voice_path)
    node = next(node for node in voice.models["duration"].root.walk() if node.type == "prim4")
    parameter = MatrixParameter(node)
    parameter().sum().backward()
    assert parameter.weight.grad is not None and torch.all(parameter.weight.grad == 1)


def test_preserve_mode_loads_only_acoustic_and_keeps_other_model_bytes(voice_path, tmp_path):
    if not is_info_voice(voice_path):
        pytest.skip("the configured test voice is a regular voice.nofs, not info.nofs")

    source = load_voice(voice_path, preserve_non_acoustic_models=True)
    assert source.info is not None
    assert set(source.models) == {"acoustic", "vocoder"}
    original_ids = {
        role: source.info.info[INFO_KINDS[role]]
        for role in ROLES if role != "acoustic"
    }
    original_bytes = {
        role: source.info.models[role].read_bytes()
        for role in ROLES if role != "acoustic"
    }
    duration_style = source.info.info["duration_style"]
    f0_style = source.info.info["f0_style"]

    voice = Voice(source)
    assert set(voice.models) == {"acoustic", "vocoder"}
    assert not hasattr(voice, "duration")
    assert not hasattr(voice, "pitch")
    database = tmp_path / "synthv" / "databases" / "PreservedVoice"
    acoustic_version = source.models["acoustic"].version
    output = voice.save(
        database, name="PreservedVoice", version="101",
        acoustic_dnni_version=acoustic_version,
    )
    document = json.loads(output.read_text("utf-8"))
    assert document["duration_style"] == duration_style
    assert document["f0_style"] == f0_style
    for role, model_id in original_ids.items():
        kind = INFO_KINDS[role]
        assert document[kind] == model_id
        copied = tmp_path / "synthv" / "models" / kind / f"{model_id}.dnni"
        assert copied.read_bytes() == original_bytes[role]
