import numpy as np
import torch

from opensv_train.dnni import DnniFile, _read_quantization, read_matrix, write_matrix
from opensv_train.layers import MatrixParameter, VectorParameter
from opensv_train.voice import ROLES, load_voice


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
