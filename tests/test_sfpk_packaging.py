from __future__ import annotations

import hashlib
import json
import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from opensv_train.sfpk import build_sfpk


MODEL_IDS = {
    "acoustic": "a" * 32,
    "vocoder": "b" * 32,
    "duration": "c" * 32,
    "f0": "d" * 32,
}
VECTOR = "AB" * 128
DURATION_VECTOR = "CD" * 512


def _info_document(name="Test Singer", version="101"):
    return {
        "name": name,
        "version": version,
        "vendor": "TrainSV tests",
        "language": "mandarin",
        "phoneset": "xsampa",
        "support_languages": ["mandarin"],
        **MODEL_IDS,
        "styles": [
            {"name": "(base)", "data": VECTOR},
            {"name": "(default)", "data": VECTOR},
        ],
        "duration_style": DURATION_VECTOR,
        "f0_style": VECTOR,
    }


def _models_root(tmp_path, ids=MODEL_IDS):
    root = tmp_path / "models"
    for kind, model_id in ids.items():
        target = root / kind / f"{model_id}.dnni"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"dummy-{kind}".encode())
    return root


def test_build_sfpk_from_existing_info_json(tmp_path):
    models_root = _models_root(tmp_path)
    info_path = tmp_path / "info.nofs"
    info_path.write_text(json.dumps(_info_document()), encoding="utf-8")
    output = tmp_path / "Test Singer.sfpk"

    result = build_sfpk(
        voice=info_path,
        models_root=models_root,
        output=output,
        name="Localized Singer",
        version="102",
    )

    assert result == output.resolve()
    with zipfile.ZipFile(result) as archive:
        assert archive.namelist() == ["config.json", "info.nofs"]
        assert archive.testzip() is None
        config = json.loads(archive.read("config.json"))
        info = json.loads(archive.read("info.nofs"))

    assert config["name"] == info["name"] == "Localized Singer"
    assert config["version"] == info["version"] == "102"
    assert config["profile"] is True
    assert config["background"] is True
    assert config["models"] == []
    assert config["model_types"] == []
    assert config["phonesets"] == []
    assert isinstance(config["translations"], str)
    assert json.loads(config["translations"])["English"] == '"Localized Singer"'
    assert info["acoustic"] == MODEL_IDS["acoustic"]
    assert info["f0_style"] == VECTOR
    assert info["duration_style"] == DURATION_VECTOR
    assert len(info["duration_style"]) == 128 * 8
    assert len(info["f0_style"]) == 32 * 8


def test_build_sfpk_rejects_missing_model_dependency(tmp_path):
    models_root = tmp_path / "models"
    info_path = tmp_path / "info.nofs"
    info_path.write_text(json.dumps(_info_document()), encoding="utf-8")
    models_root.mkdir()

    with pytest.raises(FileNotFoundError, match="model dependency is missing"):
        build_sfpk(
            voice=info_path,
            models_root=models_root,
            output=tmp_path / "voice.sfpk",
        )


def test_build_sfpk_requires_sfpk_extension(tmp_path):
    models_root = _models_root(tmp_path)
    info_path = tmp_path / "info.nofs"
    info_path.write_text(json.dumps(_info_document()), encoding="utf-8")

    with pytest.raises(ValueError, match=r"end in .sfpk"):
        build_sfpk(
            voice=info_path,
            models_root=models_root,
            output=tmp_path / "voice.zip",
        )


def test_build_sfpk_exports_binary_voice_models_and_embedding_vectors(tmp_path, monkeypatch):
    voice_path = tmp_path / "trained-voice.nofs"
    voice_path.write_bytes(b"regular binary voice.nofs")
    models_root = tmp_path / "models"
    models_root.mkdir()

    class Vector:
        def __init__(self, size=32):
            self.size = size

        def reshape(self, *_shape):
            return self

    model_bytes = {
        "duration": b"duration model payload",
        "pitch": b"f0 model payload",
        "acoustic": b"acoustic model payload",
        "vocoder": b"vocoder model payload",
    }
    fake_files = SimpleNamespace(
        info=None,
        nofs=object(),
        metadata={
            ".name": "Source Voice",
            ".vendor": "Original Vendor",
            ".version": "106",
            ".language": "mandarin",
            ".phoneset": "xsampa",
            ".multi": "mandarin english",
        },
        models={role: SimpleNamespace(serialize=lambda role=role: model_bytes[role]) for role in model_bytes},
    )
    vector = Vector()

    class FakeVoice:
        files = fake_files
        duration = SimpleNamespace(voice_vector=lambda: Vector(128))
        pitch = SimpleNamespace(speaker=lambda: vector)
        acoustic = SimpleNamespace(
            vocal_modes=[],
            speaker_vector=lambda: vector,
            default_style_vector=lambda: vector,
            mode_style_vectors=lambda: [],
        )

        @classmethod
        def load(cls, _path):
            return cls()

    fake_voice_module = types.ModuleType("opensv_train.voice")
    fake_voice_module.INFO_KINDS = {
        "duration": "duration",
        "pitch": "f0",
        "acoustic": "acoustic",
        "vocoder": "vocoder",
    }
    fake_voice_module.ROLES = ("duration", "pitch", "acoustic", "vocoder")
    fake_voice_module.Voice = FakeVoice
    fake_nofs_module = types.ModuleType("opensv_train.nofs")
    fake_nofs_module.vector_hex = lambda value: "AB" * (value.size * 4)
    monkeypatch.setitem(sys.modules, "opensv_train.voice", fake_voice_module)
    monkeypatch.setitem(sys.modules, "opensv_train.nofs", fake_nofs_module)

    output = tmp_path / "Source Voice.sfpk"
    result = build_sfpk(
        voice=voice_path,
        models_root=models_root,
        output=output,
    )

    assert result.is_file()
    for role, payload in model_bytes.items():
        kind = fake_voice_module.INFO_KINDS[role]
        model_id = hashlib.md5(payload).hexdigest()
        assert (models_root / kind / f"{model_id}.dnni").read_bytes() == payload

    with zipfile.ZipFile(result) as archive:
        assert archive.namelist() == ["config.json", "info.nofs"]
        info = json.loads(archive.read("info.nofs"))
        config = json.loads(archive.read("config.json"))
    assert info["name"] == "Source Voice"
    assert info["vendor"] == "Original Vendor"
    assert info["version"] == "106"
    assert info["support_languages"] == ["mandarin", "english"]
    assert info["acoustic"] == hashlib.md5(model_bytes["acoustic"]).hexdigest()
    assert len(info["duration_style"]) == 128 * 8
    assert len(info["f0_style"]) == 32 * 8
    assert len(info["styles"][0]["data"]) == 32 * 8
    assert config["name"] == info["name"]
