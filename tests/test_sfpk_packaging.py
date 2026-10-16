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


def _png(path: Path, payload=b"sample image"):
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + payload)
    return path


def test_build_sfpk_includes_optional_profile_and_background_pngs(tmp_path):
    models_root = _models_root(tmp_path)
    info_path = tmp_path / "info.nofs"
    info_path.write_text(json.dumps(_info_document()), encoding="utf-8")
    profile = _png(tmp_path / "avatar.png", b"avatar-pixels")
    background = _png(tmp_path / "backdrop.png", b"background-pixels")
    output = tmp_path / "with-artwork.sfpk"

    build_sfpk(
        voice=info_path,
        models_root=models_root,
        output=output,
        profile=profile,
        background=background,
    )

    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == [
            "config.json", "info.nofs", "profile.png", "background.png"
        ]
        assert archive.read("profile.png") == profile.read_bytes()
        assert archive.read("background.png") == background.read_bytes()
        assert archive.testzip() is None
        config = json.loads(archive.read("config.json"))
    # Preserve the booleans observed in the reference config.json sample.
    assert config["profile"] is True
    assert config["background"] is True


def test_build_sfpk_can_include_only_one_png_asset(tmp_path):
    models_root = _models_root(tmp_path)
    info_path = tmp_path / "info.nofs"
    info_path.write_text(json.dumps(_info_document()), encoding="utf-8")
    profile = _png(tmp_path / "avatar.png")
    output = tmp_path / "profile-only.sfpk"

    build_sfpk(
        voice=info_path,
        models_root=models_root,
        output=output,
        profile=profile,
    )

    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["config.json", "info.nofs", "profile.png"]
        assert archive.read("profile.png") == profile.read_bytes()


@pytest.mark.parametrize("asset_field", ["profile", "background"])
def test_build_sfpk_rejects_invalid_png_assets(tmp_path, asset_field):
    models_root = _models_root(tmp_path)
    info_path = tmp_path / "info.nofs"
    info_path.write_text(json.dumps(_info_document()), encoding="utf-8")
    bad_file = tmp_path / "not-a-png.png"
    bad_file.write_bytes(b"this is not a png")

    with pytest.raises(ValueError, match=r"must be a PNG file"):
        build_sfpk(
            voice=info_path,
            models_root=models_root,
            output=tmp_path / "bad-asset.sfpk",
            **{asset_field: bad_file},
        )


def test_build_sfpk_rejects_missing_png_path(tmp_path):
    models_root = _models_root(tmp_path)
    info_path = tmp_path / "info.nofs"
    info_path.write_text(json.dumps(_info_document()), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="profile.png file not found"):
        build_sfpk(
            voice=info_path,
            models_root=models_root,
            output=tmp_path / "missing-asset.sfpk",
            profile=tmp_path / "missing.png",
        )
