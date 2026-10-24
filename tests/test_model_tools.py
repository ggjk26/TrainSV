from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from opensv_train.model_tools import replace_info_model

DNNI_MAGIC = 0x7FCA00FF


def _dnni_v3(payload: bytes) -> bytes:
    return struct.pack("<II", DNNI_MAGIC, 3) + payload


def _make_info_voice(root: Path) -> tuple[Path, dict[str, Path]]:
    # InfoVoice resolves <database-root>/models from info.nofs three parents up.
    info_path = root / "database" / "voices" / "source" / "info.100.nofs"
    models_root = info_path.parent.parent.parent / "models"
    ids = {
        "duration": "1" * 32,
        "f0": "2" * 32,
        "acoustic": "3" * 32,
        "vocoder": "4" * 32,
    }
    models: dict[str, Path] = {}
    for kind, model_id in ids.items():
        path = models_root / kind / f"{model_id}.dnni"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_dnni_v3((kind + "-original").encode("ascii")))
        models[kind] = path
    vector = (b"\x00\x00\x80\x3f" * 32).hex().upper()
    info = {
        **ids,
        "version": "100",
        "name": "Source",
        "vendor": "Tests",
        "support_languages": ["mandarin"],
        "duration_style": vector,
        "f0_style": vector,
        "styles": [
            {"name": "(base)", "data": vector},
            {"name": "(default)", "data": vector},
        ],
    }
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info), encoding="utf-8")
    return info_path, models


def test_replace_model_creates_new_info_and_preserves_source(tmp_path: Path):
    source, source_models = _make_info_voice(tmp_path / "source-package")
    source_bytes = source.read_bytes()
    original_model_bytes = {role: path.read_bytes() for role, path in source_models.items()}
    replacement = tmp_path / "external" / "different-vocoder.dnni"
    replacement.parent.mkdir(parents=True)
    replacement_bytes = _dnni_v3(b"external opaque vocoder candidate")
    replacement.write_bytes(replacement_bytes)
    output = source.parent / "info.101.nofs"

    result = replace_info_model(source, "vocoder", replacement, output)

    assert output.is_file()
    assert source.read_bytes() == source_bytes
    assert {role: path.read_bytes() for role, path in source_models.items()} == original_model_bytes
    document = json.loads(output.read_text("utf-8"))
    assert document["vocoder"] == hashlib.md5(replacement_bytes).hexdigest()
    assert document["duration"] == "1" * 32
    assert document["f0"] == "2" * 32
    assert document["acoustic"] == "3" * 32
    target = source.parent.parent.parent / "models" / "vocoder" / f"{document['vocoder']}.dnni"
    assert target.read_bytes() == replacement_bytes
    assert result["dnni_version"] == 3
    assert result["role_checked"] is False
    assert "could not be verified" in result["warning"]


def test_replace_model_rejects_acoustic_and_binary_voice_nofs(tmp_path: Path):
    source, _ = _make_info_voice(tmp_path / "source-package")
    model = tmp_path / "external.dnni"
    model.write_bytes(_dnni_v3(b"payload"))
    output = source.parent / "info.101.nofs"

    with pytest.raises(ValueError, match="Acoustic"):
        replace_info_model(source, "acoustic", model, output)

    binary = tmp_path / "voice.nofs"
    binary.write_bytes(b"binary-nofs")
    with pytest.raises(ValueError, match="info.nofs JSON"):
        replace_info_model(binary, "pitch", model, output)


def test_replace_model_never_overwrites_existing_output(tmp_path: Path):
    source, _ = _make_info_voice(tmp_path / "source-package")
    model = tmp_path / "external.dnni"
    model.write_bytes(_dnni_v3(b"payload"))
    output = source.parent / "info.101.nofs"
    output.write_text("keep-me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output already exists"):
        replace_info_model(source, "pitch", model, output)

    assert output.read_text("utf-8") == "keep-me"
