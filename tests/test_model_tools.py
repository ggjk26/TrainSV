from __future__ import annotations

import hashlib
import json
import struct
import shutil
from pathlib import Path

import pytest

from opensv_train.model_tools import replace_info_model, replace_info_models

DNNI_MAGIC = 0x7FCA00FF



class InfoVoiceForTest:
    def __init__(self, path: Path):
        self.path = path
        self.document = json.loads(path.read_text("utf-8"))
        self.models_root = path.parents[2] / "models"

    def all_models_resolve(self) -> bool:
        return all(
            (self.models_root / folder / f"{self.document[key]}.dnni").is_file()
            for key, folder in (("duration", "duration"), ("f0", "f0"), ("acoustic", "acoustic"), ("vocoder", "vocoder"))
        )


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


def test_batch_replace_duration_pitch_and_vocoder_with_explicit_models_root(tmp_path: Path):
    source, source_models = _make_info_voice(tmp_path / "source-package")
    source_bytes = source.read_bytes()
    original_info = json.loads(source_bytes.decode("utf-8"))
    installed_models = tmp_path / "synthv-install" / "models"
    for role, original in source_models.items():
        target = installed_models / original.parent.name / original.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)

    # Reproduce the reported issue: the default path derived from info.nofs is
    # missing a referenced model, but the installed Synthesizer V models root has it.
    source_models["duration"].unlink()
    replacements = {}
    for role, payload in (
        ("duration", b"external duration model"),
        ("pitch", b"external pitch/f0 model"),
        ("vocoder", b"external vocoder model"),
    ):
        path = tmp_path / "other-voicebank" / f"{role}.dnni"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_dnni_v3(payload))
        replacements[role] = path
    output = tmp_path / "output-package" / "database" / "voices" / "source" / "info.101.nofs"

    result = replace_info_models(source, replacements, output, models_root=installed_models)

    assert set(result["replaced"]) == {"duration", "pitch", "vocoder"}
    assert result["models_root"] == str(installed_models.resolve())
    document = json.loads(output.read_text("utf-8"))
    assert document["acoustic"] == original_info["acoustic"]
    assert document["duration"] == hashlib.md5(replacements["duration"].read_bytes()).hexdigest()
    assert document["f0"] == hashlib.md5(replacements["pitch"].read_bytes()).hexdigest()
    assert document["vocoder"] == hashlib.md5(replacements["vocoder"].read_bytes()).hexdigest()
    assert document["version"] == "101"
    for role, path in replacements.items():
        key = {"duration": "duration", "pitch": "f0", "vocoder": "vocoder"}[role]
        target = output.parent.parent.parent / "models" / key / f"{document[key]}.dnni"
        assert target.read_bytes() == path.read_bytes()
    acoustic_target = output.parent.parent.parent / "models" / "acoustic" / f"{document['acoustic']}.dnni"
    assert acoustic_target.read_bytes() == source_models["acoustic"].read_bytes()
    assert source.read_bytes() == source_bytes


def test_batch_replace_requires_at_least_one_role_and_reports_missing_model_root(tmp_path: Path):
    source, _ = _make_info_voice(tmp_path / "source-package")
    output = tmp_path / "out" / "info.101.nofs"
    with pytest.raises(ValueError, match="select at least one model"):
        replace_info_models(source, {}, output)

    missing_root = tmp_path / "not-the-models-folder"
    missing_root.mkdir()
    with pytest.raises(FileNotFoundError, match="Choose the Synthesizer V models directory"):
        replace_info_models(source, {"pitch": tmp_path / "pitch.dnni"}, output, models_root=missing_root)


def test_output_package_folder_creates_writable_self_contained_layout(tmp_path: Path):
    source, source_models = _make_info_voice(tmp_path / "source-package")
    source_bytes = source.read_bytes()
    replacement = tmp_path / "external" / "pitch.dnni"
    replacement.parent.mkdir(parents=True)
    replacement_bytes = _dnni_v3(b"external pitch model")
    replacement.write_bytes(replacement_bytes)
    output_root = tmp_path / "TrainSV-output"

    result = replace_info_models(
        source,
        {"pitch": replacement},
        output_root,
    )

    output_info = output_root / "database" / "voices" / "Source" / "info.100.nofs"
    output_models = output_root / "database" / "models"
    assert Path(result["output"]) == output_info
    assert Path(result["output_package"]) == output_root.resolve()
    assert output_info.is_file()
    document = json.loads(output_info.read_text("utf-8"))
    assert document["acoustic"] == "3" * 32
    assert document["f0"] == hashlib.md5(replacement_bytes).hexdigest()
    assert (output_models / "f0" / f"{document['f0']}.dnni").read_bytes() == replacement_bytes
    assert (output_models / "acoustic" / f"{document['acoustic']}.dnni").read_bytes() == source_models["acoustic"].read_bytes()
    assert source.read_bytes() == source_bytes
    assert InfoVoiceForTest(output_info).all_models_resolve()


def test_shallow_standalone_output_path_is_rejected_instead_of_writing_users_models(tmp_path: Path):
    source, _ = _make_info_voice(tmp_path / "source-package")
    replacement = tmp_path / "external.dnni"
    replacement.write_bytes(_dnni_v3(b"candidate"))
    shallow_output = tmp_path / "info.101.nofs"

    with pytest.raises(ValueError, match="output info.nofs path is too shallow.*output package folder"):
        replace_info_models(source, {"pitch": replacement}, shallow_output)
    assert not (tmp_path / "models").exists()


def test_output_cannot_be_the_models_folder_itself(tmp_path: Path):
    source, _ = _make_info_voice(tmp_path / "source-package")
    replacement = tmp_path / "external.dnni"
    replacement.write_bytes(_dnni_v3(b"candidate"))
    models_output = tmp_path / "SynthV" / "models"
    for folder in ("duration", "f0", "acoustic", "vocoder"):
        (models_output / folder).mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="output package folder cannot be a models directory"):
        replace_info_models(source, {"duration": replacement}, models_output)
    assert not (models_output / "database").exists()


def test_standalone_info_output_refuses_to_write_under_users_root(tmp_path: Path):
    source, _ = _make_info_voice(tmp_path / "source-package")
    replacement = tmp_path / "external.dnni"
    replacement.write_bytes(_dnni_v3(b"candidate"))
    # Emulate C:\\Users\\voices\\Source\\info.101.nofs: the computed
    # output models directory would be C:\\Users\\models.
    shallow_user_root = tmp_path / "Users" / "voices" / "Source" / "info.101.nofs"
    with pytest.raises(ValueError, match="Windows system/user root"):
        replace_info_models(source, {"pitch": replacement}, shallow_user_root)
    assert not (tmp_path / "Users" / "models").exists()
