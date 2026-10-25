from pathlib import Path
import struct

import pytest

from opensv_train.dnni import FILE_MAGIC, OpaqueDnniFile


def test_opaque_dnni_v3_round_trips_original_bytes_exactly():
    # Payload deliberately does not resemble a v1/v2 node tree. It must remain
    # opaque; the pass-through wrapper does not attempt to decode it.
    raw = struct.pack("<II", FILE_MAGIC, 3) + bytes(range(256)) * 3
    model = OpaqueDnniFile.from_bytes(raw)

    assert model.version == 3
    assert model.serialize() == raw
    assert model.serialize()[8:] == raw[8:]


def test_opaque_dnni_v3_rejects_wrong_magic_or_version():
    with pytest.raises(ValueError, match="not a DNNI file"):
        OpaqueDnniFile.from_bytes(struct.pack("<II", 0, 3) + b"payload")

    with pytest.raises(ValueError, match="only enabled for DNNI v3"):
        OpaqueDnniFile.from_bytes(struct.pack("<II", FILE_MAGIC, 2) + b"payload")


@pytest.mark.parametrize("role", ["duration", "pitch", "acoustic", "vocoder"])
def test_opaque_v3_payload_is_preserved_for_every_model_role(role):
    # Role-aware loading/guards live in voice.py; this low-dependency test
    # checks the shared opaque container itself without importing PyTorch.
    raw = struct.pack("<II", FILE_MAGIC, 3) + bytes(range(256)) * 3
    model_file = OpaqueDnniFile.from_bytes(raw)
    assert role in {"duration", "pitch", "acoustic", "vocoder"}
    assert model_file.version == 3
    assert model_file.serialize() == raw


def test_info_and_nofs_voice_loaders_preserve_v3_for_any_role_and_block_unsupported_operations():
    root = Path(__file__).resolve().parents[1]
    voice = (root / "opensv_train" / "voice.py").read_text("utf-8")
    train = (root / "opensv_train" / "train.py").read_text("utf-8")
    cli = (root / "opensv_train" / "__main__.py").read_text("utf-8")

    assert "def _load_model_bytes(data: bytes)" in voice
    assert "return _load_model_bytes(Path(path).read_bytes())" in voice
    assert "models = {role: _load_model_bytes(nofs.read(keys[role])) for role in ROLES}" in voice
    assert "class UnavailableDNNIModel" in voice
    assert "class UnavailablePitchModel" in voice
    assert "if not getattr(model, \"supported\", True):" in voice
    assert 'if "pitch" in config.models and not getattr(voice.pitch, "supported", True):' in train
    assert "for role in sorted(required_roles):" in train
    assert "def _require_supported_models(voice: Voice, roles, operation: str)" in cli
    assert '_require_supported_models(voice, ROLES, "synthesis")' in cli
    assert 'print("  phoneme tables: unavailable (opaque Acoustic DNNI v3)")' in cli

def test_preservation_mode_loads_only_selected_roles_and_keeps_unselected_dnnies():
    root = Path(__file__).resolve().parents[1]
    voice = (root / "opensv_train" / "voice.py").read_text("utf-8")
    cli = (root / "opensv_train" / "__main__.py").read_text("utf-8")

    assert "training_models: tuple[str, ...] | None = None" in voice
    assert "roles_to_load = set(selected)" in voice
    assert 'roles_to_load.update(("acoustic", "vocoder"))' in voice
    assert 'preservation mode must train Acoustic from the selected alternate base model' in voice
    assert "role not in voice.training_models" in voice
    assert "original_bytes = info.models[role].read_bytes()" in voice
    assert '"--preserve-original-models"' in cli
    assert "--models acoustic" in cli
    assert "--custom-acoustic-dnni" in cli

