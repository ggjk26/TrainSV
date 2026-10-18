from pathlib import Path
import struct

import pytest

from opensv_train.dnni import FILE_MAGIC, DnniFile, OpaqueDnniFile


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


def test_info_voice_v3_fallback_is_pitch_only_and_blocks_pitch_training():
    root = Path(__file__).resolve().parents[1]
    voice = (root / "opensv_train" / "voice.py").read_text("utf-8")
    train = (root / "opensv_train" / "train.py").read_text("utf-8")
    cli = (root / "opensv_train" / "__main__.py").read_text("utf-8")

    assert "def _load_info_model(role: str, path: Path)" in voice
    assert 'if role == "pitch":' in voice
    assert "OpaqueDnniFile.from_bytes(data)" in voice
    assert "class UnavailablePitchModel" in voice
    assert "Pitch training, " in voice
    assert 'if "pitch" in config.models and not getattr(voice.pitch, "supported", True):' in train
    assert 'if "pitch" in models and not getattr(voice.pitch, "supported", True):' in cli
    assert 'if role == "pitch" and not getattr(model, "supported", True):' in cli
