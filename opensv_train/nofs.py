from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NOFS_MAGIC = 0xF580
NOFS_VERSION = 10
CONFIG_KEY = b"\x91"
PITCH_MODEL_KEY = b"f0model-dds"


@dataclass
class Entry:
    key: bytes
    value_offset: int
    value_size: int

    @property
    def name(self) -> str | None:
        try:
            text = self.key.decode("utf-8")
        except UnicodeDecodeError:
            return None
        return text if all(ord(c) >= 0x20 and c != "\x7f" for c in text) else None


class NofsFile:
    def __init__(self, path: Path, data: bytes, index_size: int, entries: list[Entry]):
        self.path = path
        self.data = data
        self.index_size = index_size
        self.entries = entries
        self._by_key = {entry.key: entry for entry in entries}

    @classmethod
    def load(cls, path) -> NofsFile:
        path = Path(path)
        data = path.read_bytes()
        magic, version, length = struct.unpack_from("<IIQ", data, 0)
        if magic != NOFS_MAGIC or version != NOFS_VERSION or length != len(data):
            raise ValueError(f"{path} is not a supported NOFS v10 container")
        index_size, block_type = struct.unpack_from("<IH", data, 16)
        if block_type != 0x1000 or struct.unpack_from("<I", data, 16 + index_size - 4)[0] != index_size:
            raise ValueError("invalid NOFS index block")
        entries = []
        offset = 16 + index_size
        while offset < len(data):
            record_size, record_type, key_size = struct.unpack_from("<IHH", data, offset)
            if record_type != 1 or record_size < 16 + key_size or offset + record_size > len(data):
                raise ValueError(f"invalid NOFS record at {offset}")
            key = data[offset + 8:offset + 8 + key_size]
            (value_size,) = struct.unpack_from("<I", data, offset + 8 + key_size)
            if value_size + key_size + 16 != record_size or struct.unpack_from("<I", data, offset + record_size - 4)[0] != record_size:
                raise ValueError(f"inconsistent NOFS record at {offset}")
            entries.append(Entry(key, offset + 12 + key_size, value_size))
            offset += record_size
        return cls(path, data, index_size, entries)

    def index_offsets(self) -> list[int]:
        return [struct.unpack_from("<Q", self.data, 16 + 6 + 12 * slot + 4)[0] for slot in range(len(self.entries))]

    def record_offsets(self) -> list[int]:
        return [entry.value_offset - 12 - len(entry.key) for entry in self.entries]

    def has(self, key: bytes) -> bool:
        return key in self._by_key

    def read(self, key: bytes) -> bytes:
        entry = self._by_key[key]
        return self.data[entry.value_offset:entry.value_offset + entry.value_size]

    def metadata(self) -> dict[str, str]:
        return {entry.name: self.read(entry.key).decode("utf-8") for entry in self.entries if entry.name and entry.name.startswith(".")}

    def configuration(self) -> dict[str, list]:
        return parse_configuration(self.read(CONFIG_KEY))

    def model_keys(self) -> dict[str, bytes]:
        fields = self.configuration()
        keys = {
            "duration": fields["model_duration"][0],
            "acoustic": fields["model_timbre_pred"][0],
            "vocoder": fields["model_vocoder"][0],
            "pitch": PITCH_MODEL_KEY,
        }
        for role, architecture in (("duration", "model_duration_arch"), ("acoustic", "model_timbre_arch"), ("vocoder", "model_vocoder_arch")):
            expected = {"duration": b"gen2a", "acoustic": b"dds", "vocoder": b"nhv"}[role]
            if fields[architecture][0] != expected:
                raise ValueError(f"{role} architecture {fields[architecture][0]!r} is not supported")
        return keys

    def replace(self, replacements: dict[bytes, bytes]) -> bytes:
        for key in replacements:
            if key not in self._by_key:
                raise KeyError(key)
        if all(len(value) == self._by_key[key].value_size for key, value in replacements.items()):
            data = bytearray(self.data)
            for key, value in replacements.items():
                entry = self._by_key[key]
                data[entry.value_offset:entry.value_offset + entry.value_size] = value
            return bytes(data)
        records = []
        moved = {}
        offset = 16 + self.index_size
        for entry in self.entries:
            value = replacements.get(entry.key, self.data[entry.value_offset:entry.value_offset + entry.value_size])
            record_size = 16 + len(entry.key) + len(value)
            records.append(struct.pack("<IHH", record_size, 1, len(entry.key)) + entry.key + struct.pack("<I", len(value)) + value + struct.pack("<I", record_size))
            moved[entry.value_offset - 12 - len(entry.key)] = offset
            offset += record_size
        index = bytearray(self.data[16:16 + self.index_size])
        for slot in range(len(self.entries)):
            position = 6 + 12 * slot + 4
            (old,) = struct.unpack_from("<Q", index, position)
            if old not in moved:
                raise ValueError("the NOFS index has an unrecognised layout; cannot resize entries")
            struct.pack_into("<Q", index, position, moved[old])
        return struct.pack("<IIQ", NOFS_MAGIC, NOFS_VERSION, offset) + bytes(index) + b"".join(records)


def decode_config_name(key: bytes) -> bytes:
    state = 0xBEFCB5E0
    output = bytearray()
    for byte in key:
        output.append(byte ^ (state & 0xFF))
        state = (state * 0x41C64E6D + 0x3039) & 0xFFFFFFFF
    return bytes(output)


def parse_configuration(data: bytes) -> dict[str, list]:
    signature, count = struct.unpack_from("<II", data, 0)
    if signature != 0xFEFF:
        raise ValueError("invalid voice configuration signature")
    position = 8
    fields: dict[str, list] = {}
    for _ in range(count):
        value_type, _ordinal, key_size = struct.unpack_from("<IIH", data, position)
        position += 10
        name = decode_config_name(data[position:position + key_size]).decode("utf-8")
        position += key_size
        (value_count,) = struct.unpack_from("<I", data, position)
        position += 4
        values = []
        for _ in range(value_count):
            if value_type == 0:
                (size,) = struct.unpack_from("<H", data, position)
                values.append(data[position + 2:position + 2 + size])
                position += 2 + size
            elif value_type == 1:
                values.append(struct.unpack_from("<d", data, position)[0])
                position += 8
            elif value_type == 2:
                values.append(struct.unpack_from("<I", data, position)[0])
                position += 4
            else:
                raise ValueError(f"unsupported configuration value type {value_type}")
        fields[name] = values
    if position != len(data):
        raise ValueError("unexpected bytes after the voice configuration")
    return fields


def hex_vector(text: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(text), "<f4").copy()


def vector_hex(values: np.ndarray) -> str:
    return np.asarray(values, dtype="<f4").tobytes().hex().upper()


@dataclass
class InfoVoice:

    path: Path
    info: dict
    models: dict[str, Path]
    duration_voice: np.ndarray
    pitch_speaker: np.ndarray
    acoustic_speaker: np.ndarray
    default_style: np.ndarray
    vocal_modes: dict[str, np.ndarray] = field(default_factory=dict)
    mode_extras: dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(cls, path) -> InfoVoice:
        path = Path(path)
        info = json.loads(path.read_text("utf-8"))
        root = path.parent.parent.parent / "models"
        models = {}
        for role, kind in (("duration", "duration"), ("pitch", "f0"), ("acoustic", "acoustic"), ("vocoder", "vocoder")):
            models[role] = root / kind / f"{info[kind]}.dnni"
            if not models[role].is_file():
                raise FileNotFoundError(models[role])
        styles = {style["name"]: hex_vector(style["data"]) for style in info["styles"]}
        return cls(
            path, info, models,
            hex_vector(info["duration_style"]), hex_vector(info["f0_style"]),
            styles.pop("(base)"), styles.pop("(default)"), styles,
        )

    def to_json(self) -> dict:
        info = dict(self.info)
        info["duration_style"] = vector_hex(self.duration_voice)
        info["f0_style"] = vector_hex(self.pitch_speaker)
        vectors = {"(base)": self.acoustic_speaker, "(default)": self.default_style, **self.vocal_modes}
        original = {style["name"]: style for style in self.info["styles"]}
        info["styles"] = []
        for name, vector in vectors.items():
            style = {**original.get(name, {"name": name}), "data": vector_hex(vector)}
            if name in self.mode_extras:
                style["extra"] = vector_hex(np.array([self.mode_extras[name]]))
            info["styles"].append(style)
        return info


def is_info_voice(path) -> bool:
    with open(path, "rb") as stream:
        return stream.read(1) == b"{"
