from __future__ import annotations

import hashlib
import struct
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .dnni import DnniFile
from .nofs import InfoVoice, NofsFile, is_info_voice

ROLES = ("duration", "pitch", "acoustic", "vocoder")
INFO_KINDS = {"duration": "duration", "pitch": "f0", "acoustic": "acoustic", "vocoder": "vocoder"}


@dataclass
class VoiceFiles:
    path: Path
    models: dict[str, DnniFile]
    metadata: dict[str, str]
    nofs: NofsFile | None = None
    info: InfoVoice | None = None

    @property
    def vocal_modes(self) -> list[str]:
        if self.info is not None:
            return list(self.info.vocal_modes)
        return self.metadata.get(".timbre_styles", "").split()

    @property
    def rap_languages(self) -> list[str]:
        return self.metadata.get(".feature_rap_languages", "").split()


def load_voice(path) -> VoiceFiles:
    path = Path(path).expanduser()
    if is_info_voice(path):
        info = InfoVoice.load(path)
        models = {role: DnniFile.load(info.models[role]) for role in ROLES}
        metadata = {".name": info.info.get("name", ""), ".language": info.info.get("language", "")}
        return VoiceFiles(path, models, metadata, info=info)
    nofs = NofsFile.load(path)
    keys = nofs.model_keys()
    models = {role: DnniFile.parse(nofs.read(keys[role])) for role in ROLES}
    return VoiceFiles(path, models, nofs.metadata(), nofs=nofs)


def model_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def write_nofs_voice(voice: VoiceFiles, output: Path, name: str | None = None) -> Path:
    keys = voice.nofs.model_keys()
    replacements = {keys[role]: voice.models[role].serialize() for role in ROLES}
    if name is not None:
        replacements[b".name"] = name.encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(voice.nofs.replace(replacements))
    return output


def write_info_voice(voice: VoiceFiles, database_directory: Path, version: str, name: str | None = None) -> Path:
    info = voice.info
    models_root = database_directory.parent.parent / "models"
    document = info.to_json()
    for role in ROLES:
        data = voice.models[role].serialize()
        original = info.models[role]
        reference = original.stem if data == original.read_bytes() else model_hash(data)
        target = models_root / INFO_KINDS[role] / f"{reference}.dnni"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        document[INFO_KINDS[role]] = reference
    if name is not None:
        document["name"] = name
    document["version"] = version
    database_directory.mkdir(parents=True, exist_ok=True)
    output = database_directory / f"info.{version}.nofs"
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2, separators=(",", " : ")), "utf-8")
    return output


class Voice(torch.nn.Module):

    def __init__(self, files: VoiceFiles):
        super().__init__()
        from .acoustic import AcousticModel
        from .duration import DurationModel
        from .pitch import PitchModel
        from .vocoder import Vocoder

        self.files = files
        info = files.info
        if info is None:
            self.duration = DurationModel(files.models["duration"])
            self.pitch = PitchModel(files.models["pitch"])
            self.acoustic = AcousticModel(files.models["acoustic"], files.vocal_modes)
        else:
            self.duration = DurationModel(files.models["duration"], info.duration_voice)
            self.pitch = PitchModel(files.models["pitch"], info.pitch_speaker)
            self.acoustic = AcousticModel(files.models["acoustic"], list(info.vocal_modes), info.acoustic_speaker, info.default_style, list(info.vocal_modes.values()))
        self.vocoder = Vocoder(files.models["vocoder"])

    @classmethod
    def load(cls, path) -> "Voice":
        return cls(load_voice(path))

    @property
    def models(self) -> dict[str, torch.nn.Module]:
        return {"duration": self.duration, "pitch": self.pitch, "acoustic": self.acoustic, "vocoder": self.vocoder}

    def export(self) -> VoiceFiles:
        from .layers import export_parameters

        for model in self.models.values():
            export_parameters(model)
        info = self.files.info
        if info is not None:
            info.duration_voice = self.duration.voice.detach().cpu().numpy()
            info.pitch_speaker = self.pitch.speaker.detach().cpu().numpy()
            info.acoustic_speaker = self.acoustic.speaker.detach().cpu().numpy()
            info.default_style = self.acoustic.default_style.detach().cpu().numpy()
            info.vocal_modes = {name: style.detach().cpu().numpy() for name, style in zip(self.acoustic.vocal_modes, self.acoustic.mode_styles)}
        return self.files

    def set_vocal_modes(self, names: list[str]) -> None:
        acoustic = self.acoustic
        if names == acoustic.vocal_modes:
            return
        if self.files.info is None:
            raise ValueError("vocal modes can only be added to or removed from info voices")
        current = dict(zip(acoustic.vocal_modes, acoustic.mode_styles))
        acoustic.mode_styles = torch.nn.ParameterList(
            current[name] if name in current else torch.nn.Parameter(acoustic.default_style.detach().clone()) for name in names
        )
        acoustic.vocal_modes = list(names)

    def save(self, output, name: str | None = None, version: str | None = None) -> Path:
        files = self.export()
        output = Path(output).expanduser()
        if files.info is None:
            return write_nofs_voice(files, output, name)
        return write_info_voice(files, output, version or files.info.info.get("version", "100"), name)

def initialize_pitch_from_file(voice: Voice, source_path) -> int:
    """Initialize a voice's Pitch network from a strictly compatible DNNI model.

    The target voice's speaker vector is deliberately preserved. A source is
    accepted only when its model layout, phone tables, normalization buffers,
    non-parameter node payloads, and parameter shapes match.
    """
    from .pitch import PitchModel

    target_file = voice.files.models["pitch"]
    target = voice.pitch
    try:
        source_file = DnniFile.load(source_path)
        source = PitchModel(source_file)
    except (OSError, ValueError, KeyError, struct.error) as error:
        raise ValueError(f"the selected file is not a supported Synthesizer V Pitch DNNI model: {error}") from error

    if _phone_set_signature(target.phone_sets) != _phone_set_signature(source.phone_sets):
        raise ValueError("Pitch model phoneme tables do not match the selected voice")
    if _phone_set_signature([target.unified]) != _phone_set_signature([source.unified]):
        raise ValueError("Pitch model unified phoneme tables do not match the selected voice")

    def same_tree(a, b, root=False):
        if a.type_id != b.type_id or len(a.children) != len(b.children):
            return False
        parameter_nodes = {"prim0", "prim1", "prim2", "prim3", "prim4", "prim5"}
        if not root and a.type not in parameter_nodes and a.payload != b.payload:
            return False
        return all(same_tree(x, y) for x, y in zip(a.children, b.children))

    if not same_tree(target_file.root, source_file.root, root=True):
        raise ValueError("Pitch model network layout/configuration is not an exact match")

    target_buffers = dict(target.named_buffers())
    source_buffers = dict(source.named_buffers())
    if target_buffers.keys() != source_buffers.keys():
        raise ValueError("Pitch model normalization/mask layout does not match")
    for name, target_value in target_buffers.items():
        source_value = source_buffers[name]
        if target_value.shape != source_value.shape or not torch.equal(
            target_value.detach().cpu(), source_value.detach().cpu()
        ):
            raise ValueError(f"Pitch model normalization or sparsity mask differs at {name}")

    def trainable_parameters(model):
        # Keep the target singer identity. Its parameter can be stored inside the
        # DNNI file or in the info voice's metadata, depending on voice format.
        return {
            name: value for name, value in model.named_parameters()
            if not name.startswith("speaker")
        }

    target_params = trainable_parameters(target)
    source_params = trainable_parameters(source)
    if target_params.keys() != source_params.keys():
        raise ValueError("Pitch model trainable parameter layout does not match")
    for name, target_value in target_params.items():
        if target_value.shape != source_params[name].shape:
            raise ValueError(f"Pitch model parameter shape differs at {name}")

    with torch.no_grad():
        for name, target_value in target_params.items():
            target_value.copy_(source_params[name].detach().to(target_value.device))
    target.expression = source.expression
    target.expression_strength = source.expression_strength
    target.rap_expression = source.rap_expression
    target_file.root.payload = source_file.root.payload[:12] + target_file.root.payload[12:]
    return len(target_params)


def _phone_set_signature(phone_sets) -> tuple:
    return tuple(
        (phone_set.name, tuple(phone_set.symbols), tuple(phone_set.categories),
         tuple(phone_set.unified), tuple(phone_set.classes))
        for phone_set in phone_sets
    )
