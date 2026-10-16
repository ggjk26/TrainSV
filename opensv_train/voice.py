from __future__ import annotations

import hashlib
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



def initialize_pitch_model(voice: Voice, model_path: str | Path) -> Path:
    """Replace the voice's Pitch network with a compatible pretrained DNNI.

    The current voice's speaker embedding is retained. The candidate must load
    as a supported PitchModel, have the same phone-table definitions, and
    expose a speaker vector of the same shape. This deliberately rejects
    arbitrary DNNI files and models for a different phoneme frontend.
    """
    model_path = Path(model_path).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"pretrained Pitch DNNI not found: {model_path}")
    if model_path.suffix.lower() != ".dnni":
        raise ValueError("pretrained Pitch model must be a .dnni file")

    candidate_data = DnniFile.load(model_path)
    from .pitch import PitchModel

    try:
        candidate_pitch = PitchModel(candidate_data)
    except (ValueError, IndexError, KeyError, struct.error) as error:
        raise ValueError(
            f"{model_path} is not a Pitch DNNI architecture supported by this TrainSV build: {error}"
        ) from error

    def phone_set_signature(phone_set):
        return (
            phone_set.name,
            tuple(phone_set.symbols),
            tuple(phone_set.categories),
            tuple(phone_set.unified),
            tuple(phone_set.classes),
        )

    current_pitch = voice.pitch
    current_tables = tuple(phone_set_signature(table) for table in current_pitch.phone_sets)
    candidate_tables = tuple(phone_set_signature(table) for table in candidate_pitch.phone_sets)
    if current_tables != candidate_tables or phone_set_signature(current_pitch.unified) != phone_set_signature(candidate_pitch.unified):
        raise ValueError(
            "pretrained Pitch model uses different phone tables from the selected base voice; "
            "choose a model exported from a voice with matching language/phoneme definitions"
        )

    current_speaker = current_pitch.speaker_vector().detach().cpu().numpy()
    candidate_speaker = candidate_pitch.speaker_vector().detach().cpu().numpy()
    if current_speaker.shape != candidate_speaker.shape:
        raise ValueError(
            f"pretrained Pitch speaker vector has shape {candidate_speaker.shape}, "
            f"but the selected base voice requires {current_speaker.shape}"
        )

    if voice.files.info is None:
        # Keep the VectorParameter wrapper so Voice.export() writes the retained
        # speaker vector into the replacement DNNI for single-file NOFS voices.
        speaker = getattr(candidate_pitch, "speaker", None)
        value = getattr(speaker, "value", None)
        if value is None or tuple(value.shape) != tuple(current_speaker.shape):
            raise ValueError("pretrained Pitch model speaker embedding cannot be exported into a voice.nofs")
        with torch.no_grad():
            value.copy_(torch.from_numpy(current_speaker).to(value.device, value.dtype))
    else:
        # Shared-model info voices store this voice-specific embedding in info.nofs.
        candidate_pitch = PitchModel(candidate_data, speaker=current_speaker)

    voice.pitch = candidate_pitch
    voice.files.models["pitch"] = candidate_data
    return model_path.resolve()
