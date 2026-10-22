from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

from .dnni import DnniFile, OpaqueDnniFile, serialize_model_for_output
from .nofs import InfoVoice, NofsFile, is_info_voice

ROLES = ("duration", "pitch", "acoustic", "vocoder")
INFO_KINDS = {"duration": "duration", "pitch": "f0", "acoustic": "acoustic", "vocoder": "vocoder"}


@dataclass
class VoiceFiles:
    path: Path
    models: dict[str, DnniFile | OpaqueDnniFile]
    metadata: dict[str, str]
    nofs: NofsFile | None = None
    info: InfoVoice | None = None
    preserve_non_acoustic_models: bool = False

    @property
    def vocal_modes(self) -> list[str]:
        if self.info is not None:
            return list(self.info.vocal_modes)
        return self.metadata.get(".timbre_styles", "").split()

    @property
    def rap_languages(self) -> list[str]:
        return self.metadata.get(".feature_rap_languages", "").split()


def _load_model_bytes(data: bytes) -> DnniFile | OpaqueDnniFile:
    """Parse a supported DNNI model or preserve a v3 file without interpreting it.

    DNNI v3 is only recognized by its container header here. Its payload,
    graph, tensors, and inference behavior remain unsupported.
    """
    try:
        return DnniFile.parse(data)
    except ValueError as parse_error:
        try:
            return OpaqueDnniFile.from_bytes(data)
        except ValueError:
            raise parse_error


def _load_info_model(role: str, path: Path) -> DnniFile | OpaqueDnniFile:
    """Load one model role while preserving an opaque DNNI v3 dependency.

    The role is retained by the caller for role-specific safety checks; the
    v3 bytes are not decoded or treated as an executable network.
    """
    return _load_model_bytes(Path(path).read_bytes())


class UnavailableDNNIModel(torch.nn.Module):
    """Placeholder for an opaque DNNI dependency that cannot run inference."""

    supported = False

    def __init__(self, role: str, version: int):
        super().__init__()
        self.role = role
        self.version = version
        self.reason = (
            f"{role.title()} DNNI v{version} is preserved byte-for-byte without parsing; "
            f"{role} inference and training are unavailable."
        )


class UnavailablePitchModel(UnavailableDNNIModel):
    """Role-specific message retained for compatibility with existing callers."""

    def __init__(self, version: int):
        super().__init__("pitch", version)
        self.reason = (
            f"Pitch training, Pitch-model initialization, and Pitch-network inference are unavailable "
            f"because DNNI v{version} is preserved opaquely. Choose a training configuration that "
            "does not include Pitch to keep this dependency unchanged."
        )


def load_voice(path, preserve_non_acoustic_models: bool = False) -> VoiceFiles:
    path = Path(path).expanduser()
    if is_info_voice(path):
        info = InfoVoice.load(path)
        # Preserve-mode deliberately avoids loading/validating the untouched
        # Duration and Pitch payloads, including opaque DNNI v3 dependencies.
        roles = ("acoustic", "vocoder") if preserve_non_acoustic_models else ROLES
        models = {role: _load_info_model(role, info.models[role]) for role in roles}
        metadata = {".name": info.info.get("name", ""), ".language": info.info.get("language", "")}
        return VoiceFiles(
            path, models, metadata, info=info,
            preserve_non_acoustic_models=preserve_non_acoustic_models,
        )
    if preserve_non_acoustic_models:
        raise ValueError("preserving non-acoustic models is available only for info.nofs shared-model voicebanks")
    nofs = NofsFile.load(path)
    keys = nofs.model_keys()
    models = {role: _load_model_bytes(nofs.read(keys[role])) for role in ROLES}
    return VoiceFiles(path, models, nofs.metadata(), nofs=nofs)

def model_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def write_nofs_voice(
    voice: VoiceFiles, output: Path, name: str | None = None,
    acoustic_dnni_version: int = 2,
) -> Path:
    keys = voice.nofs.model_keys()
    replacements = {
        keys[role]: serialize_model_for_output(voice.models[role], role, acoustic_dnni_version)
        for role in ROLES
    }
    if name is not None:
        replacements[b".name"] = name.encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(voice.nofs.replace(replacements))
    return output


def write_info_voice(
    voice: VoiceFiles, database_directory: Path, version: str, name: str | None = None,
    acoustic_dnni_version: int = 2,
) -> Path:
    info = voice.info
    models_root = database_directory.parent.parent / "models"
    document = info.to_json()
    if voice.files.preserve_non_acoustic_models:
        # Keep source embedding bytes/IDs intact for models that were not loaded.
        document["duration_style"] = info.info["duration_style"]
        document["f0_style"] = info.info["f0_style"]
    for role in ROLES:
        if voice.files.preserve_non_acoustic_models and role != "acoustic":
            # Copy the original opaque model bytes when the output database is
            # in a different Synthesizer V installation. Never parse, instantiate,
            # validate or reserialize this network in preservation mode.
            reference = info.info[INFO_KINDS[role]]
            target = models_root / INFO_KINDS[role] / f"{reference}.dnni"
            original_bytes = info.models[role].read_bytes()
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(original_bytes)
            elif target.read_bytes() != original_bytes:
                raise ValueError(
                    f"existing preserved model differs from the source model: {target}; "
                    "refusing to overwrite or reference different bytes"
                )
            document[INFO_KINDS[role]] = reference
            continue
        data = serialize_model_for_output(voice.models[role], role, acoustic_dnni_version)
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
        if files.preserve_non_acoustic_models:
            if info is None:
                raise ValueError("preserving non-acoustic models requires an info.nofs voicebank")
            acoustic_file = files.models["acoustic"]
            vocoder_file = files.models["vocoder"]
            # This loss renders predicted audio; the acoustic model and frozen
            # decoder still need executable network payloads. Other roles can
            # remain opaque and untouched in the source info voice.
            if isinstance(acoustic_file, OpaqueDnniFile):
                raise ValueError("acoustic training is unavailable because the source Acoustic DNNI v3 payload is opaque")
            if isinstance(vocoder_file, OpaqueDnniFile):
                raise ValueError("acoustic waveform training requires a runnable Vocoder; the source Vocoder DNNI v3 payload is opaque")
            self.acoustic = AcousticModel(
                acoustic_file, list(info.vocal_modes), info.acoustic_speaker,
                info.default_style, list(info.vocal_modes.values()),
            )
            self.vocoder = Vocoder(vocoder_file)
            return

        if info is None:
            duration_file = files.models["duration"]
            pitch_file = files.models["pitch"]
            acoustic_file = files.models["acoustic"]
            vocoder_file = files.models["vocoder"]
            self.duration = UnavailableDNNIModel("duration", duration_file.version) if isinstance(duration_file, OpaqueDnniFile) else DurationModel(duration_file)
            self.pitch = UnavailablePitchModel(pitch_file.version) if isinstance(pitch_file, OpaqueDnniFile) else PitchModel(pitch_file)
            self.acoustic = UnavailableDNNIModel("acoustic", acoustic_file.version) if isinstance(acoustic_file, OpaqueDnniFile) else AcousticModel(acoustic_file, files.vocal_modes)
            self.vocoder = UnavailableDNNIModel("vocoder", vocoder_file.version) if isinstance(vocoder_file, OpaqueDnniFile) else Vocoder(vocoder_file)
        else:
            duration_file = files.models["duration"]
            pitch_file = files.models["pitch"]
            acoustic_file = files.models["acoustic"]
            vocoder_file = files.models["vocoder"]
            self.duration = UnavailableDNNIModel("duration", duration_file.version) if isinstance(duration_file, OpaqueDnniFile) else DurationModel(duration_file, info.duration_voice)
            self.pitch = UnavailablePitchModel(pitch_file.version) if isinstance(pitch_file, OpaqueDnniFile) else PitchModel(pitch_file, info.pitch_speaker)
            self.acoustic = UnavailableDNNIModel("acoustic", acoustic_file.version) if isinstance(acoustic_file, OpaqueDnniFile) else AcousticModel(acoustic_file, list(info.vocal_modes), info.acoustic_speaker, info.default_style, list(info.vocal_modes.values()))
            self.vocoder = UnavailableDNNIModel("vocoder", vocoder_file.version) if isinstance(vocoder_file, OpaqueDnniFile) else Vocoder(vocoder_file)

    @classmethod
    def load(cls, path, preserve_non_acoustic_models: bool = False) -> "Voice":
        return cls(load_voice(path, preserve_non_acoustic_models=preserve_non_acoustic_models))

    @property
    def models(self) -> dict[str, torch.nn.Module]:
        return {role: getattr(self, role) for role in ROLES if hasattr(self, role)}

    def export(self) -> VoiceFiles:
        from .layers import export_parameters

        for role, model in self.models.items():
            if self.files.preserve_non_acoustic_models and role != "acoustic":
                continue
            if not getattr(model, "supported", True):
                continue
            export_parameters(model)
        info = self.files.info
        if info is not None:
            # Preserve mode leaves duration and Pitch embeddings in the source
            # descriptor; only acoustic embeddings are copied from the trainer.
            if hasattr(self, "duration") and getattr(self.duration, "supported", True):
                info.duration_voice = self.duration.voice.detach().cpu().numpy()
            if hasattr(self, "pitch") and getattr(self.pitch, "supported", True):
                info.pitch_speaker = self.pitch.speaker.detach().cpu().numpy()
            if hasattr(self, "acoustic") and getattr(self.acoustic, "supported", True):
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

    def save(
        self, output, name: str | None = None, version: str | None = None,
        acoustic_dnni_version: int = 2,
    ) -> Path:
        files = self.export()
        output = Path(output).expanduser()
        if files.info is None:
            return write_nofs_voice(files, output, name, acoustic_dnni_version)
        return write_info_voice(
            files, output, version or files.info.info.get("version", "100"), name,
            acoustic_dnni_version,
        )



def initialize_pitch_from_file(voice: Voice, source_path) -> int:
    """Initialize a voice's Pitch network from a strictly compatible DNNI model.

    The target voice's speaker vector is deliberately preserved. A source is
    accepted only when its model layout, phone tables, normalization buffers,
    non-parameter node payloads, and parameter shapes match.
    """
    model_path = Path(source_path).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"pretrained Pitch DNNI not found: {model_path}")
    if model_path.suffix.lower() != ".dnni":
        raise ValueError("pretrained Pitch model must be a .dnni file")

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


def initialize_pitch_model(voice: Voice, model_path: str | Path) -> Path:
    """Backward-compatible wrapper returning the resolved model path."""
    resolved = Path(model_path).expanduser().resolve()
    initialize_pitch_from_file(voice, resolved)
    return resolved
