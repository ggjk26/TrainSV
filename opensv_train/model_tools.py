from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from .dnni import DnniFile, OpaqueDnniFile
from .nofs import InfoVoice, is_info_voice

ROLE_TO_INFO_KEY = {
    "duration": "duration",
    "pitch": "f0",
    "vocoder": "vocoder",
}
ROLE_TO_MODEL_KIND = {
    "duration": "duration",
    "pitch": "f0",
    "vocoder": "vocoder",
}
ALL_ROLES = ("duration", "pitch", "acoustic", "vocoder")


def _check_model_role(data: bytes, role: str) -> tuple[int, bool]:
    """Validate a supported model's role; recognize v3 only as opaque bytes.

    DNNI v3 is intentionally not parsed or executed. A valid v3 header is enough
    for lossless pass-through, not proof that the model is compatible with a
    voicebank or runnable in TrainSV/Synthesizer V.
    """
    if len(data) < 8:
        raise ValueError("DNNI file is too short to contain a header")
    magic, version = struct.unpack_from("<II", data, 0)
    if magic != 0x7FCA00FF:
        raise ValueError("the selected file is not a DNNI container")
    if version == 3:
        OpaqueDnniFile.from_bytes(data)
        return version, False
    if version not in (1, 2):
        raise ValueError(f"unsupported DNNI version {version}; only v1/v2 role validation and v3 opaque pass-through are supported")

    try:
        parsed = DnniFile.parse(data)
        # The role parsers validate model-specific top-level layout and the
        # nested network structure. A v1 tree is converted in-memory only so
        # the parsers can validate its hashed type IDs; original bytes are saved.
        checked = parsed.as_version(2) if version == 1 else parsed
        if role == "duration":
            from .duration import DurationModel
            model = DurationModel(checked)
        elif role == "pitch":
            from .pitch import PitchModel
            model = PitchModel(checked)
        elif role == "vocoder":
            from .vocoder import Vocoder
            model = Vocoder(checked)
        else:
            raise ValueError(f"Acoustic models cannot be replaced by this tool (role={role!r})")
        del model
    except (ValueError, KeyError, IndexError, struct.error) as error:
        raise ValueError(
            f"the selected DNNI is not a supported {role} model, or its structure is incompatible: {error}"
        ) from error
    return version, True


def _copy_model_if_safe(source: Path, target: Path) -> None:
    data = source.read_bytes()
    if target.exists():
        if target.read_bytes() != data:
            raise FileExistsError(
                f"model destination already exists with different bytes: {target}; "
                "nothing was overwritten"
            )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def replace_info_model(
    source_info_path: str | Path,
    role: str,
    replacement_model_path: str | Path,
    output_info_path: str | Path,
) -> dict:
    """Create a new info.nofs variant with one non-acoustic DNNI dependency changed.

    Source voicebank files are never modified. The output info.nofs must follow
    the same directory-depth convention as installed info voices because its
    models root is resolved as output.parent.parent.parent / "models".
    """
    if role not in ROLE_TO_INFO_KEY:
        raise ValueError("role must be one of: duration, pitch, vocoder; Acoustic is intentionally excluded")

    source_info_path = Path(source_info_path).expanduser().resolve()
    replacement_model_path = Path(replacement_model_path).expanduser().resolve()
    output_info_path = Path(output_info_path).expanduser().resolve()

    if source_info_path == output_info_path:
        raise ValueError("output info.nofs must be a new path; the original voicebank will not be overwritten")
    if source_info_path.suffix.lower() != ".nofs" or not source_info_path.is_file():
        raise FileNotFoundError(f"source info.nofs was not found: {source_info_path}")
    if not is_info_voice(source_info_path):
        raise ValueError("the source must be an info.nofs JSON voicebank, not a regular binary voice.nofs")
    if output_info_path.suffix.lower() != ".nofs":
        raise ValueError("output path must end with .nofs")
    if output_info_path.exists():
        raise FileExistsError(f"output already exists; choose a new path: {output_info_path}")
    if replacement_model_path.suffix.lower() != ".dnni" or not replacement_model_path.is_file():
        raise FileNotFoundError(f"replacement model must be an existing .dnni file: {replacement_model_path}")

    # Validate the complete source descriptor and all of its existing model links.
    source_voice = InfoVoice.load(source_info_path)
    source_document = json.loads(source_info_path.read_text("utf-8"))
    replacement_bytes = replacement_model_path.read_bytes()
    model_version, role_checked = _check_model_role(replacement_bytes, role)

    source_models_root = source_info_path.parent.parent.parent / "models"
    output_models_root = output_info_path.parent.parent.parent / "models"
    info_key = ROLE_TO_INFO_KEY[role]
    model_kind = ROLE_TO_MODEL_KIND[role]
    replacement_id = hashlib.md5(replacement_bytes).hexdigest()
    replacement_target = output_models_root / model_kind / f"{replacement_id}.dnni"

    # Make the output self-consistent even when placed under a separate voicebank
    # root: copy unchanged dependencies as raw bytes, refusing ID collisions.
    for existing_role in ALL_ROLES:
        if existing_role == role:
            continue
        existing_kind = ROLE_TO_MODEL_KIND.get(existing_role, "acoustic")
        source_model = source_voice.models[existing_role]
        original_id = source_document[existing_kind]
        target_model = output_models_root / existing_kind / f"{original_id}.dnni"
        _copy_model_if_safe(source_model, target_model)

    _copy_model_if_safe(replacement_model_path, replacement_target)
    output_document = dict(source_document)
    output_document[info_key] = replacement_id
    if output_info_path.name.lower().startswith("info.") and output_info_path.name.lower().endswith(".nofs"):
        output_document["version"] = output_info_path.name[5:-5]
    output_info_path.parent.mkdir(parents=True, exist_ok=True)
    output_info_path.write_text(
        json.dumps(output_document, ensure_ascii=False, indent=2, separators=(",", " : ")) + "\n",
        encoding="utf-8",
    )

    # Final referential check: reload the generated descriptor and ensure every
    # role still resolves. This does not claim inference compatibility for v3.
    generated = InfoVoice.load(output_info_path)
    expected_target = output_models_root / model_kind / f"{replacement_id}.dnni"
    if generated.models[role] != expected_target or generated.models[role].read_bytes() != replacement_bytes:
        output_info_path.unlink(missing_ok=True)
        raise RuntimeError("generated info.nofs failed its DNNI-reference round-trip check")

    return {
        "source": str(source_info_path),
        "output": str(output_info_path),
        "role": role,
        "old_model_id": source_document[info_key],
        "new_model_id": replacement_id,
        "dnni_version": model_version,
        "role_checked": role_checked,
        "warning": "" if role_checked else (
            "DNNI v3 was copied opaquely. Its internal model role, architecture, "
            "phoneme tables and runtime compatibility could not be verified."
        ),
    }
