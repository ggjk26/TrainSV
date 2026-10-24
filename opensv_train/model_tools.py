from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path

from .dnni import DnniFile, OpaqueDnniFile
from .nofs import InfoVoice, is_info_voice

ROLE_TO_INFO_KEY = {"duration": "duration", "pitch": "f0", "vocoder": "vocoder"}
ROLE_TO_MODEL_KIND = {"duration": "duration", "pitch": "f0", "vocoder": "vocoder"}
ALL_ROLES = ("duration", "pitch", "acoustic", "vocoder")
ROLE_TO_KIND = {"duration": "duration", "pitch": "f0", "acoustic": "acoustic", "vocoder": "vocoder"}
MODEL_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")


def _check_model_role(data: bytes, role: str) -> tuple[int, bool]:
    """Validate supported models by role; recognize v3 only as opaque bytes."""
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
        raise ValueError(f"the selected DNNI is not a supported {role} model, or its structure is incompatible: {error}") from error
    return version, True


def _copy_model_if_safe(target: Path, data: bytes) -> None:
    if target.exists():
        if target.read_bytes() != data:
            raise FileExistsError(
                f"model destination already exists with different bytes: {target}; "
                "nothing was overwritten"
            )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def _read_source_info(source_info_path: Path, models_root: str | Path | None) -> tuple[dict, dict[str, Path], Path]:
    if source_info_path.suffix.lower() != ".nofs" or not source_info_path.is_file():
        raise FileNotFoundError(f"source info.nofs was not found: {source_info_path}")
    if not is_info_voice(source_info_path):
        raise ValueError("the source must be an info.nofs JSON voicebank, not a regular binary voice.nofs")
    try:
        document = json.loads(source_info_path.read_text("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"source info.nofs is not valid UTF-8 JSON: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("source info.nofs must contain a JSON object")

    root = (
        Path(models_root).expanduser().resolve()
        if models_root
        else (source_info_path.parent.parent.parent / "models").resolve()
    )
    if not root.is_dir():
        raise FileNotFoundError(
            f"models root does not exist: {root}. Select the Synthesizer V models folder "
            "containing duration, f0, acoustic, and vocoder subfolders."
        )

    source_models: dict[str, Path] = {}
    missing: list[str] = []
    for role in ALL_ROLES:
        key = ROLE_TO_KIND[role]
        model_id = document.get(key)
        if not isinstance(model_id, str) or not MODEL_ID_RE.fullmatch(model_id):
            raise ValueError(f"source info.nofs field {key!r} must be a 32-character hexadecimal model ID")
        model_path = root / key / f"{model_id}.dnni"
        if not model_path.is_file():
            missing.append(f"{role} ({key}): {model_path}")
        source_models[role] = model_path
    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            "could not resolve all source DNNI files under the selected models root. "
            "Choose the Synthesizer V models directory containing the referenced files:\n" + details
        )

    for field in ("duration_style", "f0_style", "styles"):
        if field not in document:
            raise ValueError(f"source info.nofs is missing required field {field!r}")
    if not isinstance(document["styles"], list):
        raise ValueError("source info.nofs field 'styles' must be a list")
    return document, source_models, root


def replace_info_models(
    source_info_path: str | Path,
    replacements: dict[str, str | Path],
    output_info_path: str | Path,
    models_root: str | Path | None = None,
) -> dict:
    """Create a new info.nofs with one or more non-acoustic model dependencies changed.

    models_root is the directory containing duration, f0, acoustic, and vocoder.
    If omitted, the conventional root derived from the source info.nofs path is
    used. The source voicebank is never edited, and the output descriptor is
    written only after source and replacement model files have been validated.
    """
    if not replacements:
        raise ValueError("select at least one model to replace: duration, pitch/F0, and/or vocoder")
    if unknown := set(replacements) - set(ROLE_TO_INFO_KEY):
        raise ValueError(
            "only duration, pitch, and vocoder can be replaced; Acoustic is intentionally excluded "
            f"(unsupported roles: {', '.join(sorted(unknown))})"
        )

    source_info_path = Path(source_info_path).expanduser().resolve()
    output_info_path = Path(output_info_path).expanduser().resolve()
    if source_info_path == output_info_path:
        raise ValueError("output info.nofs must be a new path; the original voicebank will not be overwritten")
    if output_info_path.suffix.lower() != ".nofs":
        raise ValueError("output path must end with .nofs")
    if output_info_path.exists():
        raise FileExistsError(f"output already exists; choose a new path: {output_info_path}")

    source_document, source_models, resolved_source_root = _read_source_info(source_info_path, models_root)
    replacement_data: dict[str, bytes] = {}
    validation: dict[str, tuple[int, bool]] = {}
    for role, raw_path in replacements.items():
        path = Path(raw_path).expanduser().resolve()
        if path.suffix.lower() != ".dnni" or not path.is_file():
            raise FileNotFoundError(f"replacement {role} model must be an existing .dnni file: {path}")
        data = path.read_bytes()
        try:
            validation[role] = _check_model_role(data, role)
        except (ValueError, OSError) as error:
            raise ValueError(f"replacement {role} DNNI '{path}' failed validation: {error}") from error
        replacement_data[role] = data

    output_models_root = output_info_path.parent.parent.parent / "models"
    output_document = dict(source_document)
    result_models: dict[str, tuple[Path, bytes]] = {}
    role_results: dict[str, dict] = {}
    for role in ALL_ROLES:
        info_key = ROLE_TO_KIND[role]
        if role in replacements:
            data = replacement_data[role]
            new_id = hashlib.md5(data).hexdigest()
            result_models[role] = (output_models_root / info_key / f"{new_id}.dnni", data)
            output_document[info_key] = new_id
            role_results[role] = {
                "old_model_id": source_document[info_key],
                "new_model_id": new_id,
                "dnni_version": validation[role][0],
                "role_checked": validation[role][1],
                "warning": "" if validation[role][1] else (
                    "DNNI v3 was copied opaquely. Its internal model role, architecture, "
                    "phoneme tables and runtime compatibility could not be verified."
                ),
            }
        else:
            original_id = source_document[info_key]
            data = source_models[role].read_bytes()
            result_models[role] = (output_models_root / info_key / f"{original_id}.dnni", data)

    # Preflight all destinations to avoid an overwrite or an avoidable partial copy.
    for role, (target, data) in result_models.items():
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(
                f"model destination for {role} already exists with different bytes: {target}; "
                "nothing was overwritten"
            )
    for role, (target, data) in result_models.items():
        _copy_model_if_safe(target, data)

    if output_info_path.name.lower().startswith("info.") and output_info_path.name.lower().endswith(".nofs"):
        output_document["version"] = output_info_path.name[5:-5]
    output_info_path.parent.mkdir(parents=True, exist_ok=True)
    output_info_path.write_text(
        json.dumps(output_document, ensure_ascii=False, indent=2, separators=(",", " : ")) + "\n",
        encoding="utf-8",
    )

    try:
        generated = InfoVoice.load(output_info_path)
        for role in ALL_ROLES:
            expected_path, expected_data = result_models[role]
            if generated.models[role] != expected_path or generated.models[role].read_bytes() != expected_data:
                raise RuntimeError(f"generated info.nofs failed the {role} model-reference round-trip check")
    except Exception:
        output_info_path.unlink(missing_ok=True)
        raise

    warning_messages = [f"{role}: {item['warning']}" for role, item in role_results.items() if item["warning"]]
    return {
        "source": str(source_info_path),
        "output": str(output_info_path),
        "models_root": str(resolved_source_root),
        "replaced": role_results,
        "warning": "\n".join(warning_messages),
    }


def replace_info_model(
    source_info_path: str | Path,
    role: str,
    replacement_model_path: str | Path,
    output_info_path: str | Path,
    models_root: str | Path | None = None,
) -> dict:
    """Backwards-compatible single-model wrapper for replace_info_models."""
    if role not in ROLE_TO_INFO_KEY:
        raise ValueError("role must be one of: duration, pitch, vocoder; Acoustic is intentionally excluded")
    result = replace_info_models(
        source_info_path, {role: replacement_model_path}, output_info_path, models_root=models_root
    )
    item = result["replaced"][role]
    return {**result, "role": role, **item}
