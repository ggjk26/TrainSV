"""Build experimental SFPK packages that reference models in a local SynthV models tree.

The inspected SFPK sample is a ZIP archive containing config.json and info.nofs.
For a regular binary voice.nofs, this module exports the four DNNI model payloads
to a caller-selected local models root, derives the Flat-compatible embedding
vectors from the source voice, and writes an info.nofs JSON referencing those
model IDs. The SFPK itself contains only config.json and info.nofs, matching the
sample layout; it is not a portable package of all model dependencies.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

INFO_MODEL_KINDS = {
    "duration": "duration",
    "pitch": "f0",
    "acoustic": "acoustic",
    "vocoder": "vocoder",
}
INFO_MODEL_FIELDS = ("acoustic", "vocoder", "duration", "f0")
_TRANSLATION_LOCALES = (
    "English",
    "Deutsch",
    "Español",
    "Français",
    "日本語",
    "한국어",
    "Português",
    "Русский",
    "Tiếng Việt",
    "简体中文",
    "繁體中文",
)


def _is_info_json(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(1) == b"{"


def _write_model_if_missing(target: Path, data: bytes) -> None:
    """Write a content-addressed DNNI without overwriting existing model data."""
    if target.is_file():
        if target.read_bytes() != data:
            raise ValueError(
                f"model path already exists with different content: {target}; "
                "refusing to overwrite a shared model"
            )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        temp.write_bytes(data)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def _tensor_vector_hex(value, label: str, *, expected_values: int = 32) -> str:
    from .nofs import vector_hex

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    flattened = value.reshape(-1)
    if flattened.size != expected_values:
        raise ValueError(
            f"{label} has {flattened.size} values; the observed Flat format expects "
            f"{expected_values} float32 values"
        )
    encoded = vector_hex(flattened)
    expected_hex_chars = expected_values * 8
    if len(encoded) != expected_hex_chars:
        raise ValueError(
            f"{label} serialized to {len(encoded)} hex characters; expected {expected_hex_chars}"
        )
    return encoded


def _build_info_from_binary_voice(
    source: Path,
    models_root: Path,
    *,
    name: str | None,
    vendor: str | None,
    version: str | None,
) -> dict:
    # Import lazily: inspecting or packaging an existing info.nofs does not
    # need to import the full PyTorch training stack.
    from .voice import INFO_KINDS, ROLES, Voice

    source_voice = Voice.load(source)
    if source_voice.files.info is not None or source_voice.files.nofs is None:
        raise ValueError("source is not a regular binary voice.nofs file")

    metadata = source_voice.files.metadata
    resolved_name = (name or metadata.get(".name") or source.stem).strip()
    resolved_vendor = (vendor or metadata.get(".vendor") or "TrainSV").strip()
    resolved_version = (version or metadata.get(".version") or "101").strip()
    language = (metadata.get(".language") or "japanese").strip()
    phoneset = (metadata.get(".phoneset") or "romaji").strip()
    support_languages = (metadata.get(".multi") or language).split()

    if not resolved_name:
        raise ValueError("voicebank name must not be empty")
    if not resolved_vendor:
        raise ValueError("vendor name must not be empty")
    if not resolved_version:
        raise ValueError("version must not be empty")

    model_ids: dict[str, str] = {}
    for role in ROLES:
        model_data = source_voice.files.models[role].serialize()
        model_id = hashlib.md5(model_data).hexdigest()
        model_kind = INFO_KINDS[role]
        _write_model_if_missing(models_root / model_kind / f"{model_id}.dnni", model_data)
        model_ids[model_kind] = model_id

    acoustic = source_voice.acoustic
    styles = [
        {"name": "(base)", "data": _tensor_vector_hex(acoustic.speaker_vector(), "acoustic base style")},
        {"name": "(default)", "data": _tensor_vector_hex(acoustic.default_style_vector(), "acoustic default style")},
    ]
    mode_names = list(acoustic.vocal_modes)
    mode_vectors = acoustic.mode_style_vectors()
    if len(mode_names) != len(mode_vectors):
        raise ValueError("vocal mode names do not match the acoustic style vectors")
    for mode_name, mode_vector in zip(mode_names, mode_vectors):
        if mode_name in {"(base)", "(default)"}:
            raise ValueError(f"vocal mode name is reserved: {mode_name}")
        styles.append({"name": mode_name, "data": _tensor_vector_hex(mode_vector, f"vocal mode {mode_name}")})

    return {
        "name": resolved_name,
        "version": resolved_version,
        "vendor": resolved_vendor,
        "language": language,
        "phoneset": phoneset,
        "support_languages": support_languages,
        "acoustic": model_ids["acoustic"],
        "vocoder": model_ids["vocoder"],
        "duration": model_ids["duration"],
        "f0": model_ids["f0"],
        "styles": styles,
        "duration_style": _tensor_vector_hex(source_voice.duration.voice_vector(), "duration style", expected_values=128),
        "f0_style": _tensor_vector_hex(source_voice.pitch.speaker(), "F0 style"),
    }


def _validate_info_document(info: dict, models_root: Path) -> dict:
    for key in ("name", "version", "vendor", "language", "phoneset"):
        if not isinstance(info.get(key), str) or not info[key].strip():
            raise ValueError(f"info.nofs has an empty or missing {key!r} field")
    if not isinstance(info.get("support_languages"), list) or not all(
        isinstance(language, str) and language.strip() for language in info["support_languages"]
    ):
        raise ValueError("info.nofs support_languages must be a list of non-empty strings")

    normalized = dict(info)
    for field in INFO_MODEL_FIELDS:
        value = info.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{32}", value) is None:
            raise ValueError(f"info.nofs field {field!r} must be a 32-character model ID")
        model_id = value.lower()
        normalized[field] = model_id
        model_path = models_root / field / f"{model_id}.dnni"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"model dependency is missing: {model_path}. "
                "Choose the correct Synthesizer V models directory or install the referenced model first."
            )

    embedding_lengths = {"duration_style": 128 * 8, "f0_style": 32 * 8}
    for field, expected_hex_chars in embedding_lengths.items():
        value = info.get(field)
        if not isinstance(value, str) or re.fullmatch(rf"[0-9a-fA-F]{{{expected_hex_chars}}}", value) is None:
            expected_values = expected_hex_chars // 8
            raise ValueError(
                f"info.nofs field {field!r} must be a {expected_hex_chars}-character "
                f"hex string ({expected_values} float32 values)"
            )

    styles = info.get("styles")
    if not isinstance(styles, list) or not styles:
        raise ValueError("info.nofs styles must be a non-empty list")
    seen_names = set()
    for index, style in enumerate(styles):
        if not isinstance(style, dict):
            raise ValueError(f"info.nofs styles[{index}] must be an object")
        style_name = style.get("name")
        data = style.get("data")
        if not isinstance(style_name, str) or not style_name.strip():
            raise ValueError(f"info.nofs styles[{index}] has an empty name")
        if style_name in seen_names:
            raise ValueError(f"info.nofs contains duplicate style name: {style_name}")
        seen_names.add(style_name)
        if not isinstance(data, str) or re.fullmatch(r"[0-9a-fA-F]{256}", data) is None:
            raise ValueError(f"info.nofs style {style_name!r} must have a 256-character float32 embedding")
    if "(base)" not in seen_names or "(default)" not in seen_names:
        raise ValueError("info.nofs must contain both (base) and (default) styles")
    return normalized


def _config_for(info: dict) -> dict:
    name = info["name"]
    translations = {locale: "" for locale in _TRANSLATION_LOCALES}
    translations["English"] = f'"{name}"'
    translations["简体中文"] = f'"{name}"'
    translations["繁體中文"] = f'"{name}"'
    # These names/fields follow the inspected Flat SFPK sample. This sample
    # carries no separate PNG members and declares no bundled model assets.
    return {
        "name": name,
        "version": info["version"],
        "profile": True,
        "background": True,
        "models": [],
        "model_types": [],
        "phonesets": [],
        "translations": json.dumps(translations, ensure_ascii=False, separators=(",", ":")),
    }


def _optional_png(path: str | Path | None, label: str) -> Path | None:
    """Resolve an optional PNG asset and reject non-PNG content."""
    if path is None or not str(path).strip():
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} file not found: {resolved}")
    with resolved.open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"{label} must be a PNG file: {resolved}")
    return resolved


def build_sfpk(
    *,
    voice: str | Path,
    models_root: str | Path,
    output: str | Path,
    name: str | None = None,
    vendor: str | None = None,
    version: str | None = None,
    profile: str | Path | None = None,
    background: str | Path | None = None,
) -> Path:
    """Build a Flat SFPK with local DNNI dependencies and optional PNG artwork.

    For a binary voice.nofs, DNNI payloads are split out to the selected model
    directory and an info.nofs JSON is derived from its speaker/style vectors.
    For an existing info.nofs, that metadata is reused after validating its
    referenced DNNI files. Selected profile.png and background.png artwork is
    placed at the SFPK archive root. Model binaries remain in the local models
    tree and are not bundled into the SFPK.
    """
    source = Path(voice).expanduser()
    models_root = Path(models_root).expanduser().resolve()
    output = Path(output).expanduser()
    profile_path = _optional_png(profile, "profile.png")
    background_path = _optional_png(background, "background.png")

    if not source.is_file():
        raise FileNotFoundError(f"voice file not found: {source}")
    if not models_root.is_dir():
        raise NotADirectoryError(f"models root must be an existing directory: {models_root}")
    if output.suffix.lower() != ".sfpk":
        raise ValueError("output filename must end in .sfpk")

    if _is_info_json(source):
        try:
            info = json.loads(source.read_text("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{source} is not valid info.nofs JSON: {error}") from error
        info = dict(info)
        if name is not None:
            info["name"] = name.strip()
        if vendor is not None:
            info["vendor"] = vendor.strip()
        if version is not None:
            info["version"] = version.strip()
    else:
        info = _build_info_from_binary_voice(
            source, models_root, name=name, vendor=vendor, version=version
        )

    info = _validate_info_document(info, models_root)
    config = _config_for(info)
    config["version"] = info["version"]
    config_bytes = (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    info_bytes = (json.dumps(info, ensure_ascii=False, indent=2, separators=(",", " : ")) + "\n").encode("utf-8")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_name(output.name + ".tmp")
    try:
        with zipfile.ZipFile(
            temp_output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            archive.writestr("config.json", config_bytes)
            archive.writestr("info.nofs", info_bytes)
            if profile_path is not None:
                archive.write(profile_path, arcname="profile.png")
            if background_path is not None:
                archive.write(background_path, arcname="background.png")

        expected_members = ["config.json", "info.nofs"]
        if profile_path is not None:
            expected_members.append("profile.png")
        if background_path is not None:
            expected_members.append("background.png")
        with zipfile.ZipFile(temp_output) as archive:
            if archive.namelist() != expected_members:
                raise ValueError(f"unexpected SFPK members: {archive.namelist()}")
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"corrupt SFPK member: {bad_member}")
            checked_config = json.loads(archive.read("config.json"))
            checked_info = json.loads(archive.read("info.nofs"))
            if checked_config.get("name") != checked_info.get("name"):
                raise ValueError("config.json and info.nofs names do not match")
            if checked_config.get("version") != checked_info.get("version"):
                raise ValueError("config.json and info.nofs versions do not match")
            _validate_info_document(checked_info, models_root)
        os.replace(temp_output, output)
    finally:
        temp_output.unlink(missing_ok=True)

    return output
