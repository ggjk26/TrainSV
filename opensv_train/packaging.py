"""Experimental package writers based on the user-provided package layouts.

The writers intentionally use explicit inputs and preserve the supplied SFPK
config rather than inventing model records or rewriting info.nofs internals.
"""
from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

SVPK_PROFILE_NAME = "package-profile"
SVPK_REQUIRED_PROFILE_FIELDS = (
    "productName", "vendorName", "versionString",
    "versionNumber", "dependingVersionNumber",
)
SFPK_CONFIG_FIELDS = ("name", "version", "profile", "background", "models", "model_types", "phonesets", "translations")


def _existing_file(value: str | Path | None, label: str, *, required: bool = False) -> Path | None:
    if value is None or str(value).strip() == "":
        if required:
            raise ValueError(f"{label} is required")
        return None
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def _archive(output: str | Path, entries: Iterable[tuple[str, Path | bytes]]) -> Path:
    destination = Path(output).expanduser()
    if destination.suffix.lower() not in {".svpk", ".sfpk"}:
        raise ValueError(f"output extension must match a supported package type: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    names: set[str] = set()
    normalized: list[tuple[str, Path | bytes]] = []
    for archive_name, source in entries:
        if not archive_name or archive_name.startswith(("/", "\\")) or ".." in Path(archive_name).parts:
            raise ValueError(f"unsafe archive path: {archive_name!r}")
        if archive_name in names:
            raise ValueError(f"duplicate archive entry: {archive_name}")
        names.add(archive_name)
        normalized.append((archive_name, source))
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for archive_name, source in normalized:
                if isinstance(source, Path):
                    archive.write(source, archive_name)
                else:
                    archive.writestr(archive_name, source)
        with zipfile.ZipFile(temporary, "r") as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"generated archive failed CRC check at {bad}")
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return destination


def package_svpk(
    *,
    voice: str | Path,
    output: str | Path,
    product_name: str,
    vendor_name: str,
    version_string: str,
    version_number: int,
    depending_version_number: int = 0,
    profile: str | Path | None = None,
    background: str | Path | None = None,
    license_file: str | Path | None = None,
) -> Path:
    """Create a ZIP-based SVPK using the supplied sample's root entry names."""
    model = _existing_file(voice, "voice.nofs", required=True)
    avatar = _existing_file(profile, "profile image")
    backdrop = _existing_file(background, "background image")
    license_path = _existing_file(license_file, "license file")
    for label, image in (("profile image", avatar), ("background image", backdrop)):
        if image is not None and image.suffix.lower() != ".png":
            raise ValueError(f"{label} must be PNG to use the sample package filename: {image}")
    if not product_name.strip() or not vendor_name.strip() or not version_string.strip():
        raise ValueError("product name, vendor name, and version string must not be blank")
    if version_number < 0 or depending_version_number < 0:
        raise ValueError("version numbers must be non-negative")
    entries: list[tuple[str, Path | bytes]] = [("voice.nofs", model)]
    if backdrop:
        entries.append(("background.png", backdrop))
    if license_path:
        entries.append(("license.txt", license_path))
    if avatar:
        entries.append(("profile.png", avatar))
    metadata = {
        "productName": product_name,
        "vendorName": vendor_name,
        "versionString": version_string,
        "versionNumber": int(version_number),
        "dependingVersionNumber": int(depending_version_number),
    }
    entries.append((SVPK_PROFILE_NAME, json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")))
    destination = Path(output).expanduser()
    if destination.suffix.lower() != ".svpk":
        raise ValueError("SVPK output path must end in .svpk")
    return _archive(destination, entries)


def package_sfpk(
    *,
    info: str | Path,
    config: str | Path,
    output: str | Path,
    profile: str | Path | None = None,
    background: str | Path | None = None,
) -> Path:
    """Package explicitly supplied info.nofs and config.json without rewriting either."""
    info_path = _existing_file(info, "info.nofs", required=True)
    config_path = _existing_file(config, "config.json", required=True)
    avatar = _existing_file(profile, "profile image")
    backdrop = _existing_file(background, "background image")
    try:
        config_data = json.loads(config_path.read_text("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"config file is not valid UTF-8 JSON: {error}") from error
    if not isinstance(config_data, dict):
        raise ValueError("config.json must contain a JSON object")
    missing = [field for field in SFPK_CONFIG_FIELDS if field not in config_data]
    if missing:
        raise ValueError("config.json is missing fields from the supplied SFPK schema: " + ", ".join(missing))
    if not isinstance(config_data["models"], list) or not isinstance(config_data["model_types"], list) or not isinstance(config_data["phonesets"], list):
        raise ValueError("config.json fields models, model_types, and phonesets must be arrays")
    if not isinstance(config_data["profile"], bool) or not isinstance(config_data["background"], bool):
        raise ValueError("config.json fields profile and background must be booleans")
    if bool(avatar) != config_data["profile"]:
        raise ValueError("config.json profile flag must match whether --profile was supplied")
    if bool(backdrop) != config_data["background"]:
        raise ValueError("config.json background flag must match whether --background was supplied")
    entries: list[tuple[str, Path | bytes]] = [
        ("info.nofs", info_path),
        ("config.json", config_path),
    ]
    if avatar:
        entries.append(("profile.png" if avatar.suffix.lower() == ".png" else avatar.name, avatar))
    if backdrop:
        entries.append(("background.png" if backdrop.suffix.lower() == ".png" else backdrop.name, backdrop))
    destination = Path(output).expanduser()
    if destination.suffix.lower() != ".sfpk":
        raise ValueError("SFPK output path must end in .sfpk")
    return _archive(destination, entries)
