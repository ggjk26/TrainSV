"""Read-only diagnostics for Synthesizer V Flat package candidates.

This module deliberately does not assume an .sfpk manifest or directory layout.
It inspects ZIP containers and extracted directories without extracting archives.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

MAX_ENTRIES = 5000
MAX_CONFIG_BYTES = 2 * 1024 * 1024
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MODEL_SUFFIXES = {".nofs", ".dnni", ".svpk", ".sfpk"}


def _kind(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in MODEL_SUFFIXES:
        return "model-or-package"
    if suffix in {".json", ".toml", ".yaml", ".yml", ".xml", ".ini", ".cfg"}:
        return "configuration-candidate"
    return "other"


def _json_summary(name: str, data: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {"path": name, "parsed": False}
    if len(data) > MAX_CONFIG_BYTES:
        result["note"] = f"Skipped: larger than {MAX_CONFIG_BYTES} bytes"
        return result
    try:
        value = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        result["note"] = "Not valid UTF-8 JSON"
        return result
    result["parsed"] = True
    if isinstance(value, dict):
        result["top_level_keys"] = list(value.keys())[:200]
        # Show only short scalar values useful for identifying a package schema.
        scalars = {}
        for key, item in value.items():
            if isinstance(item, (str, int, float, bool)) or item is None:
                text = str(item)
                scalars[str(key)] = text[:240]
                if len(scalars) >= 80:
                    break
        result["scalar_fields"] = scalars
    elif isinstance(value, list):
        result["json_type"] = "array"
        result["item_count"] = len(value)
        if value and isinstance(value[0], dict):
            result["first_item_keys"] = list(value[0].keys())[:100]
    else:
        result["json_type"] = type(value).__name__
    return result


def inspect_sfpk(path: str | Path, max_entries: int = MAX_ENTRIES) -> dict[str, Any]:
    """Summarize an .sfpk file or extracted directory without modifying it."""
    target = Path(path).expanduser()
    report: dict[str, Any] = {
        "input": str(target),
        "exists": target.exists(),
        "container_type": "unknown",
        "entry_count": 0,
        "truncated": False,
        "entries": [],
        "configuration_candidates": [],
        "warnings": [],
        "packaging_supported": False,
        "packaging_note": (
            "Read-only inspection only. A package writer must be based on a real "
            ".sfpk sample and verified by opening the result in Synthesizer V Flat."
        ),
    }
    if not target.exists():
        report["warnings"].append("Path does not exist.")
        return report

    entries: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    if target.is_dir():
        report["container_type"] = "extracted-directory"
        for item in sorted(target.rglob("*"), key=lambda p: str(p).lower()):
            if len(entries) >= max_entries:
                report["truncated"] = True
                break
            try:
                if not item.is_file():
                    continue
                rel = item.relative_to(target).as_posix()
                size = item.stat().st_size
                entries.append({"path": rel, "size": size, "kind": _kind(rel)})
                if item.suffix.lower() == ".json":
                    try:
                        configs.append(_json_summary(rel, item.read_bytes()))
                    except OSError as error:
                        configs.append({"path": rel, "parsed": False, "note": str(error)})
            except OSError as error:
                report["warnings"].append(f"Could not inspect {item}: {error}")
    elif target.is_file() and zipfile.is_zipfile(target):
        report["container_type"] = "zip-container"
        try:
            with zipfile.ZipFile(target) as archive:
                infos = archive.infolist()
                report["archive_comment"] = archive.comment.decode("utf-8", "replace")[:500]
                report["truncated"] = len(infos) > max_entries
                for info in infos[:max_entries]:
                    name = info.filename
                    # Never extract; flag path names that would escape an extraction root.
                    posix = PurePosixPath(name)
                    if posix.is_absolute() or ".." in posix.parts:
                        report["warnings"].append(f"Unsafe archive path detected: {name}")
                    if info.is_dir():
                        continue
                    entries.append({"path": name, "size": info.file_size, "compressed_size": info.compress_size,
                                    "kind": _kind(name)})
                    if Path(name).suffix.lower() == ".json" and info.file_size <= MAX_CONFIG_BYTES:
                        try:
                            configs.append(_json_summary(name, archive.read(info)))
                        except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                            configs.append({"path": name, "parsed": False, "note": str(error)})
        except (OSError, zipfile.BadZipFile) as error:
            report["warnings"].append(f"Archive could not be read: {error}")
    else:
        report["container_type"] = "single-file"
        report["warnings"].append("Not a recognized ZIP container or extracted directory; no package format was inferred.")
        try:
            report["entries"].append({"path": target.name, "size": target.stat().st_size, "kind": _kind(target.name)})
        except OSError:
            pass

    report["entries"] = entries or report["entries"]
    report["entry_count"] = len(report["entries"])
    report["configuration_candidates"] = configs
    report["model_candidates"] = [entry for entry in report["entries"] if entry["kind"] == "model-or-package"]
    report["image_candidates"] = [entry for entry in report["entries"] if entry["kind"] == "image"]
    if report["container_type"] == "zip-container":
        report["warnings"].append(
            "ZIP readability does not prove this is a valid SFPK. The internal schema and required files remain unverified."
        )
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"Input: {report['input']}",
        f"Container: {report['container_type']}",
        f"Entries inspected: {report['entry_count']}" + (" (truncated)" if report.get("truncated") else ""),
        f"Model/package candidates: {len(report.get('model_candidates', []))}",
        f"Image candidates: {len(report.get('image_candidates', []))}",
        f"JSON configuration candidates: {len(report.get('configuration_candidates', []))}",
    ]
    for item in report.get("entries", []):
        lines.append(f"  {item['kind']:24s} {item['size']:>10,}  {item['path']}")
    for config in report.get("configuration_candidates", []):
        lines.append(f"JSON {config['path']}: " + (", ".join(config.get("top_level_keys", [])) if config.get("parsed") else config.get("note", "not parsed")))
    for warning in report.get("warnings", []):
        lines.append("WARNING: " + warning)
    lines.append(report["packaging_note"])
    return "\n".join(lines)
