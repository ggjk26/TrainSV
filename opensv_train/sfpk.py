"""Read-only diagnostics for Synthesizer V Flat package candidates.

This module deliberately does not assume an .sfpk manifest or directory layout.
It inspects ZIP containers and extracted directories without extracting archives.
"""
from __future__ import annotations

import json
import re
import struct
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from .dnni import DnniFile, FILE_MAGIC as DNNI_MAGIC

MAX_ENTRIES = 5000
MAX_CONFIG_BYTES = 2 * 1024 * 1024
MAX_DNNI_PARSE_BYTES = 128 * 1024 * 1024
MAX_DNNI_INVENTORY_FILES = 2000
MAX_DNNI_SUMMARIES_PER_ROLE = 25
MODEL_ROLES = ("acoustic", "vocoder", "duration", "f0")
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




def _summarize_dnni(path: Path, root: Path) -> dict[str, Any]:
    """Return a bounded structural signature without loading tensors into a model."""
    item: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "name": path.name,
        "model_id": path.stem,
        "size": None,
        "header_valid": False,
        "parsed": False,
    }
    try:
        size = path.stat().st_size
        item["size"] = size
        with path.open("rb") as stream:
            header = stream.read(8)
        if len(header) != 8:
            item["error"] = "file is shorter than the DNNI header"
            return item
        magic, version = struct.unpack("<II", header)
        item["header_valid"] = magic == DNNI_MAGIC and version in (1, 2)
        item["dnni_version"] = version
        if not item["header_valid"]:
            item["error"] = "unrecognized DNNI magic or version"
            return item
        if size > MAX_DNNI_PARSE_BYTES:
            item["note"] = f"graph parse skipped because file exceeds {MAX_DNNI_PARSE_BYTES} bytes"
            return item
        graph = DnniFile.load(path)
        nodes = list(graph.root.walk())
        item.update({
            "parsed": True,
            "root_type": graph.root.type,
            "node_count": len(nodes),
            "node_type_counts": dict(sorted(Counter(node.type for node in nodes).items())),
            "payload_bytes_total": sum(len(node.payload) for node in nodes),
            "unknown_node_types": sorted({node.type for node in nodes if node.type.startswith("0x")}),
        })
    except (OSError, ValueError, struct.error, RecursionError) as error:
        item["error"] = f"{type(error).__name__}: {error}"
    return item


def _inventory_model_store(models_root: str | Path) -> dict[str, Any]:
    """Inventory local DNNI files and produce bounded graph signatures by role."""
    root = Path(models_root).expanduser()
    report: dict[str, Any] = {
        "path": str(root),
        "exists": root.is_dir(),
        "dnni_file_count": 0,
        "truncated": False,
        "files": [],
        "files_by_role": {role: [] for role in (*MODEL_ROLES, "other")},
        "warnings": [],
        "note": (
            "DNNI graph signatures can help compare structures, but they do not prove semantic, "
            "tensor-shape, or voice compatibility and must not be used for automatic substitution."
        ),
    }
    if not root.is_dir():
        report["warnings"].append("Model root is not an existing directory.")
        return report

    grouped: dict[str, list[Path]] = {role: [] for role in (*MODEL_ROLES, "other")}
    all_paths: list[Path] = []
    try:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.casefold() != ".dnni":
                continue
            if len(all_paths) >= MAX_DNNI_INVENTORY_FILES:
                report["truncated"] = True
                break
            all_paths.append(path)
            rel_parts = [part.casefold() for part in path.relative_to(root).parts[:-1]]
            role = next((candidate for candidate in MODEL_ROLES if candidate in rel_parts), "other")
            grouped[role].append(path)
    except OSError as error:
        report["warnings"].append(f"Could not enumerate model directory: {error}")

    report["dnni_file_count"] = len(all_paths)
    report["files"] = [
        {
            "path": path.relative_to(root).as_posix(),
            "name": path.name,
            "model_id": path.stem,
            "role": next((candidate for candidate in MODEL_ROLES
                          if candidate in [part.casefold() for part in path.relative_to(root).parts[:-1]]),
                         "other"),
            "size": path.stat().st_size if path.exists() else None,
        }
        for path in all_paths
    ]
    for role, files in grouped.items():
        summaries = [
            _summarize_dnni(path, root)
            for path in sorted(files, key=lambda p: str(p).casefold())[:MAX_DNNI_SUMMARIES_PER_ROLE]
        ]
        report["files_by_role"][role] = summaries
        if len(files) > MAX_DNNI_SUMMARIES_PER_ROLE:
            report["warnings"].append(
                f"{role}: graph summaries are limited to {MAX_DNNI_SUMMARIES_PER_ROLE} of {len(files)} files."
            )
    if report["truncated"]:
        report["warnings"].append(f"Inventory was capped at {MAX_DNNI_INVENTORY_FILES} files.")
    return report


def _check_store_references(info_summaries: list[dict[str, Any]], models_root: str | Path,
                            inventory: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(models_root).expanduser()
    by_stem: dict[str, list[str]] = {}
    for item in inventory.get("files", []):
        by_stem.setdefault(str(item["model_id"]).casefold(), []).append(item["path"])
    result = []
    for info in info_summaries:
        ref_result = {"info_path": info.get("path"), "models": {}}
        for role, identifier in info.get("model_references", {}).items():
            role_dir = "f0" if role == "f0" else role
            expected = root / role_dir / f"{identifier}.dnni"
            same_id_paths = by_stem.get(identifier.casefold(), [])
            candidates = inventory.get("files_by_role", {}).get(role, [])
            ref_result["models"][role] = {
                "model_id": identifier,
                "expected_path": str(expected),
                "expected_path_found": expected.is_file(),
                "same_id_found_elsewhere": same_id_paths,
                "role_candidate_count": sum(1 for item in inventory.get("files", []) if item.get("role") == role),
                "structural_summaries_shown": len(candidates),
            }
        result.append(ref_result)
    return result

def _info_nofs_summary(name: str, data: bytes, available_paths: list[str]) -> dict[str, Any]:
    """Parse the observed JSON info.nofs and report model references conservatively."""
    result = _json_summary(name, data)
    result["shared_model_metadata"] = True
    if not result.get("parsed"):
        return result
    try:
        value = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return result
    if not isinstance(value, dict):
        result["shared_model_metadata"] = False
        result["note"] = "info.nofs JSON must be an object"
        return result

    result["voice_metadata"] = {
        key: value.get(key)
        for key in ("name", "version", "vendor", "language", "phoneset")
        if isinstance(value.get(key), (str, int, float, bool)) or value.get(key) is None
    }
    styles = value.get("styles")
    result["style_names"] = [
        style.get("name") for style in styles
        if isinstance(style, dict) and isinstance(style.get("name"), str)
    ] if isinstance(styles, list) else []
    expected_roles = ("acoustic", "vocoder", "duration", "f0")
    available = [PurePosixPath(path).as_posix().casefold() for path in available_paths]
    references: dict[str, str] = {}
    expected_paths: dict[str, str] = {}
    found: dict[str, bool] = {}
    malformed: dict[str, str] = {}
    for role in expected_roles:
        reference = value.get(role)
        if not isinstance(reference, str) or not re.fullmatch(r"[0-9a-fA-F]{32}", reference):
            malformed[role] = "missing or not a 32-character hexadecimal model identifier"
            continue
        references[role] = reference
        expected = f"models/{role}/{reference}.dnni".casefold()
        expected_paths[role] = f"models/{role}/{reference}.dnni"
        found[role] = any(path == expected or path.endswith("/" + expected) for path in available)
    result["model_references"] = references
    result["expected_model_paths_by_trainsv_convention"] = expected_paths
    result["model_payload_found_in_container"] = found
    result["model_references_malformed"] = malformed
    result["model_payloads_not_found_in_container"] = [
        role for role, exists in found.items() if not exists
    ]
    result["metadata_schema_fields_missing"] = [
        key for key in ("name", "version", "vendor", "language", "phoneset",
                        "support_languages", "styles", "f0_style", "duration_style")
        if key not in value
    ]
    return result


def _config_asset_summary(name: str, data: bytes, available_paths: list[str]) -> dict[str, Any]:
    """Check config.json asset flags against the exact root filenames in the sample layout."""
    result: dict[str, Any] = {
        "path": name, "parsed": False, "asset_checks": {}, "warnings": []
    }
    if len(data) > MAX_CONFIG_BYTES:
        result["note"] = f"Skipped: larger than {MAX_CONFIG_BYTES} bytes"
        return result
    try:
        value = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        result["note"] = "Not valid UTF-8 JSON"
        return result
    if not isinstance(value, dict):
        result["note"] = "config.json must contain a JSON object"
        return result
    result["parsed"] = True
    paths = {PurePosixPath(path).as_posix().casefold() for path in available_paths}
    for field, asset_name in (("profile", "profile.png"), ("background", "background.png")):
        if field not in value:
            result["warnings"].append(f"required field {field!r} is missing")
            continue
        enabled = value[field]
        present = asset_name.casefold() in paths
        result["asset_checks"][field] = {
            "enabled": enabled if isinstance(enabled, bool) else None,
            "asset": asset_name,
            "present_in_container": present,
        }
        if not isinstance(enabled, bool):
            result["warnings"].append(f"field {field!r} is not a boolean")
        elif enabled and not present:
            result["warnings"].append(
                f"{field}=true but root asset {asset_name!r} was not found in this container"
            )
    for field in ("models", "model_types", "phonesets"):
        if field in value and not isinstance(value[field], list):
            result["warnings"].append(f"field {field!r} is not an array")
    return result

def inspect_sfpk(path: str | Path, max_entries: int = MAX_ENTRIES, models_root: str | Path | None = None) -> dict[str, Any]:
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
        "info_nofs_candidates": [],
        "config_asset_checks": [],
        "model_store_inventory": None,
        "model_store_reference_checks": [],
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
    raw_info_nofs: list[tuple[str, bytes]] = []
    raw_config_json: list[tuple[str, bytes]] = []
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
                if item.name.casefold() == "info.nofs":
                    if size > MAX_CONFIG_BYTES:
                        report["warnings"].append(f"Skipped {rel}: larger than {MAX_CONFIG_BYTES} bytes")
                    else:
                        try:
                            raw_info_nofs.append((rel, item.read_bytes()))
                        except OSError as error:
                            report["warnings"].append(f"Could not read {rel}: {error}")
                elif item.suffix.lower() == ".json":
                    try:
                        raw = item.read_bytes()
                        configs.append(_json_summary(rel, raw))
                        if item.name.casefold() == "config.json":
                            raw_config_json.append((rel, raw))
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
                    if Path(name).name.casefold() == "info.nofs":
                        if info.file_size > MAX_CONFIG_BYTES:
                            report["warnings"].append(f"Skipped {name}: larger than {MAX_CONFIG_BYTES} bytes")
                        else:
                            try:
                                raw_info_nofs.append((name, archive.read(info)))
                            except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                                report["warnings"].append(f"Could not read {name}: {error}")
                    elif Path(name).suffix.lower() == ".json" and info.file_size <= MAX_CONFIG_BYTES:
                        try:
                            raw = archive.read(info)
                            configs.append(_json_summary(name, raw))
                            if Path(name).name.casefold() == "config.json":
                                raw_config_json.append((name, raw))
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
    available_paths = [entry["path"] for entry in report["entries"]]
    info_summaries = [
        _info_nofs_summary(name, data, available_paths)
        for name, data in raw_info_nofs
    ]
    report["info_nofs_candidates"] = info_summaries
    if models_root is not None:
        inventory = _inventory_model_store(models_root)
        report["model_store_inventory"] = inventory
        report["model_store_reference_checks"] = _check_store_references(info_summaries, models_root, inventory)
        for inventory_warning in inventory.get("warnings", []):
            report["warnings"].append("Model store: " + inventory_warning)
    config_checks = [
        _config_asset_summary(name, data, available_paths)
        for name, data in raw_config_json
    ]
    report["config_asset_checks"] = config_checks
    for check in config_checks:
        for warning in check.get("warnings", []):
            report["warnings"].append(f"{check['path']}: {warning}")
    configs.extend(info_summaries)
    report["configuration_candidates"] = configs
    for summary in info_summaries:
        missing_payloads = summary.get("model_payloads_not_found_in_container", [])
        malformed_refs = summary.get("model_references_malformed", {})
        if missing_payloads:
            report["warnings"].append(
                f"{summary['path']}: model payload files were not found inside this container for: "
                + ", ".join(missing_payloads)
                + ". This is an inventory result, not proof that the installed shared-model store lacks them."
            )
        if malformed_refs:
            report["warnings"].append(
                f"{summary['path']}: model identifiers could not be validated for: "
                + ", ".join(sorted(malformed_refs))
            )
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
    for info in report.get("info_nofs_candidates", []):
        metadata = info.get("voice_metadata", {})
        lines.append(
            f"info.nofs {info['path']}: {metadata.get('name', '?')} "
            f"(version {metadata.get('version', '?')}); styles: "
            + (", ".join(info.get("style_names", [])) or "-")
        )
        for role, identifier in info.get("model_references", {}).items():
            presence = info.get("model_payload_found_in_container", {}).get(role, False)
            lines.append(f"  model {role}: {identifier} (payload found in container: {presence})")
        if info.get("metadata_schema_fields_missing"):
            lines.append("  metadata fields missing: " + ", ".join(info["metadata_schema_fields_missing"]))
    for check in report.get("config_asset_checks", []):
        summaries = [
            f"{field}={item.get('enabled')} ({'found' if item.get('present_in_container') else 'not found'}: {item.get('asset')})"
            for field, item in check.get("asset_checks", {}).items()
        ]
        if summaries:
            lines.append(f"Config assets {check['path']}: " + ", ".join(summaries))
    inventory = report.get("model_store_inventory")
    if inventory is not None:
        lines.append(f"Installed model store: {inventory.get('path')} ({inventory.get('dnni_file_count', 0)} .dnni files)")
        for role, files in inventory.get("files_by_role", {}).items():
            if files:
                lines.append(f"  structural candidates under role {role}: {len(files)} summarized")
                for item in files[:5]:
                    graph = (
                        f"v{item.get('dnni_version')}, root={item.get('root_type')}, nodes={item.get('node_count')}"
                        if item.get("parsed") else item.get("note", item.get("error", "header only"))
                    )
                    lines.append(f"    {item.get('path')} ({item.get('size')} bytes; {graph})")
        for info_check in report.get("model_store_reference_checks", []):
            for role, item in info_check.get("models", {}).items():
                if not item.get("expected_path_found"):
                    extra = f"; same ID elsewhere: {', '.join(item.get('same_id_found_elsewhere', []))}" if item.get("same_id_found_elsewhere") else ""
                    lines.append(f"  missing exact {role} ID {item['model_id']}; available same-role models are listed as candidates only{extra}")
    for warning in report.get("warnings", []):
        lines.append("WARNING: " + warning)
    lines.append(report["packaging_note"])
    return "\n".join(lines)
