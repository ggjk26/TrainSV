"""Architecture diagnostics for NOFS voice files.

Parsing a NOFS container is not the same as supporting its neural architecture.
The report separates container parsing, DNNI graph parsing, and TrainSV's known
architecture identifiers so unsupported models are not silently treated as compatible.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dnni import DnniFile, KNOWN_TYPES
from .nofs import NofsFile, is_info_voice, InfoVoice

ROLE_FIELDS = {
    "duration": ("model_duration", "model_duration_arch", b"duration"),
    "acoustic": ("model_timbre_pred", "model_timbre_arch", b"acoustic"),
    "vocoder": ("model_vocoder", "model_vocoder_arch", b"vocoder"),
    "pitch": (None, None, b"f0"),
}
SUPPORTED_ARCHITECTURES = {"duration": "gen2a", "acoustic": "dds", "vocoder": "nhv"}
KNOWN_TYPE_NAMES = set(KNOWN_TYPES)


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "backslashreplace")
    return str(value)


def diagnose_voice(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser()
    report: dict[str, Any] = {
        "path": str(target),
        "exists": target.is_file(),
        "container_parsed": False,
        "voice_kind": "unknown",
        "architecture_identifiers": {},
        "models": {},
        "warnings": [],
        "trainability": "unknown",
        "note": (
            "This is a structural diagnostic, not a compatibility guarantee. "
            "TrainSV's model constructors may impose additional graph and tensor-shape requirements."
        ),
    }
    if not target.is_file():
        report["warnings"].append("Input path is not a file.")
        return report

    model_sources: dict[str, bytes] = {}
    architectures: dict[str, str] = {}
    try:
        if is_info_voice(target):
            report["voice_kind"] = "info-shared-model"
            info = InfoVoice.load(target)
            report["container_parsed"] = True
            for role, model_path in info.models.items():
                try:
                    model_sources[role] = model_path.read_bytes()
                except OSError as error:
                    report["models"][role] = {"parsed": False, "error": str(error)}
            report["warnings"].append(
                "Info/shared-model voices reference separate DNNI files; architecture identifiers are not stored in the same NOFS config fields."
            )
        else:
            report["voice_kind"] = "nofs-v10"
            nofs = NofsFile.load(target)
            report["container_parsed"] = True
            fields = nofs.configuration()
            model_key_fields = {
                "duration": "model_duration",
                "acoustic": "model_timbre_pred",
                "vocoder": "model_vocoder",
            }
            for role, field_name in model_key_fields.items():
                architecture_field = ROLE_FIELDS[role][1]
                if field_name in fields and fields[field_name]:
                    key = fields[field_name][0]
                    try:
                        model_sources[role] = nofs.read(key)
                    except KeyError:
                        report["models"][role] = {"parsed": False, "error": f"model key {key!r} is absent from the NOFS index"}
                else:
                    report["models"][role] = {"parsed": False, "error": f"configuration field {field_name!r} is missing"}
                if architecture_field in fields and fields[architecture_field]:
                    architectures[role] = _decode(fields[architecture_field][0])
                else:
                    architectures[role] = "missing"
            pitch_key = b"f0model-dds"
            try:
                model_sources["pitch"] = nofs.read(pitch_key)
            except KeyError:
                report["models"]["pitch"] = {"parsed": False, "error": f"pitch model key {pitch_key!r} is absent"}
            report["architecture_identifiers"] = architectures
    except Exception as error:  # diagnostics should report malformed inputs, not hide context
        report["warnings"].append(f"Container/configuration parsing failed: {type(error).__name__}: {error}")
        report["trainability"] = "not-established"
        return report

    for role, data in model_sources.items():
        try:
            graph = DnniFile.parse(data)
            nodes = list(graph.root.walk())
            types = sorted({node.type for node in nodes})
            unknown = [name for name in types if name not in KNOWN_TYPE_NAMES]
            report["models"][role] = {
                "parsed": True,
                "dnni_version": graph.version,
                "node_count": len(nodes),
                "node_types": types,
                "unknown_node_types": unknown,
            }
            if unknown:
                report["warnings"].append(f"{role}: DNNI graph contains node types not recognized by TrainSV: {', '.join(unknown)}")
        except Exception as error:
            report["models"][role] = {"parsed": False, "error": f"{type(error).__name__}: {error}"}

    if report["voice_kind"] == "nofs-v10":
        mismatches = [
            f"{role}={architectures.get(role, 'missing')} (expected {expected})"
            for role, expected in SUPPORTED_ARCHITECTURES.items()
            if architectures.get(role) != expected
        ]
        if mismatches:
            report["warnings"].append("Architecture identifiers outside TrainSV's current allowlist: " + "; ".join(mismatches))
        all_models_parsed = all(report["models"].get(role, {}).get("parsed", False) for role in ("duration", "pitch", "acoustic", "vocoder"))
        all_node_types_known = all(not report["models"].get(role, {}).get("unknown_node_types") for role in ("duration", "pitch", "acoustic", "vocoder"))
        report["trainability"] = "candidate-current-architecture" if not mismatches and all_models_parsed and all_node_types_known else "requires-adaptation-or-repair"
    else:
        all_models_parsed = all(report["models"].get(role, {}).get("parsed", False) for role in ("duration", "pitch", "acoustic", "vocoder"))
        report["trainability"] = "candidate-needs-runtime-check" if all_models_parsed else "requires-adaptation-or-repair"
    return report


def format_diagnostic(report: dict[str, Any]) -> str:
    lines = [
        f"Voice: {report['path']}",
        f"Container parsed: {report['container_parsed']}",
        f"Voice kind: {report['voice_kind']}",
        f"Preliminary status: {report['trainability']}",
    ]
    for role, architecture in report.get("architecture_identifiers", {}).items():
        lines.append(f"Architecture {role}: {architecture}")
    for role, model in report.get("models", {}).items():
        if model.get("parsed"):
            lines.append(f"{role}: DNNI v{model['dnni_version']}, {model['node_count']} nodes, types: {', '.join(model['node_types'])}")
        else:
            lines.append(f"{role}: NOT PARSED — {model.get('error', 'unknown error')}")
    for warning in report.get("warnings", []):
        lines.append("WARNING: " + warning)
    lines.append(report["note"])
    return "\n".join(lines)
