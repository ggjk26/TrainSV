"""Read-only diagnostics for DNNI containers.

This module intentionally does not try to decode DNNI v3 payloads. High byte
entropy is reported as a clue only; it cannot distinguish encrypted data from
compressed or otherwise encoded data.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
import struct

from .dnni import DNNI_V2_OUTPUT_SEED, FILE_MAGIC, KNOWN_TYPES, DnniFile, OpaqueDnniFile

ENTROPY_SAMPLE_LIMIT = 256 * 1024
VALID_ROLES = ("duration", "pitch", "acoustic", "vocoder")


def _byte_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    length = len(data)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _tree_summary(model: DnniFile) -> tuple[int, int, dict[str, int], list[str]]:
    counts: Counter[str] = Counter()
    unknown: set[str] = set()
    max_depth = 0
    stack = [(model.root, 0)]
    while stack:
        node, depth = stack.pop()
        counts[node.type] += 1
        max_depth = max(max_depth, depth)
        if node.type not in KNOWN_TYPES:
            unknown.add(node.type)
        stack.extend((child, depth + 1) for child in node.children)
    return sum(counts.values()), max_depth, dict(sorted(counts.items())), sorted(unknown)


def diagnose_dnni_bytes(data: bytes, *, role: str | None = None) -> dict:
    """Summarize a DNNI header, generic tree (v1/v2), and safe capability status.

    For v1/v2, parsing the generic container does not prove that a model's
    role-specific graph is supported or that Synthesizer V can load it. For
    v3, only the outer magic/version header is examined and the bytes remain
    opaque.
    """
    if len(data) < 8:
        raise ValueError("truncated DNNI header (expected at least 8 bytes)")
    magic, version = struct.unpack_from("<II", data, 0)
    if magic != FILE_MAGIC:
        raise ValueError("not a DNNI file")
    if role is not None and role not in VALID_ROLES:
        raise ValueError(f"role must be one of {', '.join(VALID_ROLES)}")

    payload = data[8:]
    sample = payload[:ENTROPY_SAMPLE_LIMIT]
    entropy = _byte_entropy(sample)
    high_entropy = entropy >= 7.75
    report = {
        "format": "DNNI",
        "version": version,
        "role_context": role,
        "role_was_inferred": False,
        "file_size_bytes": len(data),
        "payload_size_bytes": len(payload),
        "sha256": hashlib.sha256(data).hexdigest(),
        "entropy_sample_bytes": len(sample),
        "payload_sample_entropy_bits_per_byte": round(entropy, 6),
        "high_entropy_sample": high_entropy,
        "entropy_interpretation": (
            "The sample is near-uniform; this is consistent with encryption, compression, or another encoding, "
            "but entropy alone cannot distinguish them."
            if high_entropy else
            "This sample is not near-uniform; byte entropy alone does not establish the format or its compatibility."
        ),
        "generic_container_parse": "not_attempted",
        "node_count": None,
        "max_tree_depth": None,
        "node_type_counts": {},
        "unknown_node_types": [],
        "byte_preservation_supported": version == 3,
        "trainable_by_trainsv": False if version == 3 else None,
        "inference_supported_by_trainsv": False if version == 3 else None,
        "role_compatibility": "not_checked",
        "synthesizer_v_runtime_compatibility": "not_checked",
    }

    if version == 3:
        # Validate the same minimum header contract used by the opaque wrapper.
        OpaqueDnniFile.from_bytes(data)
        report["generic_container_parse"] = "opaque-v3"
        report["capability_summary"] = (
            "TrainSV can preserve these bytes unchanged, but cannot inspect the v3 graph/tensors, "
            "run inference, or train this model."
        )
        return report

    if version not in (1, 2):
        report["generic_container_parse"] = "unsupported-version"
        report["capability_summary"] = (
            "TrainSV's generic DNNI parser supports container versions 1 and 2; this version is not enabled."
        )
        report["byte_preservation_supported"] = False
        return report

    try:
        model = DnniFile.parse(data)
    except (ValueError, struct.error) as error:
        report["generic_container_parse"] = "parse-failed"
        report["parse_error"] = str(error)
        report["capability_summary"] = (
            "The file has a DNNI v1/v2 header, but TrainSV could not parse its generic container structure."
        )
        return report

    node_count, max_depth, node_types, unknown_types = _tree_summary(model)
    report.update({
        "generic_container_parse": "parsed-v1-v2",
        "node_count": node_count,
        "max_tree_depth": max_depth,
        "node_type_counts": node_types,
        "unknown_node_types": unknown_types,
        "capability_summary": (
            "The generic container tree parsed. Role-specific network support, tensor layouts, and Synthesizer V "
            "runtime compatibility still require separate validation."
        ),
    })
    return report


def diagnose_dnni_path(path: str | Path, *, role: str | None = None) -> dict:
    """Read a .dnni path and return a JSON-serializable diagnostic report."""
    model_path = Path(path).expanduser()
    report = diagnose_dnni_bytes(model_path.read_bytes(), role=role)
    report["path"] = str(model_path.resolve())
    return report
