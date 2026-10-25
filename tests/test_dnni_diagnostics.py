from __future__ import annotations

import struct
from pathlib import Path

import pytest

from opensv_train.dnni import FILE_MAGIC, NODE_MARKERS, DnniFile, Node
from opensv_train.dnni_diagnostics import diagnose_dnni_bytes


def _minimal_v1_model() -> bytes:
    leaf_tag = b"prim1" + bytes(3)
    leaf = Node(
        NODE_MARKERS[0],
        int.from_bytes(leaf_tag, "little"),
        "prim1",
        struct.pack("<I", 2) + struct.pack("<2f", 0.25, -0.5),
        [],
    )
    root_tag = b"_rldms0" + bytes(1)
    root = Node(
        NODE_MARKERS[0],
        int.from_bytes(root_tag, "little"),
        "_rldms0",
        b"",
        [leaf],
    )
    return DnniFile(1, root).serialize()


def test_v1_v2_diagnostics_parse_generic_tree_but_do_not_claim_runtime_support():
    report = diagnose_dnni_bytes(_minimal_v1_model(), role="acoustic")

    assert report["version"] == 1
    assert report["generic_container_parse"] == "parsed-v1-v2"
    assert report["node_count"] == 2
    assert report["max_tree_depth"] == 1
    assert report["node_type_counts"] == {"_rldms0": 1, "prim1": 1}
    assert report["role_context"] == "acoustic"
    assert report["role_was_inferred"] is False
    assert report["trainable_by_trainsv"] is None
    assert report["role_compatibility"] == "not_checked"
    assert report["synthesizer_v_runtime_compatibility"] == "not_checked"


def test_v3_diagnostics_mark_payload_opaque_and_preservation_only():
    raw = struct.pack("<II", FILE_MAGIC, 3) + bytes(range(256)) * 64
    report = diagnose_dnni_bytes(raw, role="acoustic")

    assert report["version"] == 3
    assert report["generic_container_parse"] == "opaque-v3"
    assert report["byte_preservation_supported"] is True
    assert report["trainable_by_trainsv"] is False
    assert report["inference_supported_by_trainsv"] is False
    assert report["node_count"] is None
    assert report["payload_sample_entropy_bits_per_byte"] == 8.0
    assert "cannot distinguish" in report["entropy_interpretation"]
    assert "preserve" in report["capability_summary"]


def test_diagnostics_reject_bad_magic_and_truncated_headers():
    with pytest.raises(ValueError, match="truncated DNNI header"):
        diagnose_dnni_bytes(b"abc")
    with pytest.raises(ValueError, match="not a DNNI file"):
        diagnose_dnni_bytes(struct.pack("<II", 0, 3) + b"opaque")


def test_diagnostics_report_unknown_versions_without_mislabeling_them_as_v3():
    raw = struct.pack("<II", FILE_MAGIC, 4) + b"future-version-data"
    report = diagnose_dnni_bytes(raw)

    assert report["version"] == 4
    assert report["generic_container_parse"] == "unsupported-version"
    assert report["byte_preservation_supported"] is False
    assert report["trainable_by_trainsv"] is None


def test_inspect_dnni_cli_exposes_role_and_json_report_options_without_importing_training_runtime():
    root = Path(__file__).resolve().parents[1]
    cli = (root / "opensv_train" / "__main__.py").read_text("utf-8")

    assert 'commands.add_parser("inspect-dnni"' in cli
    assert 'command.add_argument("--model", required=True' in cli
    assert 'command.add_argument("--role", choices=("duration", "pitch", "acoustic", "vocoder")' in cli
    assert 'command.add_argument("--json-out"' in cli
    assert "command.set_defaults(handler=inspect_dnni_command)" in cli
