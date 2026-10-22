import struct

import pytest

from opensv_train.dnni import DNNI_V2_OUTPUT_SEED, NODE_MARKERS, Node, DnniFile, type_hash
from opensv_train.voice import serialize_model_for_output


def _v1_model():
    leaf_tag = b"prim1".ljust(8, b"\\0")
    leaf = Node(NODE_MARKERS[0], int.from_bytes(leaf_tag, "little"), "prim1",
                struct.pack("<I", 2) + struct.pack("<2f", 0.25, -0.5), [])
    root_tag = b"_rldms0".ljust(8, b"\\0")
    root = Node(NODE_MARKERS[0], int.from_bytes(root_tag, "little"), "_rldms0", b"", [leaf])
    return DnniFile(1, root)


def _types(model):
    return [node.type for node in model.root.walk()]


def test_v1_to_v2_conversion_roundtrips_known_type_ids_and_child_count():
    v1 = _v1_model()
    converted = v1.as_version(2)
    assert converted.version == 2
    assert converted.root.type_id == type_hash("_rldms0", DNNI_V2_OUTPUT_SEED)
    assert converted.root.children[0].type_id == type_hash("prim1", DNNI_V2_OUTPUT_SEED)
    parsed = DnniFile.parse(converted.serialize())
    assert parsed.version == 2
    assert _types(parsed) == _types(v1)
    assert parsed.root.children[0].payload == v1.root.children[0].payload
    assert DnniFile.parse(parsed.as_version(1).serialize()).version == 1


def test_output_serialization_converts_only_the_acoustic_model_by_default():
    source = _v1_model()
    acoustic_bytes = serialize_model_for_output(source, "acoustic")
    duration_bytes = serialize_model_for_output(source, "duration")
    assert DnniFile.parse(acoustic_bytes).version == 2
    assert DnniFile.parse(duration_bytes).version == 1
    assert DnniFile.parse(serialize_model_for_output(source, "acoustic", 1)).version == 1


def test_v2_conversion_rejects_unrecognized_hash_seed_and_unknown_node_type():
    with pytest.raises(ValueError, match="recognized hash seeds"):
        _v1_model().as_version(2, seed=1)
    unknown = _v1_model()
    unknown.root.type = "private-unknown"
    with pytest.raises(ValueError, match="unknown DNNI node type"):
        unknown.as_version(2)
