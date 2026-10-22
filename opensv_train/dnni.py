from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

FILE_MAGIC = 0x7FCA00FF
NODE_MARKERS = (0x7FCA40FF, 0x7FCA41FF)
CHILD_MASK = 0xAC5E7BD5

HASH_SEEDS = (
    0x0DCD59189D5A0F24, 0x59346D79970CA21E, 0xCF3519B773B767BF, 0x562C8E41FB7FBEE2,
    0xBD6D25457E1ED24E, 0x0123456789ABCDEF, 0x76543210FEDCBA98, 0x02468ACEECA86420,
    0xECA864202468ACEE, 0x70556F5965766947, 0x5A4D5A4D5A4D5A4D, 0x000000626D6F6379,
)

KNOWN_TYPES = (
    "prim0", "prim1", "prim2", "prim3", "prim4", "prim5",
    "modm0", "modl0", "modl1", "modl3", "modl4", "modl6",
    "moda0", "moda1", "moda2", "moda3", "moda4", "moda5", "moda7",
    "_gnc1v0", "_ncwnv0", "cmpg1", "cmpu0", "cmpu1",
    "_vocfv1", "_vocfv2", "_ppusv0", "_ppdsv0", "_psv2",
    "_rldtg0", "_rldms0", "_stbkv1", "_vqctx1", "_didsv0",
    "_ftmfv2", "_ftmfv3", "_dctov0", "_nhntv2", "_cepsg1", "_ltvfs1",
)

MATRIX_TYPES = ("prim0", "prim2", "prim3", "prim4", "prim5")


def type_hash(name: str, seed: int) -> int:
    value = seed
    for byte in name.encode("ascii"):
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


_HASHED_NAMES = {type_hash(name, seed): name for name in KNOWN_TYPES for seed in HASH_SEEDS}
# Stable known seed for conversion from literal-tag V1 files to hashed V2 files.
# Actual runtime loading still needs validation in the target Synthesizer V release.
DNNI_V2_OUTPUT_SEED = HASH_SEEDS[0]


@dataclass
class Node:
    marker: int
    type_id: int
    type: str
    payload: bytes
    children: list[Node] = field(default_factory=list)

    def child(self, *path: int) -> Node:
        node = self
        for index in path:
            node = node.children[index]
        return node

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


class DnniFile:
    def __init__(self, version: int, root: Node):
        self.version = version
        self.root = root

    @classmethod
    def parse(cls, data: bytes) -> DnniFile:
        magic, version = struct.unpack_from("<II", data, 0)
        if magic != FILE_MAGIC:
            raise ValueError("not a DNNI file")
        if version not in (1, 2):
            raise ValueError(f"unsupported DNNI version {version}")
        position, root = _parse_node(data, 8, version, 0)
        if position != len(data):
            raise ValueError("trailing bytes after the DNNI root node")
        return cls(version, root)

    @classmethod
    def load(cls, path) -> DnniFile:
        with open(path, "rb") as stream:
            return cls.parse(stream.read())

    def serialize(self) -> bytes:
        parts = [struct.pack("<II", FILE_MAGIC, self.version)]
        _serialize_node(self.root, self.version, parts)
        return b"".join(parts)

    def as_version(self, version: int, *, seed: int = DNNI_V2_OUTPUT_SEED) -> DnniFile:
        """Return a copy encoded with another supported DNNI container version.

        V1 stores literal ASCII type tags; V2 stores type hashes and an encoded
        child count. This converts both fields. It does not change model
        architecture or weights; runtime compatibility must still be tested.
        """
        if version not in (1, 2):
            raise ValueError(f"unsupported DNNI version {version}")
        if version == self.version:
            return self
        if version == 2 and seed not in HASH_SEEDS:
            raise ValueError("DNNI v2 output seed must be one of the recognized hash seeds")

        def convert(node: Node) -> Node:
            if node.type not in KNOWN_TYPES:
                raise ValueError(f"cannot convert unknown DNNI node type {node.type!r} to version {version}")
            if version == 2:
                type_id = type_hash(node.type, seed)
            else:
                encoded = node.type.encode("ascii")
                if len(encoded) > 8:
                    raise ValueError(f"DNNI v1 type tag is longer than 8 bytes: {node.type!r}")
                type_id = int.from_bytes(encoded.ljust(8, bytes(1)), "little")
            return Node(
                marker=node.marker, type_id=type_id, type=node.type,
                payload=node.payload, children=[convert(child) for child in node.children],
            )

        return DnniFile(version, convert(self.root))


def serialize_model_for_output(model: DnniFile, role: str, acoustic_dnni_version: int = 2) -> bytes:
    """Serialize only the acoustic model with the requested DNNI container version."""
    if role == "acoustic":
        return model.as_version(acoustic_dnni_version).serialize()
    return model.serialize()


def _resolve_type(type_bytes: bytes, type_id: int, version: int) -> str:
    if version == 1:
        tag = type_bytes.split(b"\0", 1)[0]
        if not tag or type_bytes[len(tag):].strip(b"\0") or not all(0x20 <= byte <= 0x7E for byte in tag):
            raise ValueError("invalid DNNI v1 type tag")
        return tag.decode("ascii")
    return _HASHED_NAMES.get(type_id, f"0x{type_id:016x}")


def _parse_node(data: bytes, position: int, version: int, depth: int) -> tuple[int, Node]:
    if depth > 256:
        raise ValueError("DNNI node depth limit exceeded")
    marker, type_low, type_high, child_field, payload_size = struct.unpack_from("<IIIII", data, position)
    if marker not in NODE_MARKERS:
        raise ValueError(f"bad DNNI node marker at 0x{position:x}")
    type_id = type_low | (type_high << 32)
    type_name = _resolve_type(data[position + 4:position + 12], type_id, version)
    if version == 2:
        child_field ^= type_low ^ type_high ^ CHILD_MASK
        if child_field % 3:
            raise ValueError(f"bad v2 child count at 0x{position:x}")
        child_field //= 3
    payload_start = position + 20
    payload = data[payload_start:payload_start + payload_size]
    if len(payload) != payload_size:
        raise ValueError("DNNI payload extends past the file")
    node = Node(marker, type_id, type_name, payload)
    position = payload_start + payload_size
    for _ in range(child_field):
        position, child = _parse_node(data, position, version, depth + 1)
        node.children.append(child)
    return position, node


def _serialize_node(node: Node, version: int, parts: list[bytes]) -> None:
    type_low = node.type_id & 0xFFFFFFFF
    type_high = node.type_id >> 32
    child_field = len(node.children)
    if version == 2:
        child_field = (child_field * 3) ^ type_low ^ type_high ^ CHILD_MASK
    parts.append(struct.pack("<IIIII", node.marker, type_low, type_high, child_field, len(node.payload)))
    parts.append(node.payload)
    for child in node.children:
        _serialize_node(child, version, parts)


def read_vector(node: Node) -> np.ndarray:
    if node.type != "prim1":
        raise ValueError(f"expected prim1, found {node.type}")
    (count,) = struct.unpack_from("<I", node.payload, 0)
    if len(node.payload) != 4 + 4 * count:
        raise ValueError("prim1 size does not match its payload")
    return np.frombuffer(node.payload, "<f4", count, 4).copy()


def write_vector(node: Node, values: np.ndarray) -> None:
    values = np.asarray(values, dtype="<f4").reshape(-1)
    if node.type != "prim1" or len(values) != struct.unpack_from("<I", node.payload, 0)[0]:
        raise ValueError("vector replacement must keep the prim1 length")
    node.payload = struct.pack("<I", len(values)) + values.tobytes()


@dataclass
class _Quantization:
    bits: int
    word1: int
    scales: np.ndarray
    header_size: int


def _read_quantization(payload: bytes) -> _Quantization:
    bits, word1, mode, count = struct.unpack_from("<IIII", payload, 0)
    if bits not in (8, 16):
        raise ValueError(f"unsupported quantized width {bits}")
    if mode != 0:
        raise ValueError("residual quantization is not supported")
    scales = np.frombuffer(payload, "<f4", count, 16).copy()
    return _Quantization(bits, word1, scales, 16 + 4 * count)


def _int_dtype(bits: int) -> str:
    return "<i1" if bits == 8 else "<i2"


@dataclass
class SparseLayout:
    block_rows: int
    block_columns: int
    column_index: np.ndarray
    row_pointer: np.ndarray

    def mask(self, rows: int, columns: int) -> np.ndarray:
        mask = np.zeros((rows, columns), dtype=bool)
        for block_row in range(rows // self.block_rows):
            for block in range(self.row_pointer[block_row], self.row_pointer[block_row + 1]):
                column = self.column_index[block] * self.block_columns
                row = block_row * self.block_rows
                mask[row:row + self.block_rows, column:column + self.block_columns] = True
        return mask


def _sparse_layout(body: bytes) -> tuple[int, int, SparseLayout, int]:
    rows, columns, block_rows, block_columns, block_count = struct.unpack_from("<IIIII", body, 0)
    pointers = rows // block_rows + 1
    column_index = np.frombuffer(body, "<u2", block_count, 20).astype(np.int64)
    row_pointer = np.frombuffer(body, "<u4", pointers, 20 + 2 * block_count).astype(np.int64)
    if row_pointer[0] != 0 or row_pointer[-1] != block_count:
        raise ValueError("sparse row pointers do not span all blocks")
    return rows, columns, SparseLayout(block_rows, block_columns, column_index, row_pointer), 20 + 2 * block_count + 4 * pointers


def _blocks_to_dense(rows, columns, layout: SparseLayout, blocks: np.ndarray) -> np.ndarray:
    dense = np.zeros((rows, columns), dtype=np.float64)
    br, bc = layout.block_rows, layout.block_columns
    for block_row in range(rows // br):
        for block in range(layout.row_pointer[block_row], layout.row_pointer[block_row + 1]):
            column = layout.column_index[block] * bc
            dense[block_row * br:(block_row + 1) * br, column:column + bc] += blocks[block]
    return dense


def _dense_to_blocks(matrix: np.ndarray, layout: SparseLayout) -> np.ndarray:
    br, bc = layout.block_rows, layout.block_columns
    blocks = np.zeros((len(layout.column_index), br, bc), dtype=np.float64)
    seen = set()
    for block_row in range(matrix.shape[0] // br):
        for block in range(layout.row_pointer[block_row], layout.row_pointer[block_row + 1]):
            key = (block_row, layout.column_index[block])
            if key in seen:
                continue
            seen.add(key)
            column = layout.column_index[block] * bc
            blocks[block] = matrix[block_row * br:(block_row + 1) * br, column:column + bc]
    return blocks


def read_matrix(node: Node) -> np.ndarray:
    payload = node.payload
    if node.type in ("prim0", "prim2"):
        rows, columns = struct.unpack_from("<II", payload, 0)
        return np.frombuffer(payload, "<f4", rows * columns, 8).reshape(rows, columns).copy()
    if node.type == "prim4":
        quantization = _read_quantization(payload)
        rows, columns = struct.unpack_from("<II", payload, quantization.header_size)
        integers = np.frombuffer(payload, _int_dtype(quantization.bits), rows * columns, quantization.header_size + 8)
        divisor = float(2 ** (quantization.bits - 1))
        values = integers.reshape(columns, rows).T.astype(np.float32) / np.float32(divisor) * quantization.scales[:, None]
        return values.astype(np.float32)
    if node.type in ("prim3", "prim5"):
        quantized = node.type == "prim5"
        quantization = _read_quantization(payload) if quantized else None
        body = payload[quantization.header_size:] if quantized else payload
        rows, columns, layout, offset = _sparse_layout(body)
        count = len(layout.column_index) * layout.block_rows * layout.block_columns
        shape = (len(layout.column_index), layout.block_columns, layout.block_rows)
        if quantized:
            integers = np.frombuffer(body, _int_dtype(quantization.bits), count, offset).reshape(shape)
            blocks = integers.transpose(0, 2, 1).astype(np.float32) / np.float32(2 ** (quantization.bits - 1))
            row_scales = quantization.scales.reshape(rows // layout.block_rows, layout.block_rows)
            block_rows = np.repeat(np.arange(rows // layout.block_rows), np.diff(layout.row_pointer))
            blocks = blocks * row_scales[block_rows][:, :, None]
        else:
            blocks = np.frombuffer(body, "<f4", count, offset).reshape(shape).transpose(0, 2, 1)
        return _blocks_to_dense(rows, columns, layout, blocks.astype(np.float32)).astype(np.float32)
    raise ValueError(f"{node.type} is not a matrix primitive")


def matrix_shape(node: Node) -> tuple[int, int]:
    payload = node.payload
    if node.type in ("prim4", "prim5"):
        payload = payload[_read_quantization(payload).header_size:]
    return struct.unpack_from("<II", payload, 0)


def sparsity_mask(node: Node) -> np.ndarray | None:
    if node.type not in ("prim3", "prim5"):
        return None
    payload = node.payload
    if node.type == "prim5":
        payload = payload[_read_quantization(payload).header_size:]
    rows, columns, layout, _ = _sparse_layout(payload)
    return layout.mask(rows, columns)


def _quantize_rows(matrix: np.ndarray, bits: int, reference_scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    limit = 2 ** (bits - 1)
    peak = np.abs(matrix).max(axis=1)
    scales = np.where(peak <= reference_scales, reference_scales, peak)
    scales = np.where(scales > 0, scales, 1.0).astype(np.float32)
    integers = np.clip(np.rint(matrix / scales[:, None] * limit), -limit, limit - 1)
    return integers.astype(_int_dtype(bits)), scales


def write_matrix(node: Node, matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != matrix_shape(node):
        raise ValueError(f"matrix replacement must keep shape {matrix_shape(node)}, got {matrix.shape}")
    payload = node.payload
    if node.type in ("prim0", "prim2"):
        node.payload = payload[:8] + matrix.astype("<f4").tobytes()
    elif node.type == "prim4":
        quantization = _read_quantization(payload)
        integers, scales = _quantize_rows(matrix, quantization.bits, quantization.scales)
        header = struct.pack("<IIII", quantization.bits, quantization.word1, 0, len(scales)) + scales.astype("<f4").tobytes()
        node.payload = header + struct.pack("<II", *matrix.shape) + integers.T.tobytes()
    elif node.type in ("prim3", "prim5"):
        quantized = node.type == "prim5"
        quantization = _read_quantization(payload) if quantized else None
        body = payload[quantization.header_size:] if quantized else payload
        rows, columns, layout, offset = _sparse_layout(body)
        blocks = _dense_to_blocks(matrix, layout)
        if quantized:
            masked = np.where(layout.mask(rows, columns), matrix, 0.0)
            integers, scales = _quantize_rows(masked, quantization.bits, quantization.scales)
            limit = 2 ** (quantization.bits - 1)
            block_row_of = np.repeat(np.arange(rows // layout.block_rows), np.diff(layout.row_pointer))
            row_scales = scales.reshape(rows // layout.block_rows, layout.block_rows)[block_row_of]
            coded = np.clip(np.rint(blocks / row_scales[:, :, None] * limit), -limit, limit - 1).astype(_int_dtype(quantization.bits))
            header = struct.pack("<IIII", quantization.bits, quantization.word1, 0, len(scales)) + scales.astype("<f4").tobytes()
            node.payload = header + body[:offset] + coded.transpose(0, 2, 1).tobytes()
        else:
            node.payload = body[:offset] + blocks.astype("<f4").transpose(0, 2, 1).tobytes()
    else:
        raise ValueError(f"{node.type} is not a matrix primitive")


def convert_to_dense_float(node: Node, matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype="<f4")
    seed = next((seed for seed in HASH_SEEDS if type_hash(node.type, seed) == node.type_id), None)
    node.type = "prim0"
    if seed is not None:
        node.type_id = type_hash("prim0", seed)
    else:
        node.type_id = int.from_bytes(b"prim0\0\0\0", "little")
    node.payload = struct.pack("<II", *matrix.shape) + matrix.tobytes()
