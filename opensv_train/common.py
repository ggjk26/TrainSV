from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .dnni import Node, read_vector

VOWEL_CATEGORIES = ("vowel", "diphthong")


@dataclass
class PhoneSet:
    name: str
    symbols: list[str]
    categories: list[str]
    unified: list[str]
    classes: list[str]

    def is_vowel(self, symbol: str) -> bool:
        return self.categories[self.symbols.index(symbol)] in VOWEL_CATEGORIES


def read_phone_set(node: Node) -> PhoneSet:
    if node.type != "_psv2" or node.children:
        raise ValueError("phone tables must be leaf _psv2 nodes")
    data = node.payload
    position = 0

    def string():
        nonlocal position
        (length,) = struct.unpack_from("<I", data, position)
        value = data[position + 4:position + 4 + length].decode("utf-8")
        position += 4 + length
        return value

    def strings():
        nonlocal position
        (count,) = struct.unpack_from("<I", data, position)
        position += 4
        return [string() for _ in range(count)]

    phone_set = PhoneSet(string(), strings(), strings(), strings(), strings())
    if position != len(data) or len(phone_set.categories) != len(phone_set.symbols) or len(phone_set.unified) < len(phone_set.symbols):
        raise ValueError(f"invalid _psv2 table {phone_set.name}")
    return phone_set


def find_language(phone_sets: list[PhoneSet], language: str, exact: bool = False) -> int:
    matches = [index for index, phone_set in enumerate(phone_sets) if (phone_set.name == language if exact else language in phone_set.name)]
    if len(matches) != 1:
        names = ", ".join(phone_set.name for phone_set in phone_sets)
        raise ValueError(f"language {language!r} matches {len(matches)} phone sets ({names})")
    return matches[0]


class MinMaxNormalizer(nn.Module):

    def __init__(self, node: Node):
        super().__init__()
        if node.type != "cmpu0" or node.payload or len(node.children) != 3:
            raise ValueError("cmpu0 requires three vector children")
        lower, upper, target = (read_vector(child) for child in node.children)
        self.register_buffer("lower", torch.from_numpy(lower))
        self.register_buffer("upper", torch.from_numpy(upper))
        self.target_lower = float(target[0])
        self.target_upper = float(target[1])

    @property
    def channels(self) -> int:
        return self.lower.numel()

    def normalize(self, x: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        target_range = np.float32(self.target_upper) - np.float32(self.target_lower)
        y = self.target_lower + float(target_range) * (x - self.lower) / torch.clamp(self.upper - self.lower, min=1e-5)
        return torch.clamp(y, self.target_lower, self.target_upper) if clamp else y

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        target_range = float(np.float32(self.target_upper) - np.float32(self.target_lower))
        return self.lower + (self.upper - self.lower) * (y - self.target_lower) / target_range


class MeanScaleNormalizer(nn.Module):

    def __init__(self, node: Node):
        super().__init__()
        if node.payload or len(node.children) != 2:
            raise ValueError("cmpu1 requires two vector children")
        self.register_buffer("mean", torch.from_numpy(read_vector(node.children[0])))
        self.register_buffer("scale", torch.from_numpy(read_vector(node.children[1])))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.clamp(self.scale, min=1e-5)


def sinusoid(values: torch.Tensor, channels: int) -> torch.Tensor:
    half = channels // 2
    frequencies = torch.exp2(torch.arange(half, dtype=torch.float32, device=values.device) * np.float32(-26.575424194335938) / np.float32(channels))
    angles = values.unsqueeze(-1) * frequencies
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def _polar(first: np.ndarray, second: np.ndarray, mt_variant: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        squared = second * second + first * first
        if not mt_variant:
            accepted = (squared <= 1.0) & (squared != 0.0)
            scale = np.sqrt(np.float32(-2.0) * np.log(squared) / squared)
            return accepted, first * scale, second * scale
        accepted = (squared < 1.0) & (first != 0.0) & (second != 0.0)
        small = accepted & (squared <= np.float32(0.0001))
        exponent = np.frexp(np.maximum(np.abs(first), np.abs(second)))[1] - 1
        first = np.where(small, np.ldexp(first, -exponent), first).astype(np.float32)
        second = np.where(small, np.ldexp(second, -exponent), second).astype(np.float32)
        squared = np.where(small, second * second + first * first, squared).astype(np.float32)
        logarithm = np.log(squared) + np.where(small, exponent.astype(np.float32) * np.float32(1.3862943649291992), np.float32(0.0))
        scale = np.sqrt(np.float32(-2.0) * logarithm / squared)
        return accepted, first * scale, second * scale


class EngineGaussian:

    def __init__(self, seed: int):
        self.rng = np.random.RandomState(seed)
        self.spare: float | None = None

    def reseed(self, seed: int) -> None:
        self.rng.seed(seed)

    def take(self, count: int) -> np.ndarray:
        output = []
        if count > 0 and self.spare is not None:
            output.append(np.float32(self.spare))
            self.spare = None
        needed = count - len(output)
        while needed > 0:
            state = self.rng.get_state()
            attempts = needed // 2 + 16 + needed // 3
            draws = self.rng.randint(0, 2 ** 32, size=2 * attempts, dtype=np.uint64).astype(np.float32)
            first = draws[0::2] * np.float32(2.0 ** -31) - np.float32(1.0)
            second = draws[1::2] * np.float32(2.0 ** -31) - np.float32(1.0)
            accepted, out_first, out_second = _polar(first, second, True)
            indices = np.flatnonzero(accepted)
            pairs = (needed + 1) // 2
            if len(indices) < pairs:
                self.rng.set_state(state)
                self.rng.randint(0, 2 ** 32, size=2 * attempts, dtype=np.uint64)
                values = np.stack([out_first[indices], out_second[indices]], axis=1).reshape(-1)
                output.extend(values)
                needed -= len(values)
                continue
            used = indices[:pairs]
            self.rng.set_state(state)
            self.rng.randint(0, 2 ** 32, size=2 * (used[-1] + 1), dtype=np.uint64)
            values = np.stack([out_first[used], out_second[used]], axis=1).reshape(-1)
            output.extend(values[:needed])
            if needed % 2:
                self.spare = float(values[-1])
            needed = 0
        return np.asarray(output, dtype=np.float32)


_RESIDUAL_NOISE = np.zeros(0, dtype=np.float32)


def _minstd(count: int) -> np.ndarray:
    modulus, multiplier, block = 2147483647, 48271, 4096
    head = np.empty(block, dtype=np.int64)
    state = 1
    for index in range(block):
        state = state * multiplier % modulus
        head[index] = state
    jump = pow(multiplier, block, modulus)
    blocks = [head]
    while block * len(blocks) < count:
        blocks.append(blocks[-1] * jump % modulus)
    return np.concatenate(blocks)[:count]


def residual_noise(count: int) -> np.ndarray:
    global _RESIDUAL_NOISE
    if len(_RESIDUAL_NOISE) < count:
        total = max(count, 2 * len(_RESIDUAL_NOISE), 1 << 16)
        draws = (_minstd(2 * (total // 2 + total // 3 + 64)) - 1).astype(np.float32) * np.float32(2.0 ** -30) - np.float32(1.0)
        accepted, first, second = _polar(draws[0::2], draws[1::2], False)
        values = np.stack([first[accepted], second[accepted]], axis=1).reshape(-1)
        _RESIDUAL_NOISE = values[:total].astype(np.float32)
    return _RESIDUAL_NOISE[:count]
