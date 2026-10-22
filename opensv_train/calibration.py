from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
import torch

from .data import Phrase
from .layers import Conv, Dense
from .voice import Voice


@dataclass
class RangeReport:
    role: str
    node_type: str
    shift: int
    peak: float

    @property
    def limit(self) -> float:
        return 2.0 ** (15 - self.shift)


def _nodes_of(layer):
    return [layer.matrix.node] if isinstance(layer, Dense) else [tap.node for tap in layer.taps]


def measure_input_peaks(voice: Voice, phrases: list[Phrase], device: torch.device) -> dict[int, tuple[str, object, float]]:
    peaks: dict[int, list] = {}
    handles = []
    for role, model in voice.models.items():
        for layer in model.modules():
            if isinstance(layer, (Dense, Conv)):
                nodes = [node for node in _nodes_of(layer) if node.type in ("prim4", "prim5")]
                if not nodes:
                    continue

                def hook(module, arguments, output, nodes=nodes, role=role):
                    peak = float(arguments[0].detach().abs().max())
                    for node in nodes:
                        entry = peaks.setdefault(id(node), [role, node, 0.0])
                        entry[2] = max(entry[2], peak)

                handles.append(layer.register_forward_hook(hook))
    try:
        with torch.no_grad():
            for phrase in phrases:
                hz, _ = phrase.frame_f0()
                frames = voice.acoustic(phrase.timed_phonemes(), torch.from_numpy(np.log(hz).astype(np.float32)).to(device)).frames
                # Preservation mode instantiates only the acoustic model. Calibrate
                # its own input ranges without forcing inference through untouched
                # duration, Pitch or Vocoder networks.
                if hasattr(voice, "vocoder"):
                    voice.vocoder(frames)
                if phrase.syllables:
                    if hasattr(voice, "duration"):
                        voice.duration(voice.duration.prepare(phrase.syllables))
                    if hasattr(voice, "pitch"):
                        notes, phonemes, _, _ = phrase.pitch_example()
                        voice.pitch(voice.pitch.prepare(notes, phonemes))
    finally:
        for handle in handles:
            handle.remove()
    return {key: tuple(value) for key, value in peaks.items()}


def fit_input_ranges(voice: Voice, phrases: list[Phrase], device: torch.device, headroom: float = 1.25) -> list[RangeReport]:
    adjusted = []
    for role, node, peak in measure_input_peaks(voice, phrases, device).values():
        bits, shift = struct.unpack_from("<II", node.payload, 0)
        if peak <= 2.0 ** (15 - shift) * (1 + 1e-4):
            continue
        fitted = max(0, min(shift, 15 - math.ceil(math.log2(peak * headroom))))
        node.payload = node.payload[:4] + struct.pack("<I", fitted) + node.payload[8:]
        adjusted.append(RangeReport(role, node.type, fitted, peak))
    return adjusted
