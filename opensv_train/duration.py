from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .common import MinMaxNormalizer, find_language, read_phone_set, sinusoid
from .dnni import DnniFile
from .layers import MatrixParameter, VectorParameter, build_network

DURATION_TYPES = {
    0x4C30EAFF390F599A, 0x210AE732BCDA3950, 0xEDB2643EC31049D7, 0x76B77E5BF8D6500C, 0x8E0F0182B380A860, 0xD4C2D7946F91E167,
    0x135EE08897501166, 0x71A4B344DF3F7ECE, 0xA4F623B00C5A97C0, 0x1FAD6BD2DFA1A17F, 0x5D5D600F0E5453E9, 0x1DB558504D28A4AD,
}
FRONTEND_TYPES = {
    0xFF369E74E595E206, 0xEA62CA7C31E37CA0, 0x319B1E6E37917BD3, 0x05909FEBA966BB54, 0x18B5591E22ABA550, 0x825A31102572A303,
    0xF0C8BDD6E4FE863A, 0x77F5F7F551D96E82, 0xD9177B5063843FB0, 0xA097F32E2E5FE07B, 0xC6EA192CF9C095D1, 0x32870689962BFF1D,
}


@dataclass
class Syllable:
    language: str
    phonemes: list[str]
    seconds: float
    pitch: int
    continuation: bool = False


@dataclass
class PhonemeDuration:
    language: str
    symbol: str
    syllable: int
    interval: int
    seconds: float


@dataclass
class DurationInputs:

    unified: torch.Tensor
    language: torch.Tensor
    position_from_start: torch.Tensor
    position_from_end: torch.Tensor
    log_interval: torch.Tensor
    pitch: torch.Tensor
    interval_of_phoneme: torch.Tensor
    intervals: list[tuple[int, int, float]]
    phonemes: list[PhonemeDuration] = field(default_factory=list)


class DurationModel(nn.Module):
    def __init__(self, model: DnniFile, voice_embedding: np.ndarray | None = None):
        super().__init__()
        root = model.root
        if root.type_id not in DURATION_TYPES or len(root.children) != 6 or struct.unpack("<II", root.payload) != (32, 16):
            raise ValueError("not a supported gen2a duration model")
        frontend = root.children[0]
        if frontend.type_id not in FRONTEND_TYPES or len(frontend.children) != 4:
            raise ValueError("unsupported duration frontend")
        self.minimum_pitch, self.maximum_pitch = struct.unpack("<ii", frontend.payload)
        self.phone_sets = [read_phone_set(child) for child in frontend.children[0].children]
        self.unified = read_phone_set(frontend.children[1])
        self.input_normalizer = MinMaxNormalizer(frontend.children[2])
        self.output_normalizer = MinMaxNormalizer(frontend.children[3])
        self.network = build_network(root.children[1])
        self.language_embedding = MatrixParameter(root.children[2].children[0])
        self.phoneme_embedding = MatrixParameter(root.children[3].children[0])
        self.position_embedding = MatrixParameter(root.children[4].children[0])
        if voice_embedding is None:
            self.voice = VectorParameter(root.children[5])
        else:
            self.voice = nn.Parameter(torch.from_numpy(np.asarray(voice_embedding, dtype=np.float32)))

    def voice_vector(self) -> torch.Tensor:
        return self.voice() if isinstance(self.voice, VectorParameter) else self.voice

    def prepare(self, syllables: list[Syllable]) -> DurationInputs:
        intervals = []
        for index, syllable in enumerate(syllables):
            language = find_language(self.phone_sets, syllable.language)
            phone_set = self.phone_sets[language]
            phones = []
            for position, symbol in enumerate(syllable.phonemes):
                source = phone_set.symbols.index(symbol)
                phones.append({
                    "syllable": index, "position": position, "language": language,
                    "unified": self.unified.symbols.index(phone_set.unified[source]),
                    "nucleus": phone_set.categories[source] in ("vowel", "diphthong"),
                })
            intervals.append({"phones": phones, "seconds": syllable.seconds, "pitch": syllable.pitch})
        intervals = _align_intervals(intervals)

        rows = []
        phonemes = []
        spans = []
        for interval_index, interval in enumerate(intervals):
            count = len(interval["phones"])
            spans.append((len(rows), len(rows) + count, interval["seconds"]))
            for position, phone in enumerate(interval["phones"]):
                rows.append((phone["unified"], phone["language"], min(position, 7), min(count - position - 1, 7), interval_index))
                syllable = syllables[phone["syllable"]]
                phonemes.append(PhonemeDuration(syllable.language, syllable.phonemes[phone["position"]], phone["syllable"], interval_index, 0.0))
        columns = np.array(rows, dtype=np.int64).T
        log_interval = np.array([np.float32(math.log(interval["seconds"] + float(np.float32(0.01)))) for interval in intervals], dtype=np.float32)
        pitch = np.array([np.float32(interval["pitch"] - self.minimum_pitch) / np.float32(self.maximum_pitch - self.minimum_pitch) for interval in intervals], dtype=np.float32)
        return DurationInputs(
            *(torch.from_numpy(column) for column in columns[:4]),
            torch.from_numpy(log_interval), torch.from_numpy(pitch), torch.from_numpy(columns[4]), spans, phonemes,
        )

    def forward(self, inputs: DurationInputs) -> torch.Tensor:
        device = self.voice_vector().device
        interval = inputs.interval_of_phoneme.to(device)
        normalized = self.input_normalizer.normalize(inputs.log_interval.to(device).unsqueeze(1)).squeeze(1)
        features = torch.cat([
            self.phoneme_embedding()[:, inputs.unified.to(device)].t(),
            self.language_embedding()[:, inputs.language.to(device)].t(),
            self.position_embedding()[:, inputs.position_from_start.to(device)].t(),
            self.position_embedding()[:, inputs.position_from_end.to(device)].t(),
            sinusoid(normalized[interval], 32),
            sinusoid(inputs.pitch.to(device)[interval], 16),
            self.voice_vector().expand(len(interval), -1),
        ], dim=1)
        return self.output_normalizer.denormalize(self.network(features)).squeeze(1)

    def predict(self, syllables: list[Syllable]) -> list[PhonemeDuration]:
        inputs = self.prepare(syllables)
        with torch.no_grad():
            weights = torch.exp(self(inputs)).cpu().numpy().astype(np.float32)
        for first, end, seconds in inputs.intervals:
            total = np.float32(weights[first:end].sum(dtype=np.float32))
            for index in range(first, end):
                fraction = weights[index] * (np.float32(1.0) / total)
                inputs.phonemes[index].seconds = float(np.float32(float(fraction) * seconds))
        return inputs.phonemes


def interval_log_fractions(log_weights: torch.Tensor, inputs: DurationInputs) -> torch.Tensor:
    index = inputs.interval_of_phoneme.to(log_weights.device)
    maxima = torch.full((len(inputs.intervals),), -torch.inf, device=log_weights.device).scatter_reduce(0, index, log_weights, "amax")
    sums = torch.zeros(len(inputs.intervals), device=log_weights.device).index_add(0, index, torch.exp(log_weights - maxima[index]))
    return log_weights - maxima[index] - torch.log(sums[index])


def _align_intervals(intervals: list[dict]) -> list[dict]:
    for index in range(1, len(intervals)):
        phones = intervals[index]["phones"]
        nucleus = next((position for position, phone in enumerate(phones) if phone["nucleus"]), None)
        if nucleus is not None:
            intervals[index - 1]["phones"].extend(phones[:nucleus])
            del phones[:nucleus]
    result = []
    for interval in intervals:
        phones = interval["phones"]
        nuclei = sum(phone["nucleus"] for phone in phones)
        if nuclei <= 1:
            result.append(interval)
            continue
        seconds = interval["seconds"] / nuclei
        begin = 0
        found = False
        for position, phone in enumerate(phones):
            if phone["nucleus"]:
                if found:
                    result.append({"phones": phones[begin:position], "seconds": seconds, "pitch": interval["pitch"]})
                    begin = position
                found = True
        result.append({"phones": phones[begin:], "seconds": seconds, "pitch": interval["pitch"]})
    return result


def quantize_durations(durations: list[PhonemeDuration], frame_seconds: float) -> list[int]:
    frames_per_second = 1.0 / float(np.float32(frame_seconds))
    accumulated = 0.0
    previous = 0
    counts = []
    for duration in durations:
        accumulated += duration.seconds
        boundary = max(previous + 1, int(accumulated * frames_per_second))
        counts.append(boundary - previous)
        previous = boundary
    return counts
