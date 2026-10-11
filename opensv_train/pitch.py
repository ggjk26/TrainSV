from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .common import EngineGaussian, MeanScaleNormalizer, find_language, read_phone_set, sinusoid
from .dnni import DnniFile
from .duration import PhonemeDuration, Syllable
from .layers import MatrixParameter, VectorParameter, build_network

WRAPPER_TYPES = {
    0xCCE2240F4753C837, 0x0BC3C8E11C368859, 0xB7D76850E579425A, 0xA81B4E0A342431ED, 0xFD1C9123B5CCF269, 0x736A014C0A00406A,
    0x933CD542E8C3839B, 0xDB0BE7FE929C02E3, 0xBBCB3355F5331EC9, 0xFDF99C04DC496AC2, 0x8CD901ED074DA870, 0x3B69B825E5C1BB3C,
}
MODEL_TYPES = {
    0x5B8562D18C978B91, 0xE781BE473240733F, 0xA8AD7B2715FC5E90, 0xEF133A930D94F0FB, 0x0461232F81E7260F, 0x6715E41310BB7CE0,
    0xBCD2DF869CFD154D, 0xA7DC6559D4DCF415, 0xC310C561C14F056F, 0xC58A4D93DD623A98, 0xE32797F2732F379A, 0x1BD8F79C0EACA55E,
}
PITCH_CONTEXT_SECONDS = 0.5
SMOOTHING = (0.95, 0.050000011920928955)


@dataclass
class PitchNote:
    syllable: Syllable
    silence: bool = False
    breath: bool = False
    continuation: bool = False
    rap: bool = False
    tone: int = 0
    vibrato: float = 1.0


@dataclass
class PitchInputs:
    note_features: torch.Tensor
    note_frames: np.ndarray
    phoneme_frames: np.ndarray
    categories: np.ndarray
    languages: np.ndarray
    vowels: np.ndarray
    frames: int
    transpose: int
    notes: list[PitchNote] = field(default_factory=list)
    noise: torch.Tensor | None = None
    controls: torch.Tensor | None = None
    vibrato: torch.Tensor | None = None


def round_half_away(value: float) -> float:
    return math.copysign(math.floor(abs(float(value)) + 0.5), value)


def _phoneme_seed(phonemes: list[PhonemeDuration], index: int, global_seed: int) -> int:
    text = "magic" + (phonemes[index - 1].symbol if index > 0 else "") + "_" + phonemes[index].symbol + "_"
    text += (phonemes[index + 1].symbol if index + 1 < len(phonemes) else "") + "_" + str(global_seed)

    def fnv(value: str) -> int:
        result = 0x811C9DC5
        for byte in value.encode("utf-8"):
            signed = byte - 256 if byte >= 128 else byte
            result = ((result ^ (signed & 0xFFFFFFFF)) * 0x1000193) & 0xFFFFFFFF
        return result

    seed = fnv(text)
    return seed if seed else fnv(" ")


class PitchModel(nn.Module):
    def __init__(self, model: DnniFile, speaker: np.ndarray | None = None):
        super().__init__()
        wrapper = model.root
        if wrapper.type_id not in WRAPPER_TYPES or len(wrapper.children) != 1 or len(wrapper.payload) != 16:
            raise ValueError("unsupported F0 model wrapper")
        self.expression, self.expression_strength, self.rap_expression = struct.unpack_from("<3f", wrapper.payload)
        body = wrapper.children[0]
        if body.type_id not in MODEL_TYPES or len(body.children) != 5:
            raise ValueError("unsupported F0 model layout")
        frontend, speaker_node, enhancement, context, decoder = body.children
        version, interval = struct.unpack("<If", frontend.payload)
        if version != 1 or interval != np.float32(0.005):
            raise ValueError("unsupported pitch frontend")
        self.frame_seconds = interval
        self.phone_sets = [read_phone_set(child) for child in frontend.children[0].children]
        self.unified = read_phone_set(frontend.children[1])
        self.note_normalizer = MeanScaleNormalizer(frontend.children[2])

        self.speaker = VectorParameter(speaker_node) if speaker is None else nn.Parameter(torch.from_numpy(np.asarray(speaker, np.float32)))
        self.enhancement = build_network(enhancement)
        if len(context.children) != 7:
            raise ValueError("unsupported pitch context")
        self.category_embedding = MatrixParameter(context.children[0].children[0])
        self.language_embedding = MatrixParameter(context.children[1].children[0])
        self.note_encoder = build_network(context.children[2])
        self.phoneme_encoder = build_network(context.children[3])
        self.projection = build_network(context.children[4])
        self.feed_forward = build_network(context.children[5])
        self.residual = build_network(context.children[6])
        self.embedding_channels, self.embedding_scale, self.frame_stride = struct.unpack("<ifi", decoder.payload)
        if len(decoder.children) != 9:
            raise ValueError("expected the nine-network pitch decoder")
        self.decoder = nn.ModuleList(build_network(child) for child in decoder.children)

    def speaker_vector(self) -> torch.Tensor:
        return self.speaker if isinstance(self.speaker, torch.Tensor) else self.speaker()

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return self.note_normalizer.mean[2] + self.note_normalizer.scale[2] * normalized

    def normalize_pitch(self, midi: torch.Tensor) -> torch.Tensor:
        return (midi - self.note_normalizer.mean[2]) / self.note_normalizer.scale[2]

    def prepare(self, notes: list[PitchNote], phonemes: list[PhonemeDuration], rap_languages: list[str] | None = None, seed: int = 0, vibrato_envelope: np.ndarray | None = None) -> PitchInputs:
        if rap_languages is not None:
            notes = [PitchNote(**{**note.__dict__, "rap": note.rap and note.syllable.language in rap_languages}) for note in notes]
        pitches = [note.syllable.pitch for note in notes]
        transpose = 79 - max(pitches) if max(pitches) > 79 else 0
        if min(pitches) < 36:
            transpose = 36 - min(pitches)
        notes = [PitchNote(**{**note.__dict__, "syllable": Syllable(**{**note.syllable.__dict__, "pitch": note.syllable.pitch + transpose})}) for note in notes]

        interval = np.float32(self.frame_seconds)
        inverse = np.float32(1.0) / interval
        note_frames, features = [], np.zeros((len(notes), 10), dtype=np.float32)
        note_seconds = np.float32(0.0)
        for index, note in enumerate(notes):
            seconds = note.syllable.seconds
            end = float(note_seconds) + seconds
            start_frame = round_half_away(np.float32(note_seconds * inverse))
            end_frame = round_half_away(end * (1.0 / float(interval)))
            note_frames.append(int(end_frame - start_frame))
            note_seconds = np.float32(end)
            features[index, :5] = (float(note.silence or note.breath), float(note.continuation), note.syllable.pitch, np.log(min(seconds + 0.001, 5.0)), float(note.rap))
        phoneme_frames, categories, languages, vowels = [], [], [], []
        phoneme_seconds = np.float32(0.0)
        for phoneme in phonemes:
            end = np.float32(phoneme_seconds + np.float32(phoneme.seconds))
            phoneme_frames.append(int(round_half_away(np.float32(end / interval)) - round_half_away(np.float32(phoneme_seconds / interval))))
            phoneme_seconds = end
            language = find_language(self.phone_sets, phoneme.language)
            phone_set = self.phone_sets[language]
            source = phone_set.symbols.index(phoneme.symbol)
            unified_category = self.unified.categories[self.unified.symbols.index(phone_set.unified[source])]
            categories.append(self.unified.classes.index(unified_category))
            languages.append(language)
            vowels.append(phone_set.categories[source] in ("vowel", "diphthong"))
        note_start = phoneme_start = phoneme_index = 0
        for index, note in enumerate(notes):
            language_frame = note_start + note_frames[index] // 3
            while phoneme_index + 1 < len(phonemes) and phoneme_start + phoneme_frames[phoneme_index] < language_frame:
                phoneme_start += phoneme_frames[phoneme_index]
                phoneme_index += 1
            if note.rap and note.tone > 0 and "mandarin" in self.phone_sets[languages[phoneme_index]].name:
                features[index, note.tone + 4] = 1.0
            note_start += note_frames[index]
        frames = max(sum(note_frames), sum(phoneme_frames))
        inputs = PitchInputs(
            self.note_normalizer.normalize(torch.from_numpy(features).to(self.note_normalizer.mean.device)), np.array(note_frames), np.array(phoneme_frames),
            np.array(categories), np.array(languages), np.array(vowels), frames, transpose, notes,
        )
        self._controls(inputs, phonemes, seed, vibrato_envelope)
        return inputs

    def _controls(self, inputs: PitchInputs, phonemes: list[PhonemeDuration], seed: int, envelope: np.ndarray | None) -> None:
        frames = inputs.frames
        expressive = np.full(frames, np.float32(self.expression), dtype=np.float32)
        vibrato = np.zeros(frames, dtype=np.float32)
        frame = 0
        for note, count in zip(inputs.notes, inputs.note_frames):
            expressive[frame:frame + count] = np.float32(self.expression) * np.float32(self.rap_expression) if note.rap else np.float32(self.expression)
            scale = np.ones(count, dtype=np.float32) if envelope is None else envelope[frame:frame + count].astype(np.float32)
            vibrato[frame:frame + count] = np.clip(np.float32(note.vibrato) * scale, 0.0, 2.0) - np.float32(1.0)
            frame += count
        keep, mix = np.float32(SMOOTHING[0]), np.float32(SMOOTHING[1])
        for index in range(1, frames):
            expressive[index] = expressive[index] * keep + expressive[index - 1] * mix
        for index in range(frames - 2, -1, -1):
            expressive[index] = expressive[index] * keep + expressive[index + 1] * mix
        strength = np.full(frames, np.float32(self.expression_strength), dtype=np.float32)
        high = expressive > np.float32(0.8)
        strength[high] = (float(self.expression_strength) * np.maximum(0.0, 3.0 - 2.5 * expressive[high].astype(np.float64))).astype(np.float32)

        reseeds = []
        phoneme_boundary = note_boundary = 0
        next_phoneme = next_note = 0
        active_seed = 0
        for frame in range(frames):
            reseed = False
            if next_phoneme < len(phonemes) and frame >= phoneme_boundary:
                active_seed = _phoneme_seed(phonemes, next_phoneme, seed)
                phoneme_boundary += inputs.phoneme_frames[next_phoneme]
                next_phoneme += 1
                reseed = True
            if next_note < len(inputs.notes) and frame >= note_boundary:
                note_boundary += inputs.note_frames[next_note]
                next_note += 1
                reseed = True
            if reseed:
                reseeds.append((frame, active_seed))
        source = EngineGaussian(123)
        gaussian = np.zeros(frames, dtype=np.float32)
        boundaries = [frame for frame, _ in reseeds] + [frames]
        if boundaries[0] > 0:
            gaussian[:boundaries[0]] = source.take(boundaries[0])
        for (frame, active), end in zip(reseeds, boundaries[1:]):
            source.reseed(active)
            gaussian[frame:end] = source.take(end - frame)
        noise = gaussian * np.minimum(np.float32(1.0), expressive * np.float32(2.0))
        controls = np.stack([np.full(frames, np.float32(0.995)), strength, np.zeros(frames, dtype=np.float32)], axis=1)
        inputs.noise = torch.from_numpy(noise)
        inputs.controls = torch.from_numpy(controls.astype(np.float32))
        inputs.vibrato = torch.from_numpy(vibrato)

    def context(self, inputs: PitchInputs) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.speaker_vector().device
        speaker = self.enhancement(torch.ones(1, 1, device=device))[0] + self.speaker_vector()
        note_context = self.note_encoder(inputs.note_features.to(device))
        phone_context = self.phoneme_encoder(self.category_embedding()[:, torch.from_numpy(inputs.categories).to(device)].t())
        frames = inputs.frames
        note_frames = int(inputs.note_frames.sum())
        phone_frames = int(inputs.phoneme_frames.sum())
        note_part = torch.zeros(frames, 128, device=device)
        note_part[:note_frames] = note_context[torch.from_numpy(np.repeat(np.arange(len(inputs.note_frames)), inputs.note_frames)).to(device)]
        phone_of_frame = np.repeat(np.arange(len(inputs.phoneme_frames)), inputs.phoneme_frames)
        phone_part = torch.zeros(frames, 128, device=device)
        phone_part[:phone_frames] = phone_context[torch.from_numpy(phone_of_frame).to(device)]
        counts = inputs.phoneme_frames
        within = np.concatenate([np.arange(count) for count in counts]).astype(np.float32) if phone_frames else np.zeros(0, np.float32)
        span = np.repeat(counts - 1, counts).astype(np.float32)
        position = np.where(span == 0, np.float32(0.5), within / np.maximum(span, 1)).astype(np.float32)
        phases = (np.float32(1.0) - np.arange(8, dtype=np.float32)) * np.float32(0.125)
        angles = (position[:, None] + phases[None, :]) * np.float32(6.2831854820251465)
        cosines = np.zeros((frames, 8), dtype=np.float32)
        cosines[:phone_frames] = np.cos(angles) * np.float32(0.5) + np.float32(0.5)
        voiced_phones = np.flatnonzero(counts > 0)
        language_of_frame = np.full(frames, inputs.languages[voiced_phones[-1]] if len(voiced_phones) else 0)
        language_of_frame[:phone_frames] = inputs.languages[phone_of_frame]
        context = torch.cat([
            note_part, phone_part, torch.from_numpy(cosines).to(device), speaker.expand(frames, -1),
            self.language_embedding()[:, torch.from_numpy(language_of_frame).to(device)].t(),
        ], dim=1)
        projected = self.projection(context)
        return self.feed_forward(projected), self.residual(projected)

    def decode(self, inputs: PitchInputs, feed_forward: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        device = residual.device
        frames, stride = inputs.frames, self.frame_stride
        groups = -(-frames // stride)
        index = torch.from_numpy(np.minimum(np.arange(groups * stride), frames - 1).reshape(groups, stride)).to(device)
        scalar = self.decoder[0](feed_forward)[:, 0]
        pooled = residual[index].sum(dim=1) / float(stride)
        control = inputs.controls.to(device)[index].sum(dim=1) * np.float32(self.embedding_scale) / np.float32(stride)
        embeddings = [sinusoid(control[:, column], self.embedding_channels) for column in range(3)]
        noise = inputs.noise.to(device)[index]
        condition_tail = torch.cat([*embeddings, noise], dim=1)
        coarse = self.decoder[5](scalar[index], torch.cat([pooled, condition_tail], dim=1))
        vibrato = torch.clamp(inputs.vibrato.to(device)[index].sum(dim=1, keepdim=True) / float(stride), -0.999, 0.999)
        correction = self.decoder[4](torch.cat([pooled, self.decoder[1](vibrato)], dim=1))
        refined = self.decoder[6](coarse, torch.cat([pooled + correction, condition_tail], dim=1))
        repeated = refined[torch.from_numpy(np.arange(frames) // stride).to(device)]
        return self.decoder[8](repeated)[:, 0]

    def forward(self, inputs: PitchInputs) -> torch.Tensor:
        feed_forward, residual = self.context(inputs)
        return smooth(self.decode(inputs, feed_forward, residual))

    def predict(self, notes: list[PitchNote], phonemes: list[PhonemeDuration], rap_languages: list[str] | None = None, seed: int = 0, vibrato_envelope: np.ndarray | None = None) -> np.ndarray:
        inputs = self.prepare(notes, phonemes, rap_languages, seed, vibrato_envelope)
        with torch.no_grad():
            midi = self.denormalize(self(inputs)).cpu().numpy().astype(np.float32)
        postprocess(inputs.notes, float(self.frame_seconds), midi)
        return midi - np.float32(inputs.transpose)


def smooth(normalized: torch.Tensor) -> torch.Tensor:
    previous = torch.cat([normalized.new_zeros(2), normalized[1:-1]])
    current = torch.cat([normalized.new_zeros(1), normalized[1:]])
    following = torch.cat([normalized[1:], normalized.new_zeros(1)])
    return previous * 0.25 + current * 0.5 + following * 0.25


def postprocess(notes: list[PitchNote], frame_seconds: float, midi: np.ndarray) -> None:
    inverse = 1.0 / float(np.float32(frame_seconds))
    seconds = 0.0
    for note in notes:
        duration = note.syllable.seconds
        end_seconds = seconds + duration
        start, end = int(seconds * inverse), int(end_seconds * inverse)
        seconds = end_seconds
        if note.silence:
            continue
        middle_start = start + (end - start) // 4
        middle_end = min(start + (end - start) * 3 // 4, len(midi))
        if middle_start >= middle_end:
            continue
        frames = np.arange(middle_start, middle_end)
        pitch = midi[frames].astype(np.float64)
        previous = midi[np.where(frames < 2, frames, frames - 1)].astype(np.float64)
        following = midi[np.minimum(frames + 1, len(midi) - 1)].astype(np.float64)
        weights = 1.0 / (np.abs(previous - pitch) + 0.1 + np.abs(following - pitch))
        offset = note.syllable.pitch - float(np.sum(weights * pitch) / np.sum(weights))
        if abs(offset) <= float(np.float32(0.2)):
            continue
        correction = offset * min(max((duration - float(np.float32(0.3))) * 10.0, 0.0), 1.0)
        span = np.arange(start, min(end, len(midi)))
        left = np.minimum((span - start) * 0.1, 1.0)
        right = np.minimum((end - span) * 0.1, 1.0)
        midi[span] = (midi[span].astype(np.float64) + left * correction * right).astype(np.float32)


def pad_pitch_notes(notes: list[PitchNote]) -> list[PitchNote]:
    first, last = notes[0].syllable, notes[-1].syllable
    leading = PitchNote(Syllable(first.language, ["sil"], PITCH_CONTEXT_SECONDS, first.pitch), silence=True)
    trailing = PitchNote(Syllable(last.language, ["sil"], PITCH_CONTEXT_SECONDS, last.pitch), silence=True)
    return [leading, *notes, trailing]


def midi_from_log_f0(log_f0):
    return 69.0 + 12.0 * (np.asarray(log_f0, dtype=np.float64) - math.log(440.0)) / math.log(2.0)
