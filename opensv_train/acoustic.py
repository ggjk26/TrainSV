from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .common import EngineGaussian, MinMaxNormalizer, find_language, read_phone_set
from .dnni import DnniFile
from .layers import MatrixParameter, VectorParameter, build_network

FRAME_SECONDS = 0.005
DIFFUSION_TEMPERATURE = 0.995
SOURCE_TEMPERATURE = 0.5
GUIDANCE = 1.0
PHASES = (1.5707963705062866, 0.0, -1.5707963705062866, -3.1415927410125732)


@dataclass
class TimedPhoneme:
    language: str
    symbol: str
    frames: int


@dataclass
class AcousticOutput:

    spectrum: torch.Tensor
    auxiliary: torch.Tensor
    normalized: torch.Tensor
    frames: torch.Tensor


class AcousticModel(nn.Module):
    def __init__(self, model: DnniFile, vocal_modes: list[str] | None = None, speaker: np.ndarray | None = None, default_style: np.ndarray | None = None, mode_styles: list[np.ndarray] | None = None):
        super().__init__()
        root = model.root
        if root.type != "_rldtg0" or len(root.children) != 2:
            raise ValueError("expected an _rldtg0 acoustic root")
        body, styles = root.children
        if body.type != "_rldms0" or len(body.children) != 8 or styles.type != "_stbkv1":
            raise ValueError("unsupported acoustic model layout")
        features, speaker_node, context, latent = body.children[:4]
        if features.type not in ("_ftmfv2", "_ftmfv3") or context.type != "_vqctx1" or latent.type != "_didsv0":
            raise ValueError("unsupported acoustic feature, context or latent layout")
        phoneme_channels, pitch_channels, acoustic_channels, low, high, dct = struct.unpack_from("<6I", features.payload)
        (frame_seconds,) = struct.unpack_from("<f", features.payload, 24)
        if (pitch_channels, acoustic_channels, low, high, dct) != (1, 71, 2, 66, 0) or frame_seconds != np.float32(FRAME_SECONDS):
            raise ValueError("unsupported acoustic feature configuration")
        if struct.unpack("<IIf", latent.payload) != (4, 64, 64.0):
            raise ValueError("unsupported latent configuration")

        self.phone_sets = [read_phone_set(child) for child in features.children[0].children]
        self.unified = read_phone_set(features.children[1])
        if len(self.unified.symbols) + 1 != phoneme_channels:
            raise ValueError("unified phone set does not match the phoneme feature width")
        self.phoneme_normalizer = MinMaxNormalizer(features.children[3])
        self.pitch_normalizer = MinMaxNormalizer(features.children[4])
        self.acoustic_normalizer = MinMaxNormalizer(features.children[5])

        self.phoneme_encoder = build_network(context.children[0])
        self.context_projection = build_network(context.children[1])
        self.context_network = build_network(context.children[2])
        self.language_embedding = MatrixParameter(context.children[3].children[0])
        self.latent_projection = build_network(latent.children[0])
        self.latent_network = build_network(latent.children[1])
        self.latent_head = build_network(latent.children[2])
        self.spectral_network = build_network(body.children[4])
        self.spectral_head = build_network(body.children[5])
        self.auxiliary_network = build_network(body.children[6])
        self.auxiliary_head = build_network(body.children[7])

        style_nodes = styles.children[0].children
        self.info_voice = speaker is not None
        if self.info_voice:
            self.speaker = nn.Parameter(torch.from_numpy(np.asarray(speaker, np.float32)))
            self.default_style = nn.Parameter(torch.from_numpy(np.asarray(default_style, np.float32)))
            self.mode_styles = nn.ParameterList(nn.Parameter(torch.from_numpy(np.asarray(style, np.float32))) for style in mode_styles or [])
            # Alternate Acoustic bases can use different per-voice embedding widths.
            # Detect incompatible info.nofs vectors before the first training step.
            expected_speaker = VectorParameter(speaker_node).value.numel()
            expected_style = VectorParameter(style_nodes[0]).value.numel()
            if self.speaker.numel() != expected_speaker:
                raise ValueError(
                    f"selected Acoustic base expects a {expected_speaker}-value speaker embedding, "
                    f"but info.nofs provides {self.speaker.numel()}; choose a structurally compatible Acoustic .dnni"
                )
            if self.default_style.numel() != expected_style:
                raise ValueError(
                    f"selected Acoustic base expects a {expected_style}-value style embedding, "
                    f"but info.nofs provides {self.default_style.numel()}; choose a structurally compatible Acoustic .dnni"
                )
            for index, style in enumerate(self.mode_styles):
                if style.numel() != expected_style:
                    raise ValueError(
                        f"vocal-mode embedding {index} has {style.numel()} values, but the selected Acoustic base "
                        f"expects {expected_style}; choose a structurally compatible Acoustic .dnni"
                    )
        else
            vocal_modes = vocal_modes or []
            if vocal_modes and len(style_nodes) - 1 != len(vocal_modes):
                raise ValueError("timbre-style metadata does not match the acoustic style vectors")
            self.speaker = VectorParameter(speaker_node)
            self.default_style = VectorParameter(style_nodes[0])
            self.mode_styles = nn.ModuleList(VectorParameter(node) for node in style_nodes[1:1 + len(vocal_modes)])
        self.vocal_modes = list(vocal_modes or [])

        frequencies = np.exp2(np.arange(32, dtype=np.float32) * np.float32(-26.575424194335938 / 64.0))
        controls = np.concatenate([
            np.sin(np.float32(64.0 * DIFFUSION_TEMPERATURE) * frequencies), np.cos(np.float32(64.0 * DIFFUSION_TEMPERATURE) * frequencies),
            np.sin(np.float32(64.0 * GUIDANCE) * frequencies), np.cos(np.float32(64.0 * GUIDANCE) * frequencies),
        ]).astype(np.float32)
        self.register_buffer("controls", torch.from_numpy(controls))

    def _vector(self, value) -> torch.Tensor:
        return value if isinstance(value, torch.Tensor) else value()

    def speaker_vector(self) -> torch.Tensor:
        return self._vector(self.speaker)

    def default_style_vector(self) -> torch.Tensor:
        return self._vector(self.default_style)

    def mode_style_vectors(self) -> list[torch.Tensor]:
        return [self._vector(style) for style in self.mode_styles]

    def phoneme_indices(self, phonemes: list[TimedPhoneme]) -> tuple[np.ndarray, np.ndarray]:
        unified, languages = [], []
        for phoneme in phonemes:
            language = find_language(self.phone_sets, phoneme.language)
            phone_set = self.phone_sets[language]
            unified.append(self.unified.symbols.index(phone_set.unified[phone_set.symbols.index(phoneme.symbol)]))
            languages.append(language)
        return np.array(unified), np.array(languages)

    def frame_features(self, phonemes: list[TimedPhoneme], log_f0: torch.Tensor, mode_weights: torch.Tensor | None = None) -> torch.Tensor:
        device = self.controls.device
        counts = np.array([phoneme.frames for phoneme in phonemes])
        if counts.min() < 1 or counts.sum() != len(log_f0):
            raise ValueError("one log-F0 value is required for each phoneme frame")
        unified, languages = self.phoneme_indices(phonemes)
        raw = np.zeros((len(phonemes), len(self.unified.symbols) + 1), dtype=np.float32)
        raw[np.arange(len(phonemes)), unified] = 1.0
        raw[:, -1] = np.log((counts * np.float32(FRAME_SECONDS)).astype(np.float32) + np.float32(0.05))
        encoded = self.phoneme_encoder(self.phoneme_normalizer.normalize(torch.from_numpy(raw).to(device)))

        phone_of_frame = torch.from_numpy(np.repeat(np.arange(len(phonemes)), counts)).to(device)
        within = np.concatenate([np.arange(count) for count in counts]).astype(np.float32)
        span = np.repeat(counts - 1, counts).astype(np.float32)
        position = np.where(span == 0, np.float32(0.5), within / np.maximum(span, 1)).astype(np.float32)
        pitch = self.pitch_normalizer.normalize(log_f0.to(device).unsqueeze(1)).squeeze(1)
        phases = torch.tensor(PHASES, dtype=torch.float32, device=device)
        scalars = torch.stack([torch.from_numpy(position).to(device), pitch], dim=1)
        cosines = 0.5 * torch.cos(scalars.unsqueeze(2) * np.float32(6.2831854820251465) + phases) + 0.5

        frames = len(log_f0)
        style = self.default_style_vector().expand(frames, -1)
        modes = self.mode_style_vectors()
        if mode_weights is not None and modes:
            deltas = torch.stack([mode - self.default_style_vector() for mode in modes])
            style = style + mode_weights.to(device) @ deltas
        return torch.cat([
            encoded[phone_of_frame],
            cosines.reshape(frames, 8),
            self.speaker_vector().expand(frames, -1),
            self.language_embedding()[:, torch.from_numpy(languages).to(device)[phone_of_frame]].t(),
            style,
        ], dim=1)

    def latent_noise(self, frames: int, seed: int | None) -> torch.Tensor:
        if seed is None:
            noise = torch.randn(frames, 4)
        else:
            noise = torch.from_numpy(EngineGaussian(seed).take(frames * 4).reshape(frames, 4))
        return (noise * SOURCE_TEMPERATURE).to(self.controls.device)

    def forward(self, phonemes: list[TimedPhoneme], log_f0: torch.Tensor, seed: int | None = 5489, mode_weights: torch.Tensor | None = None, noise: torch.Tensor | None = None) -> AcousticOutput:
        features = self.frame_features(phonemes, log_f0, mode_weights)
        frames = features.shape[0]
        context = self.context_network(features)
        projected = self.context_projection(features)
        if noise is None:
            noise = self.latent_noise(frames, seed)
        condition = torch.cat([context, self.controls.expand(frames, -1), noise], dim=1)
        latent = self.latent_head(self.latent_network(self.latent_projection(projected), condition))
        spectrum = self.spectral_head(self.spectral_network(context, latent))
        auxiliary = self.auxiliary_head(self.auxiliary_network(context, spectrum))
        normalized = torch.cat([auxiliary[:, :2], spectrum, auxiliary[:, 2:]], dim=1)
        return AcousticOutput(spectrum, auxiliary, normalized, self.decode(normalized, log_f0.to(normalized.device)))

    def decode(self, normalized: torch.Tensor, log_f0: torch.Tensor, hard_voicing: bool = True) -> torch.Tensor:
        decoded = self.acoustic_normalizer.denormalize(normalized)
        voicing = (decoded[:, :1] > 0.5).to(decoded.dtype) if hard_voicing else decoded[:, :1]
        return torch.cat([
            voicing,
            torch.exp(log_f0).unsqueeze(1),
            torch.clamp(decoded[:, 1:2], 0.01, 3.0),
            decoded[:, 2:66],
            torch.clamp(decoded[:, 66:71], 0.0, 1.0),
        ], dim=1)


def midi_to_log_f0(midi):
    return math.log(440.0) + (np.asarray(midi, dtype=np.float64) - 69.0) * math.log(2.0) / 12.0
