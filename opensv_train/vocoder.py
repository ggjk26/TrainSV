from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .common import MinMaxNormalizer, residual_noise
from .dnni import DnniFile, Node, read_vector
from .layers import build_network

REFERENCE_PERIOD = 0.005


def _make_knot(index: int) -> tuple[float, ...]:
    rd = 0.01 * 600.0 ** (index / 255.0)
    return_ratio = 0.000001 if rd < 0.21 else (4.8 * rd - 1.0) / 100.0 if rd < 2.7 else 0.323 / rd
    if index < 209:
        rk = (22.4 + 11.8 * rd) / 100.0
        rg = 0.25 * rk / (0.11 * rd / (0.5 + 1.2 * rk) - return_ratio)
    else:
        open_quotient = 1.0 - 1.0 / (2.17 * rd)
        rg = 0.0093552 + 5.96 / (7.92 - 2.0 * open_quotient)
        rk = 2.0 * rg * open_quotient - 0.9572
    peak = 0.5 / rg
    closing = peak * (1.0 + rk)
    tail = 1.0 - closing
    lower, upper = 0.0, 1.0 / return_ratio
    for _ in range(80):
        decay = (lower + upper) * 0.5
        if -math.expm1(-tail * decay) / decay > return_ratio:
            lower = decay
        else:
            upper = decay
    decay = (lower + upper) * 0.5
    tail_area = (-math.expm1(-tail * decay) - tail * decay * math.exp(-tail * decay)) / (decay * decay * return_ratio)
    omega = math.pi / peak
    sine, cosine = math.sin(omega * closing), math.cos(omega * closing)

    def residual(growth):
        return tail_area * sine * (growth * growth + omega * omega) + growth * sine - omega * cosine + omega * math.exp(-growth * closing)

    lower, upper = 0.0, 1.0
    while residual(upper) > 0.0:
        upper *= 2.0
    for _ in range(80):
        growth = (lower + upper) * 0.5
        if residual(growth) > 0.0:
            lower = growth
        else:
            upper = growth
    return rd, closing, peak, return_ratio, math.log(decay / REFERENCE_PERIOD), math.log((lower + upper) * 0.5 / REFERENCE_PERIOD)


LF_KNOTS = np.array([_make_knot(index) for index in range(256)], dtype=np.float32)


def dsp_dtype(device: torch.device) -> torch.dtype:
    return torch.float64 if device.type == "cpu" else torch.float32


def lf_parameters(shape: torch.Tensor) -> dict[str, torch.Tensor]:
    dtype = dsp_dtype(shape.device)
    knots = torch.from_numpy(LF_KNOTS).to(device=shape.device, dtype=dtype)
    upper = torch.searchsorted(knots[:, 0].contiguous(), shape.detach().to(dtype).contiguous(), right=True)
    right = knots[torch.clamp(upper, max=255)]
    left = knots[torch.clamp(upper - 1, min=0)]
    shape = shape.to(dtype)
    width = right[:, 0] - left[:, 0]
    position = torch.where(width == 0, torch.zeros_like(shape), (shape - left[:, 0]) / torch.where(width == 0, torch.ones_like(width), width))
    values = left + (right - left) * position.unsqueeze(1)
    return {
        "closing": values[:, 1], "peak": values[:, 2], "return": values[:, 3],
        "decay": torch.exp(values[:, 4]) * REFERENCE_PERIOD, "growth": torch.exp(values[:, 5]) * REFERENCE_PERIOD,
    }


def lf_spectrum(parameters: dict[str, torch.Tensor], harmonics: torch.Tensor) -> torch.Tensor:
    closing = parameters["closing"].unsqueeze(1)
    peak = parameters["peak"].unsqueeze(1)
    returning_time = parameters["return"].unsqueeze(1)
    decay = parameters["decay"].unsqueeze(1)
    growth = parameters["growth"].unsqueeze(1)
    omega = (2.0 * math.pi * harmonics.to(closing.dtype)).unsqueeze(0)
    source_omega = math.pi / peak
    angle = source_omega * closing
    sine = torch.sin(angle)
    growth_full, omega_full = torch.broadcast_tensors(growth, omega)
    s = torch.complex(growth_full, -omega_full)
    rotation = torch.polar(torch.ones_like(omega * closing), -omega * closing)
    opening = -(rotation * (s * sine - source_omega * torch.cos(angle)) + source_omega * torch.exp(-growth * closing)) / (sine * (s * s - source_omega * source_omega))
    tail = 1.0 - closing
    decay_full, omega_full = torch.broadcast_tensors(decay, omega)
    rate = torch.complex(decay_full, omega_full)
    returning = -rotation / rate
    endpoint = 1.0 - decay * returning_time
    correction = rotation * endpoint * (1.0 - torch.polar(torch.ones_like(omega * tail), -omega * tail)) / (torch.complex(torch.zeros_like(omega_full), omega_full) * returning_time * rate)
    returning = torch.where(decay * endpoint >= REFERENCE_PERIOD, returning + correction, returning)
    return opening + returning


class HarmonicSum(torch.autograd.Function):

    CHUNK = 16

    @staticmethod
    def _basis(omega, harmonics, samples):
        angle = omega[:, None, None] * harmonics[None, :, None] * samples[None, None, :]
        return torch.cos(angle), torch.sin(angle)

    @staticmethod
    def forward(ctx, real, imag, omega, length):
        harmonics = torch.arange(1, real.shape[1] + 1, dtype=real.dtype, device=real.device)
        samples = torch.arange(length, dtype=real.dtype, device=real.device)
        output = real.new_zeros(real.shape[0], length)
        for start in range(0, real.shape[0], HarmonicSum.CHUNK):
            stop = start + HarmonicSum.CHUNK
            cos, sin = HarmonicSum._basis(omega[start:stop], harmonics, samples)
            output[start:stop] = torch.einsum("fh,fhn->fn", real[start:stop], cos) - torch.einsum("fh,fhn->fn", imag[start:stop], sin)
        ctx.save_for_backward(omega)
        ctx.shape = real.shape
        ctx.length = length
        return output

    @staticmethod
    def backward(ctx, grad):
        (omega,) = ctx.saved_tensors
        frames, count = ctx.shape
        harmonics = torch.arange(1, count + 1, dtype=grad.dtype, device=grad.device)
        samples = torch.arange(ctx.length, dtype=grad.dtype, device=grad.device)
        grad_real = grad.new_zeros(frames, count)
        grad_imag = grad.new_zeros(frames, count)
        for start in range(0, frames, HarmonicSum.CHUNK):
            stop = start + HarmonicSum.CHUNK
            cos, sin = HarmonicSum._basis(omega[start:stop], harmonics, samples)
            grad_real[start:stop] = torch.einsum("fn,fhn->fh", grad[start:stop], cos)
            grad_imag[start:stop] = -torch.einsum("fn,fhn->fh", grad[start:stop], sin)
        return grad_real, grad_imag, None, None


def overlap_add(segments: torch.Tensor, hop: int, offset: int, length: int) -> torch.Tensor:
    frames, width = segments.shape
    total = (frames - 1) * hop + width
    folded = F.fold(segments.t().unsqueeze(0), output_size=(1, total), kernel_size=(1, width), stride=(1, hop)).reshape(-1)
    return F.pad(folded, (0, max(0, offset + length - total)))[offset:offset + length]


class SpectralHeads(nn.Module):
    def __init__(self, node: Node):
        super().__init__()
        outputs, bands = struct.unpack_from("<II", node.payload)
        if outputs != 3 or len(node.payload) != 8 + 16 * bands or len(node.children) != 1:
            raise ValueError("unsupported spectral heads")
        descriptors = [struct.unpack_from("<4I", node.payload, 8 + 16 * index) for index in range(bands)]
        self.networks = nn.ModuleList(build_network(child) for child in node.children[0].children)
        self.bands = []
        for first, end, fade_in, fade_out in descriptors:
            count = end - first
            numerator = np.ones(count, dtype=np.float32)
            denominator = np.ones(count, dtype=np.float32)
            bins = np.arange(count, dtype=np.float32)
            if fade_in:
                numerator[:fade_in], denominator[:fade_in] = bins[:fade_in], fade_in - 1
            if fade_out:
                numerator[count - fade_out:], denominator[count - fade_out:] = count - 1 - bins[count - fade_out:], fade_out - 1
            self.bands.append((first, count))
            self.register_buffer(f"numerator{len(self.bands) - 1}", torch.from_numpy(numerator))
            self.register_buffer(f"denominator{len(self.bands) - 1}", torch.from_numpy(denominator))
        self.bins = max(end for _, end, _, _ in descriptors)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        output = condition.new_zeros(3, condition.shape[0], self.bins)
        for index, ((first, count), network) in enumerate(zip(self.bands, self.networks)):
            predicted = network(condition).reshape(condition.shape[0], 3, count).transpose(0, 1)
            faded = predicted * getattr(self, f"numerator{index}") / getattr(self, f"denominator{index}")
            output = output.index_add(2, torch.arange(first, first + count, device=condition.device), faded)
        return output


@dataclass
class VocoderOutput:
    samples: torch.Tensor
    periodic: torch.Tensor
    aperiodic: torch.Tensor
    stages: dict[str, torch.Tensor]


class Vocoder(nn.Module):
    def __init__(self, model: DnniFile):
        super().__init__()
        root = model.root
        if root.type != "_nhntv2" or len(root.payload) != 18 or len(root.children) != 13 or not root.payload[12] & 1:
            raise ValueError("expected the 13-child _nhntv2 vocoder with a shape network")
        self.bands = struct.unpack_from("<I", root.payload, 0)[0]
        self.excitation_gain = math.exp(struct.unpack_from("<f", root.payload, 4)[0])
        self.gate_noise = bool(root.payload[13])
        self.modulation_bias = struct.unpack_from("<f", root.payload, 14)[0]
        configuration = root.children[0]
        if configuration.type not in ("_vocfv1", "_vocfv2") or struct.unpack_from("<I", configuration.payload, 0)[0] != 72:
            raise ValueError("unsupported vocoder feature configuration")
        self.force_voiced = configuration.type == "_vocfv2" and struct.unpack_from("<I", configuration.payload, 48)[0] != 0
        self.sample_rate = struct.unpack_from("<I", configuration.payload, 8)[0]
        self.frame_period = struct.unpack_from("<f", configuration.payload, 12)[0]
        self.hop = int(np.round(self.sample_rate * self.frame_period))
        if self.hop % self.bands:
            raise ValueError("the hop must be a multiple of the residual band count")
        self.input_normalizer = MinMaxNormalizer(configuration.children[0])
        self.shape_network = build_network(root.children[1])
        self.condition_network = build_network(root.children[2])
        self.spectral_heads = SpectralHeads(root.children[3])
        self.register_buffer("ratios", torch.from_numpy(read_vector(root.children[4])))
        self.modulation_network = build_network(root.children[5])
        for index in range(3):
            if struct.unpack_from("<I", root.children[6 + index].payload, 0)[0] != 2:
                raise ValueError("unsupported vocoder filter window")
        self.register_buffer("synthesis_filters", torch.from_numpy(np.stack([read_vector(child) for child in root.children[9].children])))
        self.register_buffer("analysis_filters", torch.from_numpy(np.stack([read_vector(child) for child in root.children[10].children])))
        self.residual_network = build_network(root.children[11])
        self.residual_projection = build_network(root.children[12])

        window = 2 * self.hop
        self.fft_size = 2 * (1 << (window - 2).bit_length())
        phase = np.arange(window, dtype=np.float32) * np.float32(2.0 * np.pi) / np.float32(window)
        hann = np.float32(0.5) - np.float32(0.5) * np.cos(phase)
        self.register_buffer("analysis_window", torch.from_numpy(hann / np.float32(hann[0] + hann[self.hop])))
        boundary = np.ones(self.fft_size, dtype=np.float32)
        boundary[:16] = (np.arange(16) + 1) / 16.0
        boundary[-16:] = (self.fft_size - 1 - np.arange(self.fft_size - 16, self.fft_size)) / 16.0
        self.register_buffer("boundary_window", torch.from_numpy(boundary))

    @property
    def sample_rate_float(self) -> float:
        return float(np.float32(self.hop) / np.float32(self.frame_period))

    def normalize(self, frames: torch.Tensor) -> torch.Tensor:
        voicing = torch.ones_like(frames[:, :1]) if self.force_voiced else frames[:, :1]
        return self.input_normalizer.normalize(torch.cat([voicing, torch.log(frames[:, 1:2]), frames[:, 2:]], dim=1), clamp=True)

    def forward(self, frames: torch.Tensor, source_gradient: bool = True, deterministic_noise: bool = True) -> VocoderOutput:
        f0 = frames[:, 1].detach().cpu().numpy().astype(np.float32)
        if np.any(f0 <= 0):
            raise ValueError("F0 must be positive in every vocoder frame")
        voiced = np.ones(len(f0), dtype=bool) if self.force_voiced else frames[:, 0].detach().cpu().numpy() >= 0.5
        normalized = self.normalize(frames)
        shape = self.shape_network(normalized)[:, 0]
        rd = 2.9 / (torch.exp(-shape) + 1.0) + 0.1
        condition = self.condition_network(normalized)
        spectra = self.spectral_heads(condition)
        modulation_parameters = self.modulation_network(condition)

        timing = self._timing(f0)
        excitation = self.excitation(rd if source_gradient else rd.detach(), f0, voiced, timing)
        modulation = self.modulation(modulation_parameters, f0, timing)
        modulated = excitation * (modulation + self.modulation_bias)
        residual = self.residual(normalized, excitation, deterministic_noise)
        harmonic = self.filter(spectra[0], excitation)
        aperiodic = self.filter(spectra[1], residual)
        modulated_component = self.filter(spectra[2], modulated)
        stages = {
            "normalized": normalized, "shape": shape, "condition": condition, "spectra": spectra,
            "modulation_parameters": modulation_parameters, "excitation": excitation, "modulation": modulation,
            "residual": residual, "components": (harmonic, aperiodic, modulated_component),
        }
        periodic = harmonic + modulated_component
        return VocoderOutput(periodic + aperiodic, periodic, aperiodic, stages)

    def _timing(self, f0: np.ndarray) -> dict[str, np.ndarray]:
        increments = (f0 * np.float32(self.frame_period)).astype(np.float64)
        cycles = np.cumsum(increments)
        return {"increments": increments, "cycles": cycles}

    def excitation(self, rd: torch.Tensor, f0: np.ndarray, voiced: np.ndarray, timing: dict[str, np.ndarray]) -> torch.Tensor:
        hop = self.hop
        sample_rate = self.sample_rate_float
        limit = (np.float32(0.5) * np.float32(sample_rate) / f0).astype(np.float32)
        counts = np.minimum(2.0 * hop, limit.astype(np.float64)).astype(np.int64)
        active = voiced & (counts > 0)
        frames = len(f0)
        dtype = dsp_dtype(rd.device)
        segments = rd.new_zeros(frames, 2 * hop, dtype=dtype)
        if active.any():
            index = np.flatnonzero(active)
            last = np.minimum(counts[index], 2 * hop - 1)
            harmonics = torch.arange(1, int(last.max()) + 1, device=rd.device)
            parameters = lf_parameters(rd[torch.from_numpy(index).to(rd.device)])
            spectrum = lf_spectrum(parameters, harmonics)
            reference = lf_spectrum(parameters, torch.tensor([1, 2], device=rd.device))
            reference_magnitude = torch.abs(reference[:, 1])
            reference_phase = torch.angle(reference[:, 0])
            f0_active = torch.from_numpy(f0[index].astype(np.float64)).to(rd.device, dtype)
            scale = 0.1 * sample_rate / (f0_active * torch.from_numpy(counts[index].astype(np.float64)).to(rd.device, dtype) * reference_magnitude)
            frame_phase = torch.from_numpy(2.0 * np.pi * (np.fmod(timing["cycles"][index], 1.0) - timing["increments"][index])).to(rd.device, dtype)
            phase = torch.angle(spectrum) + harmonics.to(dtype) * (frame_phase - reference_phase).unsqueeze(1)
            amplitude = torch.abs(spectrum) * scale.unsqueeze(1)
            amplitude = amplitude * (harmonics.unsqueeze(0) <= torch.from_numpy(last).to(rd.device).unsqueeze(1))
            omega = torch.from_numpy(2.0 * (np.pi * f0[index].astype(np.float64) / sample_rate)).to(rd.device, dtype)
            signal = HarmonicSum.apply(amplitude * torch.cos(phase), amplitude * torch.sin(phase), omega, 2 * hop)
            segments = segments.index_copy(0, torch.from_numpy(index).to(rd.device), signal)
        window = self._hann(2 * hop, self.excitation_gain, rd.device)
        return overlap_add(segments * window, hop, hop, frames * hop).to(rd.dtype)

    def _hann(self, length: int, gain: float, device) -> torch.Tensor:
        return gain * (0.5 - 0.5 * torch.cos(2.0 * math.pi * torch.arange(length, dtype=dsp_dtype(device), device=device) / length))

    def modulation(self, parameters: torch.Tensor, f0: np.ndarray, timing: dict[str, np.ndarray]) -> torch.Tensor:
        hop = self.hop
        ratios = self.ratios.detach().cpu().numpy().astype(np.float64)
        count = len(ratios)
        angular = 2.0 * np.pi * ratios
        cycles = timing["cycles"].astype(np.float32).astype(np.float64)
        dtype = dsp_dtype(parameters.device)
        base = torch.from_numpy(np.fmod(angular[None, :] * cycles[:, None], 2.0 * np.pi)).to(parameters.device, dtype)
        step = torch.from_numpy(angular[None, :] * f0.astype(np.float64)[:, None] * float(self.frame_period) / hop).to(parameters.device, dtype)
        offsets = torch.arange(2 * hop, dtype=dtype, device=parameters.device) - hop
        amplitude = torch.exp(parameters[:, :count].to(dtype))
        angle = (base + parameters[:, count:].to(dtype)).unsqueeze(2) + step.unsqueeze(2) * offsets
        segments = (amplitude.unsqueeze(2) * torch.cos(angle)).sum(dim=1)
        window = self._hann(2 * hop, 1.0, parameters.device)
        return overlap_add(segments * window, hop, hop, len(f0) * hop).to(parameters.dtype)

    def residual(self, normalized: torch.Tensor, excitation: torch.Tensor, deterministic_noise: bool = True) -> torch.Tensor:
        bands = self.bands
        filters = self.analysis_filters
        length = filters.shape[1]
        frames = normalized.shape[0] * self.hop // bands
        tail = length // 2
        first = bands - 1 - tail % bands
        start = first + (bands // 2 - 1) * bands - (length - 1)
        signal = excitation
        if start < 0:
            signal = F.pad(signal, (-start, 0))
        else:
            signal = signal[start:]
        needed = (frames - 1) * bands + length
        signal = F.pad(signal, (0, max(0, needed - signal.shape[0])))[:needed]
        analysed = F.conv1d(signal.view(1, 1, -1), filters.unsqueeze(1).to(signal.dtype), stride=bands)[0].t()
        if deterministic_noise:
            noise = torch.from_numpy(residual_noise(frames * bands).reshape(frames, bands)).to(signal.device, signal.dtype)
        else:
            noise = torch.randn(frames, bands, device=signal.device, dtype=signal.dtype)
        hidden = self.residual_network(torch.cat([analysed, noise], dim=1), self._interpolate(normalized, self.hop // bands))
        subbands = self.residual_projection(hidden)
        if self.gate_noise:
            subbands = noise * torch.sigmoid(subbands)
        return self._synthesise(subbands)

    @staticmethod
    def _interpolate(features: torch.Tensor, factor: int) -> torch.Tensor:
        leading = factor // 2
        offset = 0.5 if factor % 2 == 0 else 0.0
        steps = (torch.arange(factor, dtype=torch.float32, device=features.device) + offset) * np.float32(1.0 / factor)
        previous, current = features[:-1], features[1:]
        middle = ((current - previous).unsqueeze(1) * steps[None, :, None] + previous.unsqueeze(1)).reshape(-1, features.shape[1])
        return torch.cat([features[:1].expand(leading, -1), middle, features[-1:].expand(factor - leading, -1)])

    def _synthesise(self, subbands: torch.Tensor) -> torch.Tensor:
        bands = self.bands
        filters = self.synthesis_filters.to(subbands.dtype)
        length = filters.shape[1]
        latency = (length - 1) // 2
        frames = subbands.shape[0]
        total = frames * bands
        held = min(frames, -(-latency // bands))
        output = subbands.new_zeros(total + length + bands)
        if held:
            output[:length] = output[:length] + subbands[:held].sum(dim=0) @ filters
        if held < frames:
            placed = F.conv_transpose1d(subbands[held:].t().unsqueeze(0), filters.unsqueeze(1), stride=bands)[0, 0]
            begin = held * bands - latency
            output = output.index_add(0, torch.arange(begin, begin + placed.shape[0], device=subbands.device), placed)
        return output[:total]

    def filter(self, cepstra: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        hop, size = self.hop, self.fft_size
        frames, channels = cepstra.shape
        half = channels // 2
        coefficients = cepstra.new_zeros(frames, size)
        lifter_low = 10.0 / torch.arange(1, half + 2, dtype=cepstra.dtype, device=cepstra.device)
        coefficients[:, :half + 1] = cepstra[:, :half + 1] * lifter_low
        lifter_high = 10.0 / (half + 1 - torch.arange(1, half, dtype=cepstra.dtype, device=cepstra.device))
        coefficients[:, size - half + 1:] = cepstra[:, half + 1:] * lifter_high
        response = torch.exp(torch.fft.rfft(coefficients, dim=1))
        window = 2 * hop
        padding = (size - window) // 2
        padded = F.pad(source, (hop, hop))
        segments = padded.unfold(0, window, hop)[:frames] * self.analysis_window.to(source.dtype)
        buffers = F.pad(segments, (padding, size - window - padding))
        filtered = torch.fft.irfft(torch.fft.rfft(buffers, dim=1) * response, n=size, dim=1) * self.boundary_window.to(source.dtype)
        return overlap_add(filtered, hop, size // 2, frames * hop)
