from __future__ import annotations

import numpy as np
import torch
from torch import nn

RESOLUTIONS = ((512, 128), (1024, 256), (2048, 512))


def mel_filterbank(sample_rate: int, n_fft: int, mels: int, low: float, high: float) -> np.ndarray:
    def to_mel(hz):
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def to_hz(mel):
        return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)

    edges = to_hz(np.linspace(to_mel(low), to_mel(high), mels + 2))
    bins = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    lower, centre, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    weights = np.maximum(0.0, np.minimum((bins - lower) / (centre - lower), (upper - bins) / (upper - centre)))
    return (weights * (2.0 / (upper - lower))).astype(np.float32)


class SpectralLoss(nn.Module):

    def __init__(
        self, sample_rate: int = 48000, mels: int = 128,
        mel_weight: float = 1.0, floor_db: float = 50.0,
        brightness_weight: float = 0.0,
    ):
        super().__init__()
        self.mel_weight = mel_weight
        self.brightness_weight = float(brightness_weight)
        self.sample_rate = sample_rate
        self.floor = 10.0 ** (-floor_db / 20.0)
        for n_fft, _ in RESOLUTIONS:
            self.register_buffer(f"window{n_fft}", torch.hann_window(n_fft))
        self.register_buffer("mel", torch.from_numpy(mel_filterbank(sample_rate, 2048, mels, 40.0, 16000.0)))

    def _magnitude(self, signal: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
        spectrum = torch.stft(signal, n_fft, hop, window=getattr(self, f"window{n_fft}"), return_complex=True, center=True)
        return torch.sqrt(spectrum.real ** 2 + spectrum.imag ** 2 + 1e-9)

    def _log_distance(self, predicted: torch.Tensor, target: torch.Tensor, frame_weights: torch.Tensor) -> torch.Tensor:
        floor = target.detach().amax() * self.floor + 1e-7
        return torch.mean(torch.abs(torch.log(target + floor) - torch.log(predicted + floor)) * frame_weights) / frame_weights.mean()

    def _brightness_balance(
        self, predicted: torch.Tensor, target: torch.Tensor, frame_weights: torch.Tensor
    ) -> torch.Tensor:
        """Match target high-to-mid spectral balance without applying a post filter.

        The ratio uses 3–10 kHz versus 1–3 kHz power at the 2048-point STFT.
        A zero brightness weight leaves the original training objective unchanged.
        """
        frequencies = torch.fft.rfftfreq(
            2048, d=1.0 / self.sample_rate, device=predicted.device
        )
        mid_mask = ((frequencies >= 1000.0) & (frequencies < 3000.0)).to(predicted.dtype)
        high_mask = ((frequencies >= 3000.0) & (frequencies <= 10000.0)).to(predicted.dtype)
        p_power = predicted.square()
        t_power = target.square()
        p_mid = (p_power * mid_mask[:, None]).sum(dim=0)
        p_high = (p_power * high_mask[:, None]).sum(dim=0)
        t_mid = (t_power * mid_mask[:, None]).sum(dim=0)
        t_high = (t_power * high_mask[:, None]).sum(dim=0)
        floor = target.detach().amax() * self.floor + 1e-7
        p_balance = torch.log(p_high + floor) - torch.log(p_mid + floor)
        t_balance = torch.log(t_high + floor) - torch.log(t_mid + floor)
        weights = frame_weights.to(p_balance.dtype)
        return torch.mean(torch.abs(p_balance - t_balance) * weights) / weights.mean().clamp(min=1e-8)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        convergence = predicted.new_zeros(())
        log_magnitude = predicted.new_zeros(())
        brightness = predicted.new_zeros(())
        for n_fft, hop in RESOLUTIONS:
            p, t = self._magnitude(predicted, n_fft, hop), self._magnitude(target, n_fft, hop)
            frame_weights = p.new_ones(p.shape[1]) if weights is None else weights[torch.clamp(torch.arange(p.shape[1], device=p.device) * hop, max=len(weights) - 1)]
            root = frame_weights.sqrt()
            convergence = convergence + torch.linalg.norm((t - p) * root) / torch.linalg.norm(t * root).clamp(min=1e-5)
            log_magnitude = log_magnitude + self._log_distance(p, t, frame_weights)
            if n_fft == 2048:
                mel = self._log_distance(self.mel @ p, self.mel @ t, frame_weights)
                brightness = self._brightness_balance(p, t, frame_weights)
        count = len(RESOLUTIONS)
        terms = {
            "convergence": convergence / count,
            "log_magnitude": log_magnitude / count,
            "mel": mel,
            "brightness": brightness,
        }
        terms["total"] = (
            terms["convergence"] + terms["log_magnitude"]
            + self.mel_weight * mel + self.brightness_weight * brightness
        )
        return terms
