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
        nasal_weight: float = 0.0, nasal_margin_db: float = 0.0,
    ):
        super().__init__()
        self.mel_weight = mel_weight
        self.brightness_weight = float(brightness_weight)
        self.nasal_weight = float(nasal_weight)
        self.nasal_margin_db = float(nasal_margin_db)
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


    def _nasal_balance(
        self, predicted: torch.Tensor, target: torch.Tensor, frame_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalise oral-vowel frames that are more low-band heavy than the target.

        The proxy is the 200-450 Hz (nasal-murmur region) to 450-1100 Hz (first
        formant region) power ratio in dB at the 2048-point STFT. Comparing against
        the target's own ratio on the same frames removes the vowel identity. The
        hinge is one-sided: matching or undershooting the target costs nothing, and
        ``nasal_margin_db`` asks for that many dB less than the target. Returns the
        hinge in units of 10 dB and the mean signed difference in dB.
        """
        frequencies = torch.fft.rfftfreq(2048, d=1.0 / self.sample_rate, device=predicted.device)
        low = ((frequencies >= 200.0) & (frequencies < 450.0)).to(predicted.dtype)[:, None]
        reference = ((frequencies >= 450.0) & (frequencies < 1100.0)).to(predicted.dtype)[:, None]
        floor = (target.detach().amax() * self.floor).square() + 1e-12

        def ratio_db(magnitude: torch.Tensor) -> torch.Tensor:
            power = magnitude.square()
            return 10.0 * (
                torch.log10((power * low).sum(dim=0) + floor)
                - torch.log10((power * reference).sum(dim=0) + floor)
            )

        delta = ratio_db(predicted) - ratio_db(target)
        mask = frame_mask.to(delta.dtype)
        count = mask.sum().clamp(min=1.0)
        hinge = (torch.relu(delta + self.nasal_margin_db) * mask).sum() / count / 10.0
        return hinge, (delta * mask).sum() / count

    def forward(
        self, predicted: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None,
        nasal_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        convergence = predicted.new_zeros(())
        log_magnitude = predicted.new_zeros(())
        brightness = predicted.new_zeros(())
        nasal = nasal_delta = predicted.new_zeros(())
        for n_fft, hop in RESOLUTIONS:
            p, t = self._magnitude(predicted, n_fft, hop), self._magnitude(target, n_fft, hop)
            frame_weights = p.new_ones(p.shape[1]) if weights is None else weights[torch.clamp(torch.arange(p.shape[1], device=p.device) * hop, max=len(weights) - 1)]
            root = frame_weights.sqrt()
            convergence = convergence + torch.linalg.norm((t - p) * root) / torch.linalg.norm(t * root).clamp(min=1e-5)
            log_magnitude = log_magnitude + self._log_distance(p, t, frame_weights)
            if n_fft == 2048:
                mel = self._log_distance(self.mel @ p, self.mel @ t, frame_weights)
                brightness = self._brightness_balance(p, t, frame_weights)
                if nasal_mask is not None:
                    index = torch.clamp(torch.arange(p.shape[1], device=p.device) * hop, max=len(nasal_mask) - 1)
                    nasal, nasal_delta = self._nasal_balance(p, t, nasal_mask[index])
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
        if nasal_mask is not None:
            terms["nasal"] = nasal
            terms["nasal_delta_db"] = nasal_delta
            terms["total"] = terms["total"] + self.nasal_weight * nasal
        return terms
