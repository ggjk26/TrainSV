from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .common import find_language
from .data import Phrase
from .duration import interval_log_fractions
from .losses import SpectralLoss
from .voice import Voice

MODELS = ("acoustic", "duration", "pitch", "vocoder")


@dataclass
class TrainConfig:
    models: tuple[str, ...] = ("acoustic",)
    params: str = "all"
    steps: int = 2000
    learning_rate: float | None = None
    warmup: int = 0
    lr_end: float | None = None
    anchor: float = 1.0
    segment_frames: int = 160
    margin_frames: int = 12
    voicing_weight: float = 0.1
    consonant_weight: float = 1.0
    fixed_noise: bool = True
    evaluate_every: int = 250
    log_every: int = 25
    gradient_clip: float = 1.0
    batch: int = 1
    seed: int = 0
    device: str = "cpu"
    rap_languages: list[str] = field(default_factory=list)
    pitch_melody_weight: float = 1.0
    pitch_motion_weight: float = 1.0
    pitch_smoothing_hz: float = 3.0


def parameter_groups(voice: Voice, models, params: str) -> dict[str, list[torch.nn.Parameter]]:
    vectors = {
        "duration": lambda: [voice.duration.voice],
        "pitch": lambda: [voice.pitch.speaker],
        "acoustic": lambda: [voice.acoustic.speaker, voice.acoustic.default_style, *voice.acoustic.mode_styles],
        "vocoder": lambda: [],
    }
    groups: dict[str, list[torch.nn.Parameter]] = {}
    for name in models:
        key = "acoustic" if name in ("acoustic", "vocoder") else name
        chosen = [_tensor(value) for value in vectors[name]()] if params == "embeddings" else list(voice.models[name].parameters())
        groups.setdefault(key, []).extend(chosen)
    return {key: group for key, group in groups.items() if group}


def _tensor(value) -> torch.nn.Parameter:
    return value if isinstance(value, torch.nn.Parameter) else value.value


class Trainer:
    def __init__(self, voice: Voice, train: list[Phrase], validation: list[Phrase], config: TrainConfig, log=print, on_best=None, on_final=None, journal=None, on_checkpoint=None):
        self.voice = voice
        self.on_best = on_best
        self.on_final = on_final
        self.journal = journal
        self.on_checkpoint = on_checkpoint
        self.train_phrases = train
        self.validation_phrases = validation
        self.config = config
        self.log = log
        self.device = torch.device(config.device)
        voice.to(self.device)
        for parameter in voice.parameters():
            parameter.requires_grad_(False)
        self.groups = parameter_groups(voice, config.models, config.params)
        self.parameters = [parameter for group in self.groups.values() for parameter in group]
        for parameter in self.parameters:
            parameter.requires_grad_(True)
        self.anchors = [parameter.detach().clone() for parameter in self.parameters]
        self.anchor_weights = [1.0 / float(anchor.pow(2).mean().clamp(min=1e-8)) / anchor.numel() / len(self.anchors) for anchor in self.anchors]
        default_rate = 3e-3 if config.params == "embeddings" else 2e-5
        self.rate = config.learning_rate or default_rate
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.rate)
        self.spectral = SpectralLoss().to(self.device)
        self.random = random.Random(config.seed)

    def eligible(self, model: str, phrases: list[Phrase]) -> list[Phrase]:
        if model in ("duration", "pitch"):
            return [phrase for phrase in phrases if phrase.syllables]
        return phrases

    def acoustic_terms(self, phrase: Phrase, train_vocoder: bool, full: bool = False) -> dict[str, torch.Tensor]:
        acoustic, vocoder = self.voice.acoustic, self.voice.vocoder
        hz, voiced = phrase.frame_f0()
        log_f0 = torch.from_numpy(np.log(hz).astype(np.float32)).to(self.device)
        seed = 5489 if self.config.fixed_noise else None
        output = acoustic(phrase.timed_phonemes(), log_f0, seed=seed, mode_weights=mode_weights(acoustic, phrase.utterance.style))
        frames = output.frames
        target_voicing = torch.from_numpy(voiced.astype(np.float32)).to(self.device)
        frames = torch.cat([target_voicing.unsqueeze(1), frames[:, 1:]], dim=1)
        if not train_vocoder and "acoustic" not in self.config.models:
            frames = frames.detach()
        count = len(hz)
        width = count if full else min(self.config.segment_frames, count)
        first = 0 if full else self.random.randrange(0, count - width + 1)
        margin = 0 if full else self.config.margin_frames
        begin, end = max(0, first - margin), min(count, first + width + margin)
        rendered = vocoder(frames[begin:end], deterministic_noise=self.config.fixed_noise).samples
        hop = vocoder.hop
        predicted = rendered[(first - begin) * hop:(first - begin + width) * hop]
        audio = phrase.audio()[first * hop:(first + width) * hop]
        target = torch.from_numpy(audio).to(self.device, predicted.dtype)
        weights = None
        if not full and self.config.consonant_weight != 1.0:
            per_frame = np.repeat([self._phone_weight(phone.language, phone.symbol) for phone in phrase.phones], phrase.phone_frames)
            weights = torch.from_numpy(np.repeat(per_frame[first:first + width], hop).astype(np.float32)).to(self.device, predicted.dtype)
        terms = self.spectral(predicted, target, weights)
        voicing = acoustic.acoustic_normalizer.denormalize(output.normalized)[:, 0].clamp(1e-4, 1 - 1e-4)
        terms["voicing"] = F.binary_cross_entropy(voicing, target_voicing)
        terms["acoustic"] = terms["total"] + self.config.voicing_weight * terms["voicing"]
        return terms

    def _phone_weight(self, language: str, symbol: str) -> float:
        phone_sets = self.voice.acoustic.phone_sets
        phone_set = phone_sets[find_language(phone_sets, language)]
        category = phone_set.categories[phone_set.symbols.index(symbol)]
        return 1.0 if category in ("vowel", "diphthong", "silence", "breath") else self.config.consonant_weight

    def duration_terms(self, phrase: Phrase) -> dict[str, torch.Tensor]:
        model = self.voice.duration
        inputs = model.prepare(phrase.syllables)
        predicted = interval_log_fractions(model(inputs), inputs)
        target_seconds = torch.tensor(phrase.syllable_targets, dtype=torch.float32, device=self.device)
        target = interval_log_fractions(torch.log(target_seconds), inputs)
        return {"duration": torch.mean((predicted - target) ** 2)}

    def pitch_terms(self, phrase: Phrase) -> dict[str, torch.Tensor]:
        model = self.voice.pitch
        notes, phonemes, midi, mask = phrase.pitch_example()
        inputs = model.prepare(notes, phonemes, self.config.rap_languages)
        count = min(inputs.frames, len(midi))
        scale = model.note_normalizer.scale[2]
        predicted = model(inputs)[:count] * scale
        target = model.normalize_pitch(torch.from_numpy(midi[:count] + inputs.transpose).to(self.device)) * scale
        weights = torch.from_numpy(mask[:count].astype(np.float32)).to(self.device)
        voiced = weights.sum().clamp(min=1.0)
        slow_predicted = _low_pass(predicted, cutoff_hz=self.config.pitch_smoothing_hz)
        slow_target = _low_pass(target, cutoff_hz=self.config.pitch_smoothing_hz)
        melody = (torch.abs(slow_predicted - slow_target) * weights).sum() / voiced
        fast_predicted, fast_target = (predicted - slow_predicted) * weights, (target - slow_target) * weights
        motion = (_motion_spectrum(fast_predicted) - _motion_spectrum(fast_target)).abs().mean()
        frame_error = (torch.abs(predicted - target) * weights).sum() / voiced
        ratio = fast_predicted.detach().abs().sum() / fast_target.abs().sum().clamp(min=1e-6)
        weighted = self.config.pitch_melody_weight * melody + self.config.pitch_motion_weight * motion
        return {"pitch": weighted, "pitch_semitones": frame_error, "pitch_melody": melody, "pitch_motion": ratio}

    def anchor_penalty(self) -> torch.Tensor:
        return sum(torch.sum((parameter - anchor) ** 2) * weight for parameter, anchor, weight in zip(self.parameters, self.anchors, self.anchor_weights))

    def step_terms(self, model: str, phrase: Phrase, full: bool = False) -> dict[str, torch.Tensor]:
        if model in ("acoustic", "vocoder"):
            return self.acoustic_terms(phrase, model == "vocoder", full)
        if model == "duration":
            return self.duration_terms(phrase)
        return self.pitch_terms(phrase)

    def loss_key(self, model: str) -> str:
        return "acoustic" if model in ("acoustic", "vocoder") else model

    def evaluate(self) -> dict[str, float]:
        totals: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        audio_models = [m for m in self.config.models if m in ("acoustic", "vocoder")][:1]
        other = [m for m in self.config.models if m in ("duration", "pitch")]
        with torch.no_grad():
            for model in audio_models + other:
                for phrase in self.eligible(model, self.validation_phrases):
                    for key, value in self.step_terms(model, phrase, full=True).items():
                        scalar = value.detach()
                        totals[key] = scalar if key not in totals else totals[key] + scalar
                        counts[key] = counts.get(key, 0) + 1
        if not totals:
            return {}
        # Validation may visit thousands of phrases. Aggregate on-device and
        # transfer all final means together to avoid a host sync per metric.
        keys = list(totals)
        means = torch.stack([totals[key] / counts[key] for key in keys]).cpu().tolist()
        return dict(zip(keys, map(float, means)))

    def run(self, resume: dict | None = None) -> list[dict]:
        config = self.config
        history = []
        best: dict[str, tuple[float, int, list[torch.Tensor]]] = {}
        start = 0
        if resume is not None:
            start, history, best = resume["step"], resume["history"], resume["best"]
            self._restore({key: resume["weights"][key] for key in self.groups})
            self.optimizer.load_state_dict(resume["optimizer"])
            self.random.setstate(resume["random"])
            self.log(f"resumed at step {start}")
        elif self.validation_phrases:
            metrics = self.evaluate()
            self.log(f"step 0 validation: {_format(metrics)}")
            self._record({"step": 0, "validation": metrics})
            history.append({"step": 0, **metrics})
            best = {key: (self._selection(key, metrics), 0, self._snapshot(group)) for key, group in self.groups.items() if key in metrics}
        started = time.time()
        # Prepare eligible examples and cumulative duration weights once, not
        # on every optimizer step. random.choices accepts cumulative weights
        # directly, avoiding rebuilding the prefix sums for audio sampling.
        running: dict[str, list[torch.Tensor]] = {}
        audio_model = "vocoder" if "vocoder" in config.models else "acoustic"
        schedule = [m for m in config.models if m in ("duration", "pitch")] + ([audio_model] if any(m in ("acoustic", "vocoder") for m in config.models) else [])
        eligible_by_model = {model: self.eligible(model, self.train_phrases) for model in schedule}
        cumulative_weights = {}
        for model, phrases in eligible_by_model.items():
            if model in ("acoustic", "vocoder") and phrases:
                cumulative_weights[model] = np.cumsum([phrase.frames for phrase in phrases]).tolist()
        for step in range(start + 1, config.steps + 1):
            rate = learning_rate(step, config.steps, self.rate, config.warmup, config.lr_end)
            for group in self.optimizer.param_groups:
                group["lr"] = rate
            self.optimizer.zero_grad()
            loss = self.parameters[0].new_zeros(())
            for model in schedule:
                phrases = eligible_by_model[model]
                if not phrases:
                    continue
                for _ in range(config.batch):
                    if model in cumulative_weights:
                        phrase = self.random.choices(phrases, cum_weights=cumulative_weights[model])[0]
                    else:
                        phrase = self.random.choices(phrases)[0]
                    terms = self.step_terms(model, phrase)
                    loss = loss + terms[self.loss_key(model)] / config.batch
                    for key, value in terms.items():
                        # Keep detached scalars on-device and synchronize only
                        # when the log interval is emitted.
                        running.setdefault(key, []).append(value.detach())
            penalty = self.anchor_penalty()
            loss = loss + config.anchor * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, config.gradient_clip)
            self.optimizer.step()
            running.setdefault("anchor", []).append(penalty.detach())
            if step % config.log_every == 0:
                seconds = (time.time() - started) / (step - start)
                # One device-to-host transfer for the whole metrics batch,
                # rather than one scalar synchronization for every term and step.
                metric_keys = list(running)
                metric_values = [torch.stack(values).mean() for values in running.values()]
                host_values = torch.stack(metric_values).detach().cpu().tolist()
                means = dict(zip(metric_keys, map(float, host_values)))
                self.log(f"step {step}/{config.steps} ({seconds:.2f}s/step): {_format(means)}")
                self._record({"step": step, "seconds_per_step": seconds, "lr": rate, **means})
                running = {}
            if step % config.evaluate_every == 0 or step == config.steps:
                if self.validation_phrases:
                    metrics = self.evaluate()
                    self.log(f"step {step} validation: {_format(metrics)}")
                    self._record({"step": step, "validation": metrics})
                    history.append({"step": step, **metrics})
                    improved = [key for key in best if self._selection(key, metrics) < best[key][0]]
                    for key in improved:
                        best[key] = (self._selection(key, metrics), step, self._snapshot(self.groups[key]))
                    if improved and self.on_best is not None:
                        current = {key: self._snapshot(self.groups[key]) for key in best}
                        self._restore({key: entry[2] for key, entry in best.items()})
                        self.on_best(step)
                        self._restore(current)
                if self.on_checkpoint is not None:
                    self.on_checkpoint({"step": step, "history": history, "best": best, "random": self.random.getstate(),
                                        "weights": {key: self._snapshot(group) for key, group in self.groups.items()}, "optimizer": self.optimizer.state_dict()})
        if self.on_final is not None:
            self.on_final(config.steps)
        self._restore({key: entry[2] for key, entry in best.items()})
        for key, (_, step, _) in best.items():
            self.log(f"kept the {key} weights from step {step}, its best validation score")
        return history

    def _record(self, record: dict) -> None:
        if self.journal is not None:
            self.journal(record)

    def _selection(self, key: str, metrics: dict[str, float]) -> float:
        if key == "acoustic":
            return metrics["mel"] + self.config.voicing_weight * metrics["voicing"]
        return metrics[key]

    @staticmethod
    def _snapshot(group: list[torch.Tensor]) -> list[torch.Tensor]:
        return [parameter.detach().clone() for parameter in group]

    def _restore(self, snapshots: dict[str, list[torch.Tensor]]) -> None:
        with torch.no_grad():
            for key, values in snapshots.items():
                for parameter, value in zip(self.groups[key], values):
                    parameter.copy_(value)


_LOW_PASS: dict[tuple[str, float, float], torch.Tensor] = {}
_MOTION_WINDOWS: dict[tuple[str, torch.dtype], torch.Tensor] = {}


def _low_pass(curve: torch.Tensor, cutoff_hz: float = 3.0, frame_rate: float = 200.0) -> torch.Tensor:
    if not 0.0 < cutoff_hz < frame_rate / 2.0:
        raise ValueError(f"pitch smoothing cutoff must be between 0 and {frame_rate / 2:g} Hz")
    key = (str(curve.device), float(cutoff_hz), float(frame_rate))
    kernel = _LOW_PASS.get(key)
    if kernel is None:
        taps = torch.arange(-64, 65, dtype=torch.float32, device=curve.device)
        kernel = torch.sinc(2 * cutoff_hz / frame_rate * taps) * torch.hann_window(129, periodic=False, device=curve.device)
        kernel = (kernel / kernel.sum()).view(1, 1, -1)
        _LOW_PASS[key] = kernel
    padded = F.pad(curve.view(1, 1, -1), (64, 64), mode="replicate")
    return F.conv1d(padded, kernel).view(-1)


def _motion_spectrum(curve: torch.Tensor) -> torch.Tensor:
    key = (str(curve.device), curve.dtype)
    window = _MOTION_WINDOWS.get(key)
    if window is None:
        window = torch.hann_window(128, device=curve.device, dtype=curve.dtype)
        _MOTION_WINDOWS[key] = window
    padded = F.pad(curve, (0, max(0, 128 - len(curve))))
    return torch.stft(padded, 128, 16, window=window, return_complex=True).abs()


def learning_rate(step: int, steps: int, rate: float, warmup: int = 0, end: float | None = None) -> float:
    ramp = min(1.0, step / warmup) if warmup else 1.0
    if end is None:
        return ramp * rate
    return ramp * (end + (rate - end) * 0.5 * (1 + math.cos(math.pi * min(step / steps, 1.0))))


def _format(metrics: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()) if not math.isnan(value))


def level_gain(voice: Voice, phrases: list[Phrase], device: torch.device, limit: int = 24) -> float:
    sample = phrases[::max(1, len(phrases) // limit)][:limit]
    rendered_energy = recorded_energy = 0.0
    with torch.no_grad():
        for phrase in sample:
            hz, _ = phrase.frame_f0()
            frames = voice.acoustic(phrase.timed_phonemes(), torch.from_numpy(np.log(hz).astype(np.float32)).to(device)).frames
            rendered_energy += float(np.sum(voice.vocoder(frames).samples.cpu().numpy() ** 2))
            recorded_energy += float(np.sum(phrase.audio() ** 2))
    return float(np.sqrt(rendered_energy / max(recorded_energy, 1e-12)))


def mode_weights(acoustic, style: str | None) -> torch.Tensor | None:
    if style is None:
        return None
    weights = torch.zeros(len(acoustic.vocal_modes))
    weights[acoustic.vocal_modes.index(style)] = 1.0
    return weights


def mode_levels(voice: Voice, phrases: list[Phrase], device: torch.device, limit: int = 24) -> dict[str, float]:
    sample = phrases[::max(1, len(phrases) // limit)][:limit]
    energies = {}
    with torch.no_grad():
        for style in [None, *voice.acoustic.vocal_modes]:
            energies[style] = 0.0
            for phrase in sample:
                hz, _ = phrase.frame_f0()
                frames = voice.acoustic(phrase.timed_phonemes(), torch.from_numpy(np.log(hz).astype(np.float32)).to(device), mode_weights=mode_weights(voice.acoustic, style)).frames
                energies[style] += float(torch.sum(voice.vocoder(frames).samples ** 2))
    return {style: 0.5 * float(np.log(energies[style] / energies[None])) for style in voice.acoustic.vocal_modes}


def apply_gain(phrases: list[Phrase], gain: float) -> None:
    for utterance in {id(phrase.utterance): phrase.utterance for phrase in phrases}.values():
        utterance.audio = utterance.audio * np.float32(gain)
