from __future__ import annotations

import numpy as np

from .common import PhoneSet, find_language
from .duration import Syllable

REST_CONTEXT_SECONDS = 0.1
ONSET_CATEGORIES = ("stop", "affricate", "aspirate", "fricative", "nasal", "liquid", "semivowel")


def random_phrase(phone_sets: list[PhoneSet], language: str, rng: np.random.Generator, notes: int = 8, register: tuple[int, int] = (57, 72)) -> list[Syllable]:
    phone_set = phone_sets[find_language(phone_sets, language)]
    vowels = [s for s, c in zip(phone_set.symbols, phone_set.categories) if c == "vowel"]
    onsets = [s for s, c in zip(phone_set.symbols, phone_set.categories) if c in ONSET_CATEGORIES and s != "cl"]
    silence = "sil" if "sil" in phone_set.symbols else "pau"
    pitch = int(rng.integers(*register))
    syllables = [Syllable(language, [silence], REST_CONTEXT_SECONDS, pitch)]
    for index in range(notes):
        pitch = int(np.clip(pitch + rng.integers(-4, 5), *register))
        seconds = float(rng.choice([0.15, 0.25, 0.3, 0.45, 0.6, 0.9]))
        if index and rng.random() < 0.12:
            syllables.append(Syllable(language, [silence], float(rng.choice([0.1, 0.15])), pitch))
        if index and rng.random() < 0.15 and syllables[-1].phonemes[-1] in vowels:
            syllables.append(Syllable(language, [syllables[-1].phonemes[-1]], seconds, pitch, continuation=True))
            continue
        phonemes = [str(rng.choice(vowels))]
        if rng.random() < 0.8:
            phonemes.insert(0, str(rng.choice(onsets)))
        syllables.append(Syllable(language, phonemes, seconds, pitch))
    syllables.append(Syllable(language, [silence], REST_CONTEXT_SECONDS, pitch))
    return syllables


def _note_name(midi: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[midi % 12]}{midi // 12 - 1}"


def render_dataset(teacher, output, phrases: int, language: str = "japanese", seed: int = 0, notes: int = 8, log=print):
    import csv
    from pathlib import Path

    import soundfile
    import torch

    from .acoustic import TimedPhoneme, midi_to_log_f0
    from .duration import quantize_durations
    from .pitch import PitchNote, pad_pitch_notes

    output = Path(output)
    (output / "wavs").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(phrases):
        syllables = random_phrase(teacher.duration.phone_sets, language, rng, notes=notes)
        durations = teacher.duration.predict(syllables)
        frames = quantize_durations(durations, 0.005)
        interval_frames = np.zeros(len(syllables), dtype=int)
        merged = []
        for duration, count in zip(durations, frames):
            interval_frames[duration.interval] += count
            if syllables[duration.syllable].continuation and merged and merged[-1][0] == duration.symbol:
                merged[-1][1] += count
            else:
                merged.append([duration.symbol, count, duration.interval])
        score = [PitchNote(s, silence=s.phonemes in (["sil"], ["pau"]), continuation=s.continuation) for s in syllables[1:-1]]
        pitch_notes = pad_pitch_notes(score)
        midi = teacher.pitch.predict(pitch_notes, teacher.duration.predict([note.syllable for note in pitch_notes]))
        total = sum(frames)
        offset = 0.5 - REST_CONTEXT_SECONDS
        times = offset + (np.arange(total) + 0.5) * 0.005
        curve = np.interp(times / 0.005, np.arange(len(midi)), midi)
        timed = [TimedPhoneme(language, symbol, count) for symbol, count, _ in merged]
        with torch.no_grad():
            acoustic = teacher.acoustic(timed, torch.from_numpy(midi_to_log_f0(curve).astype(np.float32)))
            audio = teacher.vocoder(acoustic.frames).samples.numpy()
        name = f"synthetic_{index:04d}"
        soundfile.write(str(output / "wavs" / f"{name}.wav"), audio, teacher.vocoder.sample_rate, subtype="FLOAT")
        word_of_syllable = []
        for syllable in syllables:
            word_of_syllable.append(word_of_syllable[-1] if syllable.continuation else (word_of_syllable[-1] + 1 if word_of_syllable else 0))
        ph_num = [0] * (word_of_syllable[-1] + 1)
        for _, _, syllable in merged:
            ph_num[word_of_syllable[syllable]] += 1
        seconds = list(interval_frames * 0.005)
        rows.append({
            "name": name,
            "ph_seq": " ".join({"sil": "SP", "pau": "SP", "br": "AP"}.get(symbol, symbol) for symbol, _, _ in merged),
            "ph_dur": " ".join(f"{count * 0.005:.6f}" for _, count, _ in merged),
            "ph_num": " ".join(str(count) for count in ph_num if count),
            "note_seq": " ".join("rest" if s.phonemes in (["sil"], ["pau"]) else _note_name(s.pitch) for s in syllables),
            "note_dur": " ".join(f"{value:.6f}" for value in seconds),
            "note_slur": " ".join("1" if s.continuation else "0" for s in syllables),
        })
        log(f"rendered {name} ({total * 0.005:.2f}s)")
    with open(output / "transcriptions.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
