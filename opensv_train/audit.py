"""Dataset coverage audit for missing samples and weak phoneme transitions.

This module reports training-data risks without rewriting the dataset or
changing the labels used by the trainer.
"""
from __future__ import annotations

import csv
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


_SILENCE = {"SP", "sil", "pau"}
_BREATH = {"AP", "br"}


def _canonical_phone(token: str) -> str:
    if token in _SILENCE:
        return "sil"
    if token in _BREATH:
        return "br"
    return token


def _example_list(items: list[dict[str, Any]], limit: int = 50) -> list[dict[str, Any]]:
    return items[:limit]


def audit_dataset(
    directory: str | Path,
    *,
    min_phone_recordings: int = 3,
    min_transition_recordings: int = 2,
    short_phone_ms: float = 20.0,
    check_audio: bool = False,
) -> dict[str, Any]:
    """Audit a TrainSV dataset and return a JSON-serializable coverage report.

    Phone/transition coverage counts distinct trainable recording IDs, not just
    token frequency, so repeated labels in one WAV cannot disguise a missing
    sample. By default, WAV headers are not decoded; pass check_audio=True to
    inspect headers and compare label duration against WAV duration.
    """
    if min_phone_recordings < 1:
        raise ValueError("min_phone_recordings must be at least 1")
    if min_transition_recordings < 1:
        raise ValueError("min_transition_recordings must be at least 1")
    if not math.isfinite(short_phone_ms) or short_phone_ms < 0:
        raise ValueError("short_phone_ms must be finite and non-negative")

    root = Path(directory).expanduser()
    csv_path = root / "transcriptions.csv"
    wav_dir = root / "wavs"
    report: dict[str, Any] = {
        "dataset": str(root),
        "settings": {
            "min_phone_recordings": min_phone_recordings,
            "min_transition_recordings": min_transition_recordings,
            "short_phone_ms": short_phone_ms,
            "check_audio": check_audio,
        },
        "summary": {
            "rows": 0,
            "trainable_rows": 0,
            "unique_recordings": 0,
            "missing_wavs": 0,
            "unreadable_wavs": 0,
            "invalid_rows": 0,
            "duplicate_names": 0,
            "short_phone_occurrences": 0,
            "phone_types": 0,
            "transition_types": 0,
        },
        "fatal_errors": [],
        "missing_wavs": [],
        "unreadable_wavs": [],
        "invalid_rows": [],
        "duplicate_names": [],
        "duration_mismatches": [],
        "short_phone_examples": [],
        "phone_coverage": [],
        "rare_phones": [],
        "transition_coverage": [],
        "rare_transitions": [],
    }
    if not root.is_dir():
        report["fatal_errors"].append(f"Dataset directory does not exist: {root}")
        return report
    if not csv_path.is_file():
        report["fatal_errors"].append(f"Missing transcriptions.csv: {csv_path}")
        return report
    if not wav_dir.is_dir():
        report["fatal_errors"].append(f"Missing WAV directory: {wav_dir}")
        return report

    # One directory scan means the audit stays linear in the dataset size,
    # instead of issuing a separate filesystem lookup for every CSV row.
    wavs: dict[str, Path] = {}
    try:
        for wav in wav_dir.glob("*.wav"):
            if wav.is_file():
                wavs.setdefault(os.path.normcase(wav.stem), wav)
    except OSError as error:
        report["fatal_errors"].append(f"Could not scan WAV directory: {error}")
        return report

    phone_occurrences: Counter[str] = Counter()
    phone_recordings: dict[str, set[str]] = defaultdict(set)
    transition_occurrences: Counter[tuple[str, str]] = Counter()
    transition_recordings: dict[tuple[str, str], set[str]] = defaultdict(set)
    short_phone_counts: Counter[str] = Counter()
    seen_names: set[str] = set()
    duplicate_names: set[str] = set()
    missing: dict[str, list[int]] = defaultdict(list)
    unreadable: dict[str, list[int]] = defaultdict(list)
    invalid: list[dict[str, Any]] = []
    duration_mismatches: list[dict[str, Any]] = []
    short_examples: list[dict[str, Any]] = []
    valid_name_count = 0

    try:
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            required = {"name", "ph_seq", "ph_dur"}
            absent = sorted(required - columns)
            if absent:
                report["fatal_errors"].append(
                    "transcriptions.csv is missing required columns: " + ", ".join(absent)
                )
                return report

            for row_number, row in enumerate(reader, start=2):
                report["summary"]["rows"] += 1
                name = (row.get("name") or "").strip()
                row_errors: list[str] = []
                if not name:
                    row_errors.append("empty recording name")
                elif Path(name).name != name or "/" in name or chr(92) in name:
                    row_errors.append("recording name must be a file stem, not a path")
                norm_name = os.path.normcase(name) if name else ""

                if norm_name:
                    if norm_name in seen_names:
                        duplicate_names.add(name)
                    seen_names.add(norm_name)

                tokens = (row.get("ph_seq") or "").split()
                duration_text = (row.get("ph_dur") or "").split()
                if not tokens:
                    row_errors.append("ph_seq is empty")
                try:
                    durations = [float(value) for value in duration_text]
                except ValueError:
                    durations = []
                    row_errors.append("ph_dur contains a non-numeric value")

                if len(tokens) != len(duration_text):
                    row_errors.append(
                        f"ph_seq has {len(tokens)} phones but ph_dur has {len(duration_text)} durations"
                    )
                if len(durations) == len(tokens):
                    bad_duration = [
                        index for index, value in enumerate(durations)
                        if not math.isfinite(value) or value <= 0
                    ]
                    if bad_duration:
                        row_errors.append(
                            f"{len(bad_duration)} phone duration(s) are non-finite or not positive"
                        )

                wav_path = wavs.get(norm_name) if norm_name else None
                if name and wav_path is None:
                    missing[name].append(row_number)

                audio_unreadable = False
                if check_audio and wav_path is not None and not row_errors:
                    try:
                        import soundfile as sf
                        info = sf.info(str(wav_path))
                        if info.frames <= 0 or info.samplerate <= 0:
                            raise RuntimeError("WAV has no audio frames or an invalid sample rate")
                        label_seconds = sum(durations)
                        wav_seconds = float(info.duration)
                        tolerance = max(0.08, 0.01 * wav_seconds)
                        if abs(label_seconds - wav_seconds) > tolerance:
                            duration_mismatches.append({
                                "name": name,
                                "row": row_number,
                                "label_seconds": round(label_seconds, 4),
                                "wav_seconds": round(wav_seconds, 4),
                                "difference_seconds": round(label_seconds - wav_seconds, 4),
                            })
                    except (OSError, RuntimeError, ValueError) as error:
                        unreadable[name].append(row_number)
                        audio_unreadable = True

                if row_errors:
                    invalid.append({"name": name or f"row-{row_number}", "row": row_number, "errors": row_errors})
                    continue
                if wav_path is None or audio_unreadable:
                    continue

                valid_name_count += 1
                report["summary"]["trainable_rows"] += 1
                phones = [_canonical_phone(token) for token in tokens]
                unique_phones = {phone for phone in phones if phone not in {"sil", "br"}}
                for phone in phones:
                    phone_occurrences[phone] += 1
                for phone in unique_phones:
                    phone_recordings[phone].add(name)

                # Transitions adjacent to silence are kept out of the rare
                # transition table: they mostly measure phrase padding, not a
                # consonant/vowel handoff within speech.
                for index, (left, right) in enumerate(zip(phones, phones[1:])):
                    if left == "sil" or right == "sil":
                        continue
                    pair = (left, right)
                    transition_occurrences[pair] += 1
                    transition_recordings[pair].add(name)

                if short_phone_ms > 0:
                    for token, seconds in zip(phones, durations):
                        if token in {"sil", "br"} or seconds * 1000.0 >= short_phone_ms:
                            continue
                        short_phone_counts[token] += 1
                        if len(short_examples) < 100:
                            short_examples.append({
                                "name": name,
                                "row": row_number,
                                "phone": token,
                                "duration_ms": round(seconds * 1000.0, 3),
                            })

    except (OSError, UnicodeDecodeError, csv.Error) as error:
        report["fatal_errors"].append(f"Could not read transcriptions.csv: {error}")
        return report

    report["summary"].update({
        "unique_recordings": len(seen_names),
        "missing_wavs": len(missing),
        "unreadable_wavs": len(unreadable),
        "invalid_rows": len(invalid),
        "duplicate_names": len(duplicate_names),
        "short_phone_occurrences": sum(short_phone_counts.values()),
        "phone_types": len(phone_occurrences),
        "transition_types": len(transition_occurrences),
    })
    report["missing_wavs"] = _example_list([
        {"name": name, "rows": rows} for name, rows in sorted(missing.items())
    ])
    report["unreadable_wavs"] = _example_list([
        {"name": name, "rows": rows} for name, rows in sorted(unreadable.items())
    ])
    report["invalid_rows"] = _example_list(invalid)
    report["duplicate_names"] = sorted(duplicate_names)[:50]
    report["duration_mismatches"] = _example_list(duration_mismatches)
    report["short_phone_examples"] = short_examples

    coverage = [
        {
            "phone": phone,
            "occurrences": phone_occurrences[phone],
            "recordings": len(phone_recordings.get(phone, set())),
        }
        for phone in phone_occurrences
    ]
    coverage.sort(key=lambda item: (item["recordings"], item["phone"]))
    report["phone_coverage"] = coverage
    report["rare_phones"] = [
        item for item in coverage
        if item["phone"] not in {"sil", "br"} and item["recordings"] < min_phone_recordings
    ]

    transitions = [
        {
            "from": left,
            "to": right,
            "occurrences": transition_occurrences[(left, right)],
            "recordings": len(transition_recordings[(left, right)]),
        }
        for left, right in transition_occurrences
    ]
    transitions.sort(key=lambda item: (item["recordings"], item["from"], item["to"]))
    report["transition_coverage"] = transitions
    report["rare_transitions"] = [
        item for item in transitions if item["recordings"] < min_transition_recordings
    ]
    return report
