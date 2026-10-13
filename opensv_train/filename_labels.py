"""Generate approximate HTS/vLabeler labels from phone-encoded WAV filenames.

The filename supplies only the phoneme order. This module intentionally uses the
requested equal-duration approximation; it is not acoustic forced alignment.
All LAB timestamps are integer 100-nanosecond ticks.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import soundfile

TICKS_PER_SECOND = 10_000_000
SILENCE_FRACTION = 0.10

# Tokens explicitly represented by multiple phones in the user's reclist.
COMPOSITE_PHONES: dict[str, tuple[str, ...]] = {
    "@_n": ("@", ":n"),
    "@N": ("@", "N"),
    "a_n": ("a", ":n"),
    "iE_n": ("iE", ":n"),
    "u@N": ("u@", "N"),
    "y{_n": ("y{", ":n"),
}

# These are replacements, not splits.
PHONE_REPLACEMENTS = {
    "k_h": "kh",
    r"ts%5c": "ts\\",
}

_TABLE_KEYS = ("phones", "phonemes", "symbols", "phone_table", "phoneTable", "inventory")
_SYMBOL_KEYS = ("symbol", "phone", "phoneme", "xsampa", "x_sampa", "name")


def read_phone_table(path: str | Path | None, fallback: Iterable[str] = ()) -> set[str]:
    """Read phone symbols from a JSON table, or use the supplied voice phone set.

    JSON can be a list of symbols/records, an object with a phones,
    phonemes, symbols, or phone_table field, or an object keyed by phone symbol
    (for example, {"a": "vowel", "b": "consonant"}).
    """
    if path is None or not str(path).strip():
        symbols = {str(symbol) for symbol in fallback if str(symbol)}
        if not symbols:
            raise ValueError("no phone table was supplied and the selected voice has no phone symbols")
        symbols.add("sil")
        return symbols

    table_path = Path(path).expanduser()
    try:
        document = json.loads(table_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"could not read phone table {table_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON phone table {table_path}: {error}") from error

    symbols = _extract_symbols(document)
    if not symbols:
        raise ValueError(
            f"{table_path}: could not find phone symbols; use a list or an object "
            "with a phones/phonemes/symbols field"
        )
    symbols.add("sil")
    return symbols


def _extract_symbols(document: object) -> set[str]:
    if isinstance(document, list):
        found: set[str] = set()
        for item in document:
            if isinstance(item, str) and item:
                found.add(item)
            elif isinstance(item, dict):
                for key in _SYMBOL_KEYS:
                    value = item.get(key)
                    if isinstance(value, str) and value:
                        found.add(value)
                        break
        return found

    if isinstance(document, dict):
        for key in _TABLE_KEYS:
            if key in document:
                return _extract_symbols(document[key])
        # Supports symbol-keyed phone tables, e.g. {"a": {"type": "vowel"}}.
        if document and all(isinstance(key, str) for key in document):
            reserved = {"version", "language", "name", "description", "metadata"}
            keys = {key for key in document if key not in reserved}
            if keys and any(isinstance(value, (dict, str, int, float, bool, type(None))) for value in document.values()):
                return keys
    return set()


def parse_filename_phones(filename: str | Path, phone_table: Iterable[str]) -> list[str]:
    """Parse a WAV filename into its ordered phones, adding fixed boundary sils.

    Only the terminal _0 or _1 retake suffix is removed. Underscores inside
    the phoneme spelling are retained. The original internal sequence is never
    deduplicated: a filename containing Sil m_0.wav becomes sil sil m sil.
    """
    valid = set(phone_table)
    valid.add("sil")

    stem = Path(filename).stem
    if stem.endswith(("_0", "_1")):
        stem = stem[:-2]
    raw_phones = stem.split()
    if not raw_phones:
        raise ValueError(f"{Path(filename).name}: filename contains no space-separated phonemes")

    phones = ["sil"]
    for token in raw_phones:
        if token.casefold() == "sil":
            phones.append("sil")
            continue
        if token in valid:
            phones.append(token)
            continue

        replacement = PHONE_REPLACEMENTS.get(token)
        if replacement is not None:
            if replacement not in valid:
                raise ValueError(
                    f"{Path(filename).name}: {token!r} maps to {replacement!r}, "
                    "which is not in the phone table"
                )
            phones.append(replacement)
            continue

        composite = COMPOSITE_PHONES.get(token)
        if composite is not None:
            missing = [phone for phone in composite if phone not in valid]
            if missing:
                raise ValueError(
                    f"{Path(filename).name}: cannot expand {token!r}; "
                    f"missing phone(s) from the phone table: {', '.join(missing)}"
                )
            phones.extend(composite)
            continue

        pieces = _segment(token, valid)
        if pieces is None:
            raise ValueError(
                f"{Path(filename).name}: unknown phone {token!r}; it is not in "
                "the phone table and could not be split into existing phones"
            )
        phones.extend(pieces)

    phones.append("sil")
    return phones


def _segment(token: str, valid: set[str]) -> list[str] | None:
    """Split a composite spelling using only known symbols, preferring long phones."""
    candidates = tuple(sorted((phone for phone in valid if phone != "sil"), key=lambda p: (-len(p), p)))

    @lru_cache(maxsize=None)
    def visit(position: int) -> tuple[str, ...] | None:
        if position == len(token):
            return ()
        options: list[tuple[str, ...]] = []
        for phone in candidates:
            if token.startswith(phone, position):
                tail = visit(position + len(phone))
                if tail is not None:
                    options.append((phone, *tail))
        if not options:
            return None
        return min(options, key=lambda parts: (len(parts), tuple(-len(part) for part in parts), parts))

    result = visit(0)
    return list(result) if result else None


def labels_from_phones(
    phones: list[str], total_ticks: int, silence_fraction: float = SILENCE_FRACTION
) -> list[tuple[int, int, str]]:
    """Make contiguous labels with fixed 10% boundary sils and equal middle spans."""
    if len(phones) < 3 or phones[0] != "sil" or phones[-1] != "sil":
        raise ValueError("phone sequence must contain fixed leading and trailing sil")
    if total_ticks <= 0:
        raise ValueError("audio duration must be positive")
    if not 0 < silence_fraction < 0.5:
        raise ValueError("silence fraction must be between 0 and 0.5")

    middle_count = len(phones) - 2
    boundary_ticks = round(total_ticks * silence_fraction)
    middle_ticks = total_ticks - 2 * boundary_ticks
    if boundary_ticks < 1 or middle_ticks < middle_count:
        raise ValueError("audio is too short to give every phoneme a positive-duration interval")

    entries: list[tuple[int, int, str]] = [(0, boundary_ticks, "sil")]
    start = boundary_ticks
    for index, phone in enumerate(phones[1:-1], start=1):
        end = boundary_ticks + round(middle_ticks * index / middle_count)
        entries.append((start, end, phone))
        start = end
    entries.append((total_ticks - boundary_ticks, total_ticks, "sil"))
    return entries


def filename_recordings(
    wav_directory: str | Path, phone_table: Iterable[str]
) -> list[tuple[str, Path, list[tuple[int, int, str]]]]:
    """Build standard (name, WAV path, 100-ns label tuples) records from WAV names."""
    directory = Path(wav_directory).expanduser()
    if not directory.is_dir():
        raise ValueError(f"WAV directory does not exist: {directory}")
    wavs = sorted(directory.glob("*.wav"), key=lambda path: path.name.casefold())
    if not wavs:
        raise ValueError(f"no .wav files found in {directory}")

    recordings = []
    for wav in wavs:
        phones = parse_filename_phones(wav.name, phone_table)
        try:
            info = soundfile.info(str(wav))
        except (OSError, RuntimeError) as error:
            raise ValueError(f"{wav.name}: could not read audio duration: {error}") from error
        if info.samplerate <= 0 or info.frames <= 0:
            raise ValueError(f"{wav.name}: audio has no samples or an invalid sample rate")
        total_ticks = round(info.frames * TICKS_PER_SECOND / info.samplerate)
        entries = labels_from_phones(phones, total_ticks)
        recordings.append((wav.stem, wav, entries))
    return recordings


def write_lab_files(
    recordings: list[tuple[str, Path, list[tuple[int, int, str]]]], output_directory: str | Path
) -> list[Path]:
    """Write generated HTS/vLabeler LAB files under the requested output directory."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, _wav, entries in recordings:
        path = output / f"{name}.lab"
        text = "".join(f"{start} {end} {phone}\n" for start, end, phone in entries)
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths
