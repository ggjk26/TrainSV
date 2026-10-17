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

COMPOSITE_PHONES: dict[str, tuple[str, ...]] = {
    "@_n": ("@", ":n"),
    "@N": ("@", "N"),
    "a_n": ("a", ":n"),
    "iE_n": ("iE", ":n"),
    "u@N": ("u@", "N"),
    "y{_n": ("y{", ":n"),
}
# X-SAMPA uses a grave accent for rhoticity. These remain one phone when the
# complete symbol exists in the table; otherwise approximate with a vowel plus
# a rhotic phone from the target inventory.
RHOTICIZED_VOWELS = {"@`": "@", "3`": "3"}
RHOTIC_APPROXIMANTS = ("r\\", "ɹ", "ɻ", "r", "R", "R\\")
PHONE_REPLACEMENTS = {"k_h": "kh", r"ts%5c": "ts\\"}

# Explicit bridge from the Mandarin X-SAMPA symbols documented by Synthesizer V
# Studio 2.3 to the newer romanized phone inventory. This is language-specific:
# symbols such as p, z, and y do not have the same *phone-set label* in the new
# Mandarin inventory as their generic IPA meanings suggest.
XSAMPA_TO_SYNTHV_SIMPLE: dict[str, tuple[str, ...]] = {
    "a": ("a",), "A": ("aa",), "O": ("o/",), "@": ("e/",),
    "e": ("ee",), "7": ("e",), "U": ("o",), "u": ("u",), "i": ("i",),
    "r\\": ("iz",), ":r\\`": ("ir",), "y": ("v",),
    "au": ("ao",), "ou": ("ou",), "ia": ("ia",), "iA": ("iaa",),
    "iau": ("iao",), "ie": ("ie",), "iE": ("iae",), "iU": ("io",),
    "jou": ("iu",), "i^": ("-i",), ":n": ("-n",), "N": ("ng",),
    "p": ("b",), "p_h": ("p",), "t": ("d",), "t_h": ("t",),
    "k": ("g",), "k_h": ("k",), "ts": ("z",), "ts_h": ("c",),
    "ts\\": ("j",), "ts\\_h": ("q",), "ts`": ("zh",),
    "ts`_h": ("ch",), "x": ("h",), "s`": ("sh",), "s\\": ("x",),
    "z`": ("r",), "j": ("y",), "f": ("f",), "s": ("s",),
    "m": ("m",), "n": ("n",), "l": ("l",), "w": ("w",),
    "y{": ("va",), "yE": ("ve",), "ua": ("ua",), "uA": ("uaa",),
    "u@": ("ue/",), "ue": ("ue",), "uo": ("uo",),
    ":\\i": ("-i",), r"r\`": ("-r",),
    "ts\\h": ("q",),
    "ts`h": ("ch",),
}
# Pre-2.3 SynthV Mandarin inventory spellings that were replaced by X-SAMPA
# in the newer phone table. Map these only when the selected target uses the
# simplified/new inventory; a legacy voice should keep its own exact symbols.
SYNTHV_LEGACY_XSAMPA_ALIASES = {
    "o": "O", "AU": "au", "@U": "ou", "iAU": "iau", "i@U": "jou",
    "i\\": "r\\", "i`": ":r\\`", ":\\i": "i^",
    "ph": "p_h", "th": "t_h", "kh": "k_h", "tsh": "ts_h",
    "ts\\h": "ts\\_h", "ts`h": "ts`_h",
}
_SYNTHV_SIMPLE_MARKERS = {
    "aa", "ee", "e/", "o/", "ao", "ou", "iz", "ir", "v", "ng",
    "-n", "iaa", "iae", "io", "b", "d", "g", "z", "c",
}


def _uses_synthv_simple_mandarin(valid: set[str]) -> bool:
    """Recognize the new romanized Mandarin inventory from several markers."""
    return len(valid & _SYNTHV_SIMPLE_MARKERS) >= 2


def _map_xsampa_mandarin(
    token: str, valid: set[str], source_system: str
) -> list[str] | None:
    """Map standard or SynthV-legacy Mandarin X-SAMPA without guessing splits."""
    simple_target = _uses_synthv_simple_mandarin(valid)
    if not simple_target:
        # When the target is an X-SAMPA/legacy phone table, exact symbols keep
        # their target-defined meaning and take priority over aliases.
        if token in valid:
            return [token]
        if source_system == "synthv-xsampa":
            alias = SYNTHV_LEGACY_XSAMPA_ALIASES.get(token)
            if alias in valid:
                return [alias]
        return None

    canonical = SYNTHV_LEGACY_XSAMPA_ALIASES.get(token, token) if source_system == "synthv-xsampa" else token
    # If a known X-SAMPA symbol maps to a different simple label, never silently
    # pass through the same spelling (e.g. X-SAMPA p vs simplified Mandarin p).
    mapped = XSAMPA_TO_SYNTHV_SIMPLE.get(canonical)
    if mapped is not None:
        return list(mapped) if all(phone in valid for phone in mapped) else None

    # Same symbol, same phone in both inventories for these common Mandarin phones.
    if canonical in {"f", "m", "n", "l", "w"} and canonical in valid:
        return [canonical]
    return None


# Candidate target spellings for VOCALOID Mandarin tokens. The first
# available candidate is preferred; this supports simplified and X-SAMPA-style
# SynthV phone inventories without a user-maintained phone_table.json.
VOCALOID_MANDARIN_PHONE_CANDIDATES: dict[str, tuple[tuple[str, ...], ...]] = {
    "a": (("a",),), "A": (("aa",), ("A",)),
    "o": (("o/",), ("o",)), "@": (("e/",), ("e",), ("@",)),
    "e": (("ee",), ("e",)), "7": (("e",), ("7",)),
    "U": (("o",), ("U",)), "u": (("u",),), "i": (("i",),),
    "i\\": (("iz",), ("r\\",), ("i\\",)),
    "i¥": (("iz",), ("r\\",), ("i\\",), ("i¥",)),
    "i`": (("ir",), (":r\\`",), ("i`",)),
    "y": (("v",), ("y",)), "aI": (("ai",), ("aI",), ("a", "i")),
    "ei": (("ei",), ("eI",), ("ee", "i"), ("e", "i")), "AU": (("ao",), ("au",), ("AU",)),
    "@U": (("ou",), ("@U",)), "ia": (("ia",),),
    "iA": (("iaa",), ("iA",)), "iAU": (("iao",), ("iau",), ("iAU",)),
    "ie": (("ie",),), "iE": (("iae",), ("iE",)),
    "iU": (("io",), ("iU",)), "i@U": (("iu",), ("jou",), ("i@U",)),
    "ua": (("ua",),), "uo": (("uo",),), "ue": (("ue",),),
    "uA": (("uA",), ("ua",)), "u@": (("u@",), ("ua",)),
    "y{": (("ve",), ("vE",), ("y{",), ("v", "e")),
    "yE": (("ve",), ("vE",), ("yE",), ("v", "e")),
    "iE_r": (("iae",), ("iE",), ("ie",)),
    "yE_r": (("ve",), ("vE",), ("yE",), ("y{",), ("v", "e")),
    "uaI": (("uai",), ("uaI",)), "uei": (("uei",), ("ueI",)),
    "@`": (("er",), ("e/", "-r"), ("e/", "r"), ("e/", "rr"), ("e/", ":r\\`"), ("@", ":r\\`"), ("@`",)),
    "p": (("b",), ("p",)), "ph": (("p",), ("ph",)),
    "t": (("d",), ("t",)), "th": (("t",), ("th",)),
    "k": (("g",), ("k",)), "kh": (("k",), ("kh",)),
    "kw": (("gw",), ("kw",)), "kwh": (("kw",), ("kwh",)),
    "ts": (("z",), ("ts",)), "tsh": (("c",), ("tsh",)),
    "ts\\": (("j",), ("ts\\",)), "ts`": (("zh",), ("ts`",)),
    "ts`h": (("ch",), ("ts`h",)), "ts\\h": (("q",), ("ts\\h",)),
    "x": (("h",), ("x",)), "s`": (("sh",), ("s`",)),
    "s\\": (("x",), ("s\\",)), "z`": (("r",), ("z`",)),
    "N": (("ng",), ("N",)), ":\\i": (("-i",), (":i",), (":\\i",)),
    ":n": (("-n",), (":n",), ("n",)),
    "r\\`": (("rr",), (":r\\`",), ("r\\`",)),
    "b": (("b",),), "c": (("c",),), "d": (("d",),), "f": (("f",),),
    "g": (("g",),), "h": (("h",),), "j": (("j",),), "l": (("l",),),
    "m": (("m",),), "n": (("n",),), "s": (("s",),), "w": (("w",),),
}

# These VOCALOID tokens combine a vowel and a syllable coda. They are expanded
# only when the chosen target inventory contains every symbol in a candidate.
VOCALOID_MANDARIN_COMPOSITES: dict[str, tuple[tuple[str, ...], ...]] = {
    "@_n": (("e/", "-n"), ("e/", ":n"), ("e/", "n"), ("@", ":n"), ("@", "n")),
    "a_n": (("a", "-n"), ("a", ":n"), ("a", "n")),
    "i_n": (("i", "-n"), ("i", ":n"), ("i", "n")),
    "iE_n": (("iae", "-n"), ("iae", ":n"), ("iE", ":n"), ("iE", "-n"), ("iE", "n")),
    "ua_n": (("ua", "-n"), ("ua", ":n"), ("u", "a", "-n"), ("u", "a", ":n")),
    "u@_n": (("u@", "-n"), ("u@", ":n"), ("ua", "-n"), ("u", "e/", "-n"), ("u", "@", ":n")),
    "y_n": (("v", "-n"), ("v", ":n"), ("y", ":n"), ("y", "n")),
    "y{_n": (("ve", "-n"), ("vE", "-n"), ("v", "e", "-n"), ("y{", ":n"), ("y{", "n")),
    "AN": (("aa", "ng"), ("aa", "N"), ("A", "N")),
    "@N": (("e/", "ng"), ("e/", "N"), ("@", "N")),
    "iN": (("i", "ng"), ("i", "N")),
    "iAN": (("iaa", "ng"), ("iaa", "N"), ("iA", "N"), ("i", "aa", "ng")),
    "uAN": (("uA", "N"), ("uA", "ng"), ("ua", "ng"), ("u", "aa", "ng"), ("u", "A", "N")),
    "u@N": (("u@", "N"), ("u@", "ng"), ("u", "e/", "ng"), ("u", "@", "N")),
    "UN": (("o", "ng"), ("o", "N"), ("U", "N")),
    "iUN": (("io", "ng"), ("io", "N"), ("iU", "N")),
}

def _map_vocaloid_mandarin(token: str, valid: set[str]) -> list[str] | None:
    """Map one VOCALOID Mandarin token to spellings present in the target inventory."""
    # VOCALOID 6 uses underscore forms for aspirated consonants. Preserve the
    # old X-SAMPA spellings for a legacy target, but use SynthV 2.3 labels for
    # a simplified target inventory.
    aspirated_aliases = {
        "p_h": ("p", "ph"), "t_h": ("t", "th"), "k_h": ("k", "kh"),
        "ts_h": ("c", "tsh"), "ts\\_h": ("q", "ts\\h"), "ts`_h": ("ch", "ts`h"),
    }
    # If this is an older/X-SAMPA-labelled voice, identical names are already
    # meaningful (e.g. 7 and e are different vowels). Prefer direct matches there.
    simplified = bool(valid & {"aa", "ee", "e/", "o/", "ao", "ou", "iz", "ir", "v", "b", "d", "g", "z", "c", "ng", "-n", "-r"})
    if token in aspirated_aliases:
        preferred = aspirated_aliases[token][0 if simplified else 1]
        if preferred in valid:
            return [preferred]
        alternative = aspirated_aliases[token][1 if simplified else 0]
        if alternative in valid:
            return [alternative]
    if not simplified and token in valid:
        return [token]
    for candidate_group in (VOCALOID_MANDARIN_PHONE_CANDIDATES.get(token, ()), VOCALOID_MANDARIN_COMPOSITES.get(token, ())):
        for candidate in candidate_group:
            if candidate and all(phone in valid for phone in candidate):
                return list(candidate)
    return None


_TABLE_KEYS = ("phones", "phonemes", "symbols", "phone_table", "phoneTable", "inventory")
_SYMBOL_KEYS = ("symbol", "phone", "phoneme", "xsampa", "x_sampa", "name")


def read_phone_table(path: str | Path | None, fallback: Iterable[str] = ()) -> set[str]:
    """Read phone symbols from JSON, or use the selected voice's phone set."""
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
        if document and all(isinstance(key, str) for key in document):
            reserved = {"version", "language", "name", "description", "metadata"}
            keys = {key for key in document if key not in reserved}
            if keys and any(isinstance(value, (dict, str, int, float, bool, type(None))) for value in document.values()):
                return keys
    return set()


def parse_filename_phones(filename: str | Path, phone_table: Iterable[str], source_system: str | None = None) -> list[str]:
    """Parse ordered phones and add fixed boundary sils without deduplication."""
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
        # k_h and other underscore forms are meaningful X-SAMPA aspiration
        # markers; do not rewrite them to legacy aliases in the X-SAMPA modes.
        replacements = {} if source_system in {"xsampa", "synthv-xsampa"} else PHONE_REPLACEMENTS
        replacement = replacements.get(token)
        if replacement is not None:
            if replacement not in valid and source_system == "vocaloid-mandarin":
                converted = _map_vocaloid_mandarin(replacement, valid)
                if converted is not None:
                    phones.extend(converted)
                    continue
            if replacement not in valid:
                raise ValueError(
                    f"{Path(filename).name}: {token!r} maps to {replacement!r}, "
                    "which is not in the phone table"
                )
            phones.append(replacement)
            continue
        if source_system in {"xsampa", "synthv-xsampa"}:
            converted = _map_xsampa_mandarin(token, valid, source_system)
            if converted is not None:
                phones.extend(converted)
                continue
            # Compound legacy Mandarin syllable-final tokens are not single
            # X-SAMPA symbols; resolve them through the explicit compound table.
            if _uses_synthv_simple_mandarin(valid) and token in VOCALOID_MANDARIN_COMPOSITES:
                converted = _map_vocaloid_mandarin(token, valid)
                if converted is not None:
                    phones.extend(converted)
                    continue
        if source_system == "vocaloid-mandarin":
            converted = _map_vocaloid_mandarin(token, valid)
            if converted is not None and (converted != [token] or token not in valid):
                phones.extend(converted)
                continue
        if token in valid:
            phones.append(token)
            continue

        # @` and 3` are valid X-SAMPA rhoticized vowels. If this target
        # inventory lacks the complete symbol, approximate it without dropping
        # rhoticity; prefer an approximant spelling over a trill when available.
        base_vowel = RHOTICIZED_VOWELS.get(token)
        if base_vowel is not None:
            if base_vowel not in valid:
                raise ValueError(
                    f"{Path(filename).name}: {token!r} is not in the phone table "
                    f"and its base vowel {base_vowel!r} is unavailable"
                )
            rhotic = next((phone for phone in RHOTIC_APPROXIMANTS if phone in valid), None)
            if rhotic is None:
                raise ValueError(
                    f"{Path(filename).name}: {token!r} is a rhoticized vowel, but the "
                    "phone table has no compatible rhotic phone. Add the complete symbol "
                    "to the phone table, or include one of: "
                    + ", ".join(RHOTIC_APPROXIMANTS)
                )
            phones.extend((base_vowel, rhotic))
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
        # Explicit source conventions must never be heuristically split into
        # plausible-looking but phonetically incorrect phones. Their valid
        # conversions are all table-driven above.
        pieces = None if source_system in {"xsampa", "synthv-xsampa", "vocaloid-mandarin"} else _segment(token, valid)
        if pieces is None:
            raise ValueError(
                f"{Path(filename).name}: unknown phone {token!r}; it is not in "
                "the phone table and could not be split into existing phones"
            )
        phones.extend(pieces)
    phones.append("sil")
    return phones


def _segment(token: str, valid: set[str]) -> list[str] | None:
    """Split a composite spelling using only known symbols, preferring longer phones."""
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
    """Make contiguous labels with fixed boundary sils and equal middle spans."""
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
    wav_directory: str | Path,
    phone_table: Iterable[str],
    source_system: str | None = None,
    unmatched: list[dict[str, str]] | None = None,
) -> list[tuple[str, Path, list[tuple[int, int, str]]]]:
    """Build standard recording tuples from WAV names and actual WAV durations."""
    directory = Path(wav_directory).expanduser()
    if not directory.is_dir():
        raise ValueError(f"WAV directory does not exist: {directory}")
    wavs = sorted(directory.glob("*.wav"), key=lambda path: path.name.casefold())
    if not wavs:
        raise ValueError(f"no .wav files found in {directory}")
    recordings = []
    for wav in wavs:
        try:
            phones = parse_filename_phones(wav.name, phone_table, source_system=source_system)
        except ValueError as error:
            if unmatched is None:
                raise
            unmatched.append({"filename": wav.name, "error": str(error)})
            continue
        try:
            # Decode the file, not just its header: truncated/corrupt WAV payloads
            # must be reported as bad files, while a valid unvoiced consonant WAV
            # remains a valid recording regardless of whether Praat finds F0.
            decoded, rate = soundfile.read(str(wav), dtype="float32", always_2d=True)
            frames = int(decoded.shape[0])
            if rate <= 0 or frames <= 0 or decoded.shape[1] <= 0:
                raise ValueError("audio has no samples or an invalid sample rate")
        except (OSError, RuntimeError, ValueError) as error:
            message = f"{wav.name}: unreadable or damaged WAV: {error}"
            if unmatched is None:
                raise ValueError(message) from error
            unmatched.append({"filename": wav.name, "error": message})
            continue
        total_ticks = round(frames * TICKS_PER_SECOND / rate)
        try:
            labels = labels_from_phones(phones, total_ticks)
        except ValueError as error:
            message = f"{wav.name}: {error}"
            if unmatched is None:
                raise ValueError(message) from error
            unmatched.append({"filename": wav.name, "error": message})
            continue
        recordings.append((wav.stem, wav, labels))
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
        path.write_text("".join(f"{start} {end} {phone}\n" for start, end, phone in entries), encoding="utf-8")
        paths.append(path)
    return paths
