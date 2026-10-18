"""Generate approximate HTS/vLabeler labels from phone-encoded WAV filenames.

The filename supplies only the phoneme order. This module intentionally uses the
requested equal-duration approximation; it is not acoustic forced alignment.
All LAB timestamps are integer 100-nanosecond ticks.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .audio_io import read_wav_frames

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
# Canonical X-SAMPA aspiration tokens mapped to the pre-2.3 Mandarin labels
# found in older voice inventories. Exact target symbols still take priority.
STANDARD_XSAMPA_TO_SYNTHV_LEGACY = {
    "p_h": "ph",
    "t_h": "th",
    "k_h": "kh",
    "ts_h": "tsh",
    r"ts\_h": r"ts\h",
    "ts`_h": "ts`h",
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
        elif source_system == "xsampa":
            alias = STANDARD_XSAMPA_TO_SYNTHV_LEGACY.get(token)
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
    # Expand Mandarin triphthongs into target-supported phone sequences first.
    "uaI": (("u", "a", "I"), ("w", "a", "I"), ("u", "a", "i"), ("w", "a", "i"), ("uai",), ("uaI",)),
    "uei": (("u", "e", "i"), ("w", "e", "i"), ("uei",), ("ueI",)),
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

def _map_rhoticized_at_pair(valid: set[str]) -> list[str] | None:
    """Map the expanded at-vowel plus rhotic pair through known target aliases."""
    rhotic = "r" + chr(92)

    # First apply the curated standard-X-SAMPA bridge. For SynthV's newer
    # simplified Mandarin inventory, this maps '@' to 'e/' and 'r\\' to
    # 'iz' when those target-native labels are present. This is the key fallback
    # when the raw X-SAMPA rhotic symbol itself is not part of the model inventory.
    vowel_mapping = _map_xsampa_mandarin("@", valid, "xsampa")
    if vowel_mapping is None:
        for candidate in VOCALOID_MANDARIN_PHONE_CANDIDATES.get("@", ()):
            if len(candidate) == 1 and all(phone in valid for phone in candidate):
                vowel_mapping = list(candidate)
                break
    rhotic_mapping = _map_xsampa_mandarin(rhotic, valid, "xsampa")
    if rhotic_mapping is None:
        rhotic_aliases = (rhotic, "r", "rr", "-r", ":r" + chr(92) + chr(96), "ɹ", "ɻ", "R", "R" + chr(92))
        target_rhotic = next((phone for phone in rhotic_aliases if phone in valid), None)
        if target_rhotic is not None:
            rhotic_mapping = [target_rhotic]
    if vowel_mapping is not None and rhotic_mapping is not None:
        return [*vowel_mapping, *rhotic_mapping]

    # Prefer curated multi-phone representations supported by the target.
    for candidate in VOCALOID_MANDARIN_PHONE_CANDIDATES.get("@" + chr(96), ()):
        if len(candidate) == 2 and all(phone in valid for phone in candidate):
            return list(candidate)
    if "@" in valid and "rr" in valid:
        return ["@", "rr"]
    if "@" in valid and "-r" in valid:
        return ["@", "-r"]

    # A target-native erhua phone is safer than dropping the rhotic component.
    if "er" in valid:
        return ["er"]
    if "3" in valid:
        return ["3"]
    return None


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
    # Some VOCALOID exports percent-encode literal backslashes in phone names.
    # Normalize every occurrence before mapping, independent of hex-letter case.
    stem = re.sub(r"%5c", lambda _match: chr(92), stem, flags=re.IGNORECASE)
    # Expand only a standalone rhoticized at-vowel token. The inserted space
    # makes the vowel and rhotic component two ordered phones; embedded substrings
    # in other symbols are not modified.
    stem = re.sub(r"(?<!\S)@" + chr(96) + r"(?!\S)", lambda _match: "@" + " r" + chr(92), stem)
    raw_phones = stem.split()
    if not raw_phones:
        raise ValueError(f"{Path(filename).name}: filename contains no space-separated phonemes")

    phones = ["sil"]
    consumed_tokens: set[int] = set()
    for index, token in enumerate(raw_phones):
        if index in consumed_tokens:
            continue
        if token.casefold() == "sil":
            phones.append("sil")
            continue

        # For the rhoticized at-vowel, prefer the explicit X-SAMPA sequence
        # @ + r\ when supported, otherwise use the compatible 3 phone.
        # Consume an expanded at-vowel + rhotic pair together before mapping
        # individual symbols. Exact target phones win, then curated alternatives.
        if (
            token == "@"
            and index + 1 < len(raw_phones)
            and raw_phones[index + 1] == "r" + chr(92)
        ):
            converted_pair = _map_rhoticized_at_pair(valid)
            if converted_pair is not None:
                phones.extend(converted_pair)
                consumed_tokens.add(index + 1)
                continue

        if token == "@" + chr(96) and token not in valid:
            rhotic = "r" + chr(92)
            if "@" in valid and rhotic in valid:
                phones.extend(("@", rhotic))
                continue
            if "3" in valid:
                phones.append("3")
                continue
            # SynthV's simplified Mandarin inventory can expose erhua as the
            # dedicated Mandarin phone "er" without any separate rhotic symbol.
            if "er" in valid:
                phones.append("er")
                continue
            # Fall back only to curated, inventory-checked Mandarin erhua
            # representations. This also covers callers that do not specify
            # source_system; it never splits an unknown token heuristically.
            converted = _map_vocaloid_mandarin(token, valid)
            if converted is not None and converted != [token]:
                phones.extend(converted)
                continue

        # Force known Mandarin triphthongs to the explicit inventory-aware
        # phone sequence before any exact-token pass-through can swallow them.
        if token in {"uaI", "uei"}:
            converted = _map_vocaloid_mandarin(token, valid)
            if converted is not None and converted != [token]:
                phones.extend(converted)
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
            if _uses_synthv_simple_mandarin(valid) and (
                token in VOCALOID_MANDARIN_COMPOSITES
                or token in VOCALOID_MANDARIN_PHONE_CANDIDATES
            ):
                # Explicit curated Mandarin aliases such as aI/AU/@U/iAU/i@U;
                # this is table-driven mapping, never character-by-character guessing.
                converted = _map_vocaloid_mandarin(token, valid)
                if converted is not None:
                    phones.extend(converted)
                    continue
        if source_system == "vocaloid-mandarin":
            converted = _map_vocaloid_mandarin(token, valid)
            if converted is not None and (converted != [token] or token not in valid):
                phones.extend(converted)
                continue
            if converted is None:
                # VOCALOID's phone spellings include dialect-specific aliases,
                # but some source filenames also use canonical Mandarin X-SAMPA
                # tokens not duplicated in that alias table. Use only the explicit
                # standard X-SAMPA bridge here; never fall back to character splitting.
                converted = _map_xsampa_mandarin(token, valid, "xsampa")
                if converted is not None:
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


def is_quality_analysis_cache_wav(path: str | Path) -> bool:
    """Return true for the standalone VOCALOID vqm quality-analysis cache take."""
    stem = Path(path).stem
    if stem.endswith(("_0", "_1")):
        stem = stem[:-2]
    return stem.casefold() == "vqm"


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
    # Case-insensitive extension matching prevents .WAV/.Wav files from being
    # silently missed when a Windows dataset is processed on Linux.
    wavs = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.casefold() == ".wav" and not is_quality_analysis_cache_wav(path)),
        key=lambda path: path.name.casefold(),
    )
    if not wavs:
        raise ValueError(f"no .wav files found in {directory}")
    recordings = []
    for wav in wavs:
        try:
            phones = parse_filename_phones(wav.name, phone_table, source_system=source_system)
        except ValueError as error:
            if unmatched is None:
                raise
            message = str(error)
            token_match = re.search(r"unknown phone ['\"](.+?)['\"]", message)
            if token_match is None:
                # Preserve the blocking token for errors such as a rhoticized
                # vowel that cannot be represented by the selected inventory.
                token_match = re.search(r": ['\"]([^'\"]+)['\"]", message)

            source_token = token_match.group(1) if token_match else ""
            # If the generated rhotic component is the blocker, report the
            # original source token (@ plus grave accent), not its escaped
            # Python repr (which doubles the backslash in CSV summaries).
            source_stem = wav.stem
            if source_stem.endswith(("_0", "_1")):
                source_stem = source_stem[:-2]
            if (
                source_token == "r" + chr(92) * 2
                and ("@" + chr(96)) in source_stem.split()
            ):
                source_token = "@" + chr(96)
            unmatched.append({
                "filename": wav.name,
                "category": "phone_mapping",
                "source_token": source_token,
                "error": message,
            })
            continue
        try:
            # Decode the file, not just its header: truncated/corrupt WAV payloads
            # must be reported as bad files, while a valid unvoiced consonant WAV
            # remains a valid recording regardless of whether Praat finds F0.
            decoded, rate = read_wav_frames(wav)
            frames = int(decoded.shape[0])
            if rate <= 0 or frames <= 0 or decoded.shape[1] <= 0:
                raise ValueError("audio has no samples or an invalid sample rate")
        except (OSError, RuntimeError, ValueError) as error:
            message = f"{wav.name}: unreadable or damaged WAV: {error}"
            if unmatched is None:
                raise ValueError(message) from error
            unmatched.append({"filename": wav.name, "category": "wav_decode", "error": message})
            continue
        total_ticks = round(frames * TICKS_PER_SECOND / rate)
        try:
            labels = labels_from_phones(phones, total_ticks)
        except ValueError as error:
            message = f"{wav.name}: {error}"
            if unmatched is None:
                raise ValueError(message) from error
            unmatched.append({"filename": wav.name, "category": "label_generation", "error": message})
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
