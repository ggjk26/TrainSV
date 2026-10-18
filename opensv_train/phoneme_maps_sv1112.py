"""Synthesizer V Studio 1.11.2 Chinese phoneme labels.

These tables deliberately use the legacy ("Old") labels documented by Dreamtonics,
not the simplified labels introduced in Synthesizer V Studio 2.3.0 Beta 1.
Keys are X-SAMPA symbols; values are legacy SynthV labels. Mapping is explicit:
unknown symbols are not split heuristically.
"""

SV1112_CHINESE_XSAMPA_TO_LEGACY: dict[str, tuple[str, ...]] = {
    # Mandarin vowels and finals (official X-SAMPA -> pre-2.3 label).
    "a": ("a",), "A": ("A",), "O": ("o",), "@": ("@",),
    "e": ("e",), "7": ("7",), "U": ("U",), "u": ("u",), "i": ("i",),
    "r\\": ("i\\",), "y": ("y",),
    "au": ("AU",), "ou": ("@U",), "ia": ("ia",), "iA": ("iA",),
    "iau": ("iAU",), "ie": ("ie",), "iE": ("iE",), "iU": ("iU",),
    "jou": ("i@U",),
    # Common onsets/codas. Exact legacy labels are preserved.
    "p": ("p",), "p_h": ("ph",), "t": ("t",), "t_h": ("th",),
    "k": ("k",), "k_h": ("kh",), "ts": ("ts",), "ts_h": ("tsh",),
    "ts\\": ("ts\\",), "ts\\_h": ("ts\\h",),
    "f": ("f",), "h": ("h",), "s": ("s",),
    "s\\": ("s\\",), "m": ("m",), "n": ("n",),
    "N": ("N",), "l": ("l",), "w": ("w",), "j": ("j",),
    ":n": (":n",), ":\\i": (":\\i",),
}

SV1112_CANTONESE_XSAMPA_TO_LEGACY: dict[str, tuple[str, ...]] = {
    # Cantonese initials: aspiration contrasts are distinct legacy labels.
    "ts": ("ts",), "tsh": ("tsh",), "f": ("f",), "h": ("h",),
    "s": ("s",), "l": ("l",), "m": ("m",), "n": ("n",), "N": ("N",),
    "w": ("w",), "j": ("j",),
    "p": ("p",), "ph": ("ph",), "t": ("t",), "th": ("th",),
    "k": ("k",), "kh": ("kh",), "kw": ("kw",), "kwh": ("kwh",),
    # Cantonese vowels are not Mandarin vowels with alternate spellings.
    "a": ("a",), "6": ("6",), "E": ("E",), "e": ("e",),
    "i": ("i",), "I": ("I",), "O": ("O",), "o": ("o",),
    "u": ("u",), "U": ("U",), "9": ("9",), "8": ("8",), "y": ("y",),
    "m=": ("m=",), "N=": ("N=",),
    # Rime codas; keep the source's explicit coda symbol.
    ":i": (":i",), ":y": (":y",), ":u": (":u",), ":m": (":m",),
    ":n": (":n",), ":N": (":N",), ":p}": (":p}",),
    ":t}": (":t}",), ":k}": (":k}",),
}


# Symbols containing the X-SAMPA grave accent are built explicitly so the
# mapping remains readable without confusing the accent with punctuation.
SV1112_CHINESE_XSAMPA_TO_LEGACY[":r" + chr(92) + chr(96)] = ("i" + chr(96),)
SV1112_CHINESE_XSAMPA_TO_LEGACY["ts" + chr(96)] = ("ts" + chr(96),)
SV1112_CHINESE_XSAMPA_TO_LEGACY["ts" + chr(96) + "_h"] = ("ts" + chr(96) + "h",)
SV1112_CHINESE_XSAMPA_TO_LEGACY["s" + chr(96)] = ("s" + chr(96),)
SV1112_CHINESE_XSAMPA_TO_LEGACY["z" + chr(96)] = ("z" + chr(96),)

SV1112_LEGACY_XSAMPA_MAPS = {
    "sv1112-chinese": SV1112_CHINESE_XSAMPA_TO_LEGACY,
    "sv1112-cantonese": SV1112_CANTONESE_XSAMPA_TO_LEGACY,
}


def map_sv1112_xsampa(token: str, valid: set[str], language_mode: str) -> list[str] | None:
    """Map an X-SAMPA token to a legacy 1.11.2 label only if target supports it."""
    table = SV1112_LEGACY_XSAMPA_MAPS.get(language_mode)
    if table is None or token not in table:
        return None
    mapped = table[token]
    return list(mapped) if all(phone in valid for phone in mapped) else None
