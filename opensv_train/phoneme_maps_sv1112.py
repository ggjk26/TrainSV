"""Legacy Synthesizer V Studio 1.11.2 Chinese phoneme maps.

The mapping direction is Dreamtonics' published X-SAMPA column -> Old column.
This targets SV Studio 1.11.2, not the simplified inventory introduced in 2.3 beta.
Every result must exist in the selected target voice's phone inventory.
"""

# Mandarin / Chinese: X-SAMPA spelling -> pre-2.3 SynthV Old spelling.
SV1112_CHINESE_XSAMPA_TO_OLD: dict[str, tuple[str, ...]] = {
    # Vowels and rimes
    "a": ("a",), "A": ("A",), "O": ("o",), "@": ("@",),
    "e": ("e",), "7": ("7",), "U": ("U",), "u": ("u",), "i": ("i",),
    "r\\": ("i\\",), "y": ("y",),
    "au": ("AU",), "ou": ("@U",), "ia": ("ia",), "iA": ("iA",),
    "iau": ("iAU",), "ie": ("ie",), "iE": ("iE",), "iU": ("iU",),
    "jou": ("i@U",),
    # Initial consonants; do not rewrite them to new b/d/g/z/c labels.
    "ts": ("ts",), "tsh": ("tsh",), "f": ("f",), "h": ("h",),
    "s": ("s",), "l": ("l",), "m": ("m",), "n": ("n",), "N": ("N",),
    "w": ("w",), "j": ("j",), "p": ("p",), "ph": ("ph",),
    "t": ("t",), "th": ("th",), "k": ("k",), "kh": ("kh",),
    "kw": ("kw",), "kwh": ("kwh",),
    # X-SAMPA coda and aspiration labels.
    ":n": (":n",),
    "p_h": ("ph",), "t_h": ("th",), "k_h": ("kh",), "ts_h": ("tsh",),
}
# Symbols that include a grave accent are built explicitly to avoid confusing
# it with punctuation in this source file.
SV1112_CHINESE_XSAMPA_TO_OLD.update({
    ":r" + chr(92) + chr(96): ("i" + chr(96),),
    "ts" + chr(92): ("ts" + chr(92),),
    "ts" + chr(92) + "h": ("ts" + chr(92) + "h",),
    "ts" + chr(96): ("ts" + chr(96),),
    "ts" + chr(96) + "h": ("ts" + chr(96) + "h",),
    "s" + chr(92): ("s" + chr(92),),
    "s" + chr(96): ("s" + chr(96),),
    "z" + chr(96): ("z" + chr(96),),
    "ts" + chr(92) + "_h": ("ts" + chr(92) + "h",),
    "ts" + chr(96) + "_h": ("ts" + chr(96) + "h",),
})

# Explicit source-specific aliases used by common VOCALOID Mandarin filename dictionaries.
# Candidate order is intentional and is accepted only if all output phones exist.
SV1112_VOCALOID_MANDARIN_ALIASES: dict[str, tuple[tuple[str, ...], ...]] = {
    "@_n": (("@", ":n"),),
    "a_n": (("a", ":n"),),
    "i_n": (("i", ":n"),),
    "iE_n": (("iE", ":n"),),
    "ua_n": (("u", "a", ":n"),),
    "u@_n": (("u", "@", ":n"),),
    "y_n": (("y", ":n"),),
    "y{_n": (("y", "e", ":n"), ("y", ":n")),
    "AN": (("A", "N"),),
    "@N": (("@", "N"),),
    "iN": (("i", "N"),),
    "iAN": (("iA", "N"),),
    "uAN": (("u", "A", "N"),),
    "u@N": (("u", "@", "N"),),
    "UN": (("U", "N"),),
    "iUN": (("iU", "N"),),
    "aI": (("a", "i"),),
    "ei": (("e", "i"),),
    "uaI": (("u", "a", "i"),),
    "uei": (("u", "e", "i"),),
}
SV1112_VOCALOID_MANDARIN_ALIASES["@" + chr(96)] = (("i" + chr(92),), ("i" + chr(96),))

# Cantonese: X-SAMPA column -> Old phoneme label.
SV1112_CANTONESE_XSAMPA_TO_OLD: dict[str, tuple[str, ...]] = {
    # Initials
    "ts": ("ts",), "tsh": ("tsh",), "f": ("f",), "h": ("h",), "s": ("s",),
    "l": ("l",), "m": ("m",), "n": ("n",), "N": ("N",), "w": ("w",), "j": ("j",),
    "p": ("p",), "ph": ("ph",), "t": ("t",), "th": ("th",),
    "k": ("k",), "kh": ("kh",), "kw": ("kw",), "kwh": ("kwh",),
    # Vowels
    "a": ("a",), "6": ("6",), "E": ("E",), "e": ("e",), "i": ("i",),
    "I": ("I",), "O": ("O",), "o": ("o",), "u": ("u",), "U": ("U",),
    "9": ("9",), "8": ("8",), "y": ("y",),
    # Syllabic nasals and rhyme codas
    "m=": ("m=",), "N=": ("N=",),
    ":i": (":i",), ":y": (":y",), ":u": (":u",), ":m": (":m",),
    ":n": (":n",), ":N": (":N",), ":p}": (":p}",), ":t}": (":t}",), ":k}": (":k}",),
}

SV1112_XSAMPA_MAPS = {
    "sv1112-chinese": SV1112_CHINESE_XSAMPA_TO_OLD,
    "sv1112-cantonese": SV1112_CANTONESE_XSAMPA_TO_OLD,
}


def map_sv1112_phone(token: str, valid: set[str], language_mode: str) -> list[str] | None:
    """Map one source phone to an old 1.11.2 label if the target supports it."""
    table = SV1112_XSAMPA_MAPS.get(language_mode)
    if table is None:
        return None

    # Source-specific VOCALOID aliases apply only to Mandarin mode.
    if language_mode == "sv1112-chinese":
        for candidate in SV1112_VOCALOID_MANDARIN_ALIASES.get(token, ()):
            if candidate and all(phone in valid for phone in candidate):
                return list(candidate)

    mapped = table.get(token)
    if mapped is not None:
        return list(mapped) if all(phone in valid for phone in mapped) else None

    # Some source filenames already use a documented target Old spelling.
    # Exact pass-through is allowed only when the selected voice contains it.
    documented_old = {phone for values in table.values() for phone in values}
    if token in documented_old and token in valid:
        return [token]
    return None
