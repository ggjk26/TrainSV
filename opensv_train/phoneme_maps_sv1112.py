"""Legacy Synthesizer V Studio 1.11.2 phoneme maps.

Keys follow the X-SAMPA column on Dreamtonics' phoneme-change page; values follow
its Old column, i.e. the pre-2.3 phoneme inventory. Results are checked against
the selected voice's actual phone inventory before being emitted.
"""

# Mandarin / Chinese: 55 official X-SAMPA entries -> Old labels.
SV1112_CHINESE_XSAMPA_TO_OLD: dict[str, tuple[str, ...]] = {
    "a": ("a",), "A": ("A",), "O": ("o",), "@": ("@",), "e": ("e",),
    "7": ("7",), "U": ("U",), "u": ("u",), "i": ("i",),
    "r\\": ("i\\",), "y": ("y",), "au": ("AU",), "ou": ("@U",),
    "ia": ("ia",), "iA": ("iA",), "iau": ("iAU",), "ie": ("ie",),
    "iE": ("iE",), "iU": ("iU",), "jou": ("i@U",), "y{": ("y{",),
    "yE": ("yE",), "ua": ("ua",), "uA": ("uA",), "u@": ("u@",),
    "ue": ("ue",), "uo": ("uo",), ":\\i": (":\\i",), ":n": (":n",),
    "N": ("N",), "p": ("p",), "ph": ("ph",), "t": ("t",),
    "th": ("th",), "k": ("k",), "kh": ("kh",), "ts\\": ("ts\\",),
    "ts": ("ts",), "tsh": ("tsh",), "x": ("x",), "f": ("f",),
    "s": ("s",), "m": ("m",), "n": ("n",), "l": ("l",), "w": ("w",), "j": ("j",),
}
# Symbols containing a grave accent are built from character codes to keep this
# mapping source free of delimiter ambiguity.
SV1112_CHINESE_XSAMPA_TO_OLD.update({
    ":r" + chr(92) + chr(96): ("i" + chr(96),),
    "r" + chr(92) + chr(96): ("r" + chr(92) + chr(96),),
    "ts" + chr(92): ("ts" + chr(92),),
    "ts" + chr(92) + "h": ("ts" + chr(92) + "h",),
    "ts" + chr(96): ("ts" + chr(96),),
    "ts" + chr(96) + "h": ("ts" + chr(96) + "h",),
    "s" + chr(92): ("s" + chr(92),),
    "s" + chr(96): ("s" + chr(96),),
    "z" + chr(96): ("z" + chr(96),),
})

# Explicit source aliases used by common VOCALOID Mandarin filename dictionaries.
SV1112_VOCALOID_MANDARIN_ALIASES: dict[str, tuple[tuple[str, ...], ...]] = {
    "@_n": (("@", ":n"),), "a_n": (("a", ":n"),), "i_n": (("i", ":n"),),
    "iE_n": (("iE", ":n"),), "ua_n": (("u", "a", ":n"),),
    "u@_n": (("u", "@", ":n"),), "y_n": (("y", ":n"),),
    "y{_n": (("y{", ":n"),),
    "AN": (("A", "N"),), "@N": (("@", "N"),), "iN": (("i", "N"),),
    "iAN": (("iA", "N"),), "uAN": (("u", "A", "N"),),
    "u@N": (("u", "@", "N"),), "UN": (("U", "N"),), "iUN": (("iU", "N"),),
    "aI": (("a", "i"),), "ei": (("e", "i"),),
    "uaI": (("u", "a", "i"),), "uei": (("u", "e", "i"),),
}
# VOCALOID 6 source spellings for aspirated phones differ from the official X-SAMPA names.
SV1112_VOCALOID_MANDARIN_ALIASES.update({
    "p_h": (("ph",),),
    "t_h": (("th",),),
    "k_h": (("kh",),),
    "ts_h": (("tsh",),),
})
SV1112_VOCALOID_MANDARIN_ALIASES["ts" + chr(92) + "_h"] = (("ts" + chr(92) + "h",),)
SV1112_VOCALOID_MANDARIN_ALIASES["ts" + chr(96) + "_h"] = (("ts" + chr(96) + "h",),)

# In VOCALOID filenames, at-vowel + grave denotes Mandarin erhua.
SV1112_VOCALOID_MANDARIN_ALIASES["@" + chr(96)] = (("i" + chr(92),), ("i" + chr(96),))

# Cantonese: source X-SAMPA -> Old labels. The old table uses :i for both
# source coda symbols :i and :y; the newer phoneme inventory splits them.
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
    # Syllabic nasals and rime codas
    "m=": ("m=",), "N=": ("N=",),
    ":i": (":i",), ":y": (":i",), ":u": (":u",), ":m": (":m",),
    ":n": (":n",), ":N": (":N",), ":p}": (":p_}",), ":t}": (":t_}",),
    ":k}": (":k_}",),
}

SV1112_XSAMPA_MAPS = {
    "sv1112-chinese": SV1112_CHINESE_XSAMPA_TO_OLD,
    "sv1112-cantonese": SV1112_CANTONESE_XSAMPA_TO_OLD,
}


def map_sv1112_phone(token: str, valid: set[str], language_mode: str) -> list[str] | None:
    """Map one source symbol to an Old label if the selected target supports it."""
    table = SV1112_XSAMPA_MAPS.get(language_mode)
    if table is None:
        return None

    if language_mode == "sv1112-chinese":
        for candidate in SV1112_VOCALOID_MANDARIN_ALIASES.get(token, ()):
            if candidate and all(phone in valid for phone in candidate):
                return list(candidate)

    mapped = table.get(token)
    if mapped is not None:
        return list(mapped) if all(phone in valid for phone in mapped) else None

    # A documented Old label may pass through only when present in the target.
    documented_old = {phone for values in table.values() for phone in values}
    if token in documented_old and token in valid:
        return [token]
    return None
