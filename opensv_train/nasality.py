"""Frame selection for nasality-focused training terms (no PyTorch needed).

Nasal-sounding renders usually come from oral vowels picking up the nasal
coloring of their neighbours. These helpers mark the frames where a vowel should
be oral, so the training loss can emphasise them. They are heuristics over phone
categories, not a nasality classifier.
"""
from __future__ import annotations

import numpy as np

VOWEL_CATEGORIES = ("vowel", "diphthong")
NASAL_CATEGORIES = ("nasal",)


def oral_vowel_frames(categories: list[str], phone_frames: list[int]) -> np.ndarray:
    """Return 1.0 on frames of vowel phones with no directly adjacent nasal phone, else 0.0.

    ``categories`` holds one phone-set category per phone and ``phone_frames`` its
    duration in frames. A vowel next to a nasal consonant (a nasal coda, or the
    nasal of the following syllable) is left out because nasalization is expected
    there, so no loss term should push those frames towards "oral".
    """
    if len(categories) != len(phone_frames):
        raise ValueError("categories and phone_frames must have one entry per phone")
    if any(int(count) < 1 for count in phone_frames):
        raise ValueError("phone frames must be positive")
    mask = np.zeros(sum(int(count) for count in phone_frames), dtype=np.float32)
    position = 0
    for index, (category, count) in enumerate(zip(categories, phone_frames)):
        count = int(count)
        if category in VOWEL_CATEGORIES:
            before = categories[index - 1] if index > 0 else None
            after = categories[index + 1] if index + 1 < len(categories) else None
            if before not in NASAL_CATEGORIES and after not in NASAL_CATEGORIES:
                mask[position:position + count] = 1.0
        position += count
    return mask
