"""Cross-language phoneme coverage diagnostics.

This module reports shared phonetic units and target-language gaps without
rewriting phones or silently substituting one language's pronunciation.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

def _find_phone_set(phone_sets: list, language: str) -> int:
    normalized = language.casefold()
    matches = [
        index for index, phone_set in enumerate(phone_sets)
        if normalized in phone_set.name.casefold()
    ]
    if len(matches) != 1:
        names = ", ".join(phone_set.name for phone_set in phone_sets)
        raise ValueError(f"language {language!r} matches {len(matches)} phone sets ({names})")
    return matches[0]


def audit_cross_language_coverage(
    dataset: str | Path,
    phone_sets: list,
    unified_symbols: list[str],
    source_language: str,
    target_language: str,
) -> dict[str, Any]:
    """Compare target-language phones with unified units in a training corpus.

    Unprefixed corpus tokens use source_language; tokens such as ja/k or zh/a
    explicitly select a language. Shared units indicate possible articulation
    transfer, not proof that language-specific pronunciation has been learned.
    """
    source_index = _find_phone_set(phone_sets, source_language)
    target_index = _find_phone_set(phone_sets, target_language)
    source_set, target_set = phone_sets[source_index], phone_sets[target_index]
    unified_index = {symbol: index for index, symbol in enumerate(unified_symbols)}
    aliases = {"zh": "mandarin", "ja": "japanese", "en": "english",
               "yue": "cantonese", "es": "spanish"}
    counts: Counter[str] = Counter()
    recordings: dict[str, set[str]] = defaultdict(set)
    transitions: Counter[tuple[str, str]] = Counter()
    transition_recordings: dict[tuple[str, str], set[str]] = defaultdict(set)
    invalid: list[dict[str, Any]] = []
    root = Path(dataset)

    with (root / "transcriptions.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row_number, row in enumerate(csv.DictReader(stream), start=2):
            name = (row.get("name") or "").strip()
            mapped: list[str | None] = []
            for token in (row.get("ph_seq") or "").split():
                language, symbol = source_language, token
                if "/" in token:
                    prefix, symbol = token.split("/", 1)
                    language = aliases.get(prefix, prefix)
                try:
                    language_index = _find_phone_set(phone_sets, language)
                    phone_set = phone_sets[language_index]
                    position = phone_set.symbols.index(symbol)
                    unified = phone_set.unified[position]
                    if unified not in unified_index:
                        raise ValueError("unified phone is absent from model inventory")
                except (ValueError, IndexError):
                    invalid.append({"row": row_number, "name": name, "phone": token})
                    mapped.append(None)
                    continue
                counts[unified] += 1
                recordings[unified].add(name)
                mapped.append(unified)
            for left, right in zip(mapped, mapped[1:]):
                if left is None or right is None or left in ("sil", "pau") or right in ("sil", "pau"):
                    continue
                pair = (left, right)
                transitions[pair] += 1
                transition_recordings[pair].add(name)

    source_coverage = {
        symbol: {"occurrences": counts[symbol], "recordings": len(recordings[symbol])}
        for symbol in counts
    }
    target_rows = []
    for position, phone in enumerate(target_set.symbols):
        unified = target_set.unified[position]
        coverage = source_coverage.get(unified, {"occurrences": 0, "recordings": 0})
        equivalents = [
            source_set.symbols[i]
            for i, candidate in enumerate(source_set.unified)
            if candidate == unified
        ]
        target_rows.append({
            "phone": phone,
            "category": target_set.categories[position],
            "unified": unified,
            "source_equivalent_phones": equivalents,
            "source_occurrences": coverage["occurrences"],
            "source_recordings": coverage["recordings"],
            "status": "covered" if coverage["recordings"] else "unseen",
        })
    target_rows.sort(key=lambda row: (row["source_recordings"], row["category"], row["phone"]))
    return {
        "dataset": str(root),
        "source_language": source_set.name,
        "target_language": target_set.name,
        "summary": {
            "source_phone_units": len(source_coverage),
            "target_phone_count": len(target_rows),
            "covered_target_phones": sum(row["status"] == "covered" for row in target_rows),
            "unseen_target_phones": sum(row["status"] == "unseen" for row in target_rows),
            "unmapped_source_tokens": len(invalid),
            "source_transition_types": len(transitions),
        },
        "target_phones": target_rows,
        "unmapped_source_tokens": invalid[:100],
        "source_transitions": [
            {"from": left, "to": right, "occurrences": count,
             "recordings": len(transition_recordings[(left, right)])}
            for (left, right), count in sorted(transitions.items())
        ],
        "limitations": [
            "Shared unified phones indicate phonetic overlap, not equivalent language-specific pronunciation.",
            "A monolingual corpus does not teach target-language timing, phonotactics, or acoustic realization.",
            "Unseen target phones are reported, never automatically replaced with a nearest source-language phone.",
        ],
    }
