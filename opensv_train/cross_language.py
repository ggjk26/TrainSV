"""Monolingual cross-language phoneme coverage diagnostics.

This audit deliberately treats the training corpus as one source language.
A shared unified phoneme is only a representation-level overlap; it is not
target-language acoustic supervision.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


_LANGUAGE_ALIASES = {
    "zh": "mandarin",
    "ja": "japanese",
    "en": "english",
    "yue": "cantonese",
    "es": "spanish",
}


def _find_phone_set(phone_sets: list, language: str) -> int:
    normalized = language.casefold()
    matches = [
        index for index, phone_set in enumerate(phone_sets)
        if normalized in phone_set.name.casefold()
    ]
    if len(matches) != 1:
        names = ", ".join(phone_set.name for phone_set in phone_sets)
        raise ValueError(
            f"language {language!r} matches {len(matches)} phone sets ({names})"
        )
    return matches[0]


def _token_language(token: str, default_language: str) -> tuple[str, str]:
    if "/" not in token:
        return default_language, token
    prefix, symbol = token.split("/", 1)
    return _LANGUAGE_ALIASES.get(prefix.casefold(), prefix), symbol


def audit_cross_language_coverage(
    dataset: str | Path,
    phone_sets: list,
    unified_symbols: list[str],
    source_language: str,
    target_language: str,
) -> dict[str, Any]:
    """Audit one-language corpus evidence against a target phone inventory.

    Unprefixed tokens use source_language. Explicitly prefixed source-language
    tokens (for example zh/k) are accepted. Tokens tagged as another language
    are reported as mixed-language input and excluded from source statistics.
    They are never merged into the monolingual source-language counts.
    """
    source_index = _find_phone_set(phone_sets, source_language)
    target_index = _find_phone_set(phone_sets, target_language)
    source_set, target_set = phone_sets[source_index], phone_sets[target_index]
    unified_inventory = set(unified_symbols)

    source_symbol_occurrences: Counter[str] = Counter()
    source_symbol_recordings: dict[str, set[str]] = defaultdict(set)
    unit_occurrences: Counter[str] = Counter()
    unit_recordings: dict[str, set[str]] = defaultdict(set)
    transitions: Counter[tuple[str, str]] = Counter()
    transition_recordings: dict[tuple[str, str], set[str]] = defaultdict(set)
    invalid: list[dict[str, Any]] = []
    mixed: list[dict[str, Any]] = []
    root = Path(dataset)

    with (root / "transcriptions.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        for row_number, row in enumerate(csv.DictReader(stream), start=2):
            name = (row.get("name") or "").strip()
            mapped: list[str | None] = []
            for token in (row.get("ph_seq") or "").split():
                token_language, symbol = _token_language(token, source_language)
                try:
                    language_index = _find_phone_set(phone_sets, token_language)
                except ValueError:
                    invalid.append({
                        "row": row_number, "name": name, "phone": token,
                        "reason": f"unknown language prefix: {token_language}",
                    })
                    mapped.append(None)
                    continue

                if language_index != source_index:
                    mixed.append({
                        "row": row_number, "name": name, "phone": token,
                        "language": phone_sets[language_index].name,
                    })
                    mapped.append(None)
                    continue

                try:
                    position = source_set.symbols.index(symbol)
                    unified = source_set.unified[position]
                    if unified not in unified_inventory:
                        raise ValueError("unified phone is absent from model inventory")
                except (ValueError, IndexError):
                    invalid.append({
                        "row": row_number, "name": name, "phone": token,
                        "reason": "unknown source-language phone",
                    })
                    mapped.append(None)
                    continue

                source_symbol_occurrences[symbol] += 1
                source_symbol_recordings[symbol].add(name)
                unit_occurrences[unified] += 1
                unit_recordings[unified].add(name)
                mapped.append(unified)

            for left, right in zip(mapped, mapped[1:]):
                if left is None or right is None or left in ("sil", "pau") or right in ("sil", "pau"):
                    continue
                pair = (left, right)
                transitions[pair] += 1
                transition_recordings[pair].add(name)

    target_rows = []
    for position, phone in enumerate(target_set.symbols):
        unified = target_set.unified[position]
        source_equivalents = [
            source_set.symbols[index]
            for index, candidate in enumerate(source_set.unified)
            if candidate == unified
        ]
        recordings = len(unit_recordings[unified])
        occurrences = unit_occurrences[unified]
        if source_index == target_index and source_symbol_occurrences[phone]:
            status = "target_supervised"
        elif not source_equivalents:
            status = "no_shared_unit"
        elif not recordings:
            status = "shared_unit_not_observed"
        else:
            status = "shared_unit_only"

        target_rows.append({
            "phone": phone,
            "category": target_set.categories[position],
            "unified": unified,
            "source_equivalent_phones": source_equivalents,
            "source_unit_occurrences": occurrences,
            "source_unit_recordings": recordings,
            "target_language_supervision": status == "target_supervised",
            "status": status,
        })
    target_rows.sort(
        key=lambda row: (
            row["source_unit_recordings"], row["category"], row["phone"]
        )
    )

    return {
        "dataset": str(root),
        "source_language": source_set.name,
        "target_language": target_set.name,
        "monolingual_source_corpus": not mixed,
        "summary": {
            "source_phone_types_observed": len(source_symbol_occurrences),
            "source_phone_occurrences": sum(source_symbol_occurrences.values()),
            "target_phone_count": len(target_rows),
            "target_phones_with_shared_unit": sum(
                bool(row["source_equivalent_phones"]) for row in target_rows
            ),
            "target_phones_without_shared_unit": sum(
                not row["source_equivalent_phones"] for row in target_rows
            ),
            "target_phones_with_source_data_on_shared_unit": sum(
                row["status"] in ("shared_unit_only", "target_supervised")
                for row in target_rows
            ),
            "target_language_supervised_phones": sum(
                row["target_language_supervision"] for row in target_rows
            ),
            "mixed_language_tokens": len(mixed),
            "unmapped_source_tokens": len(invalid),
            "source_transition_types": len(transitions),
        },
        "target_phones": target_rows,
        "mixed_language_tokens": mixed[:100],
        "unmapped_source_tokens": invalid[:100],
        "source_transitions": [
            {
                "from": left,
                "to": right,
                "occurrences": count,
                "recordings": len(transition_recordings[(left, right)]),
            }
            for (left, right), count in sorted(transitions.items())
        ],
        "limitations": [
            "This audit assumes one source language per training corpus; foreign-language-prefixed tokens are excluded.",
            "A shared unified phone is only a common model input representation, not target-language acoustic supervision.",
            "With a Mandarin-only corpus, Japanese-specific language-conditioned pronunciation, mora timing, and phonotactics receive no target-language training signal.",
            "The audit never substitutes Mandarin phones for Japanese phones or claims that shared units are fully covered.",
        ],
    }
