from types import SimpleNamespace

from opensv_train.cross_language import audit_cross_language_coverage


def _phone_set(name, symbols, unified, categories=None):
    return SimpleNamespace(name=name, symbols=symbols, unified=unified,
                           categories=categories or ["consonant"] * len(symbols))


def test_cross_language_audit_uses_unified_overlap_and_keeps_unseen_phones(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "transcriptions.csv").write_text(
        "name,ph_seq,ph_dur\ntake1,k a n,0.1 0.2 0.1\ntake2,zh/k zh/a zh/n,0.1 0.2 0.1\n",
        encoding="utf-8",
    )
    mandarin = _phone_set("Mandarin", ["k", "a", "n"], ["K", "A", "N"],
                          ["consonant", "vowel", "consonant"])
    japanese = _phone_set("Japanese", ["k", "a", "N", "ts", "r"],
                          ["K", "A", "N", "TS", "R"],
                          ["consonant", "vowel", "consonant", "consonant", "consonant"])
    report = audit_cross_language_coverage(
        dataset, [mandarin, japanese], ["K", "A", "N", "TS", "R"], "mandarin", "japanese"
    )
    rows = {row["phone"]: row for row in report["target_phones"]}
    assert report["summary"]["covered_target_phones"] == 3
    assert report["summary"]["unseen_target_phones"] == 2
    assert rows["k"]["source_recordings"] == 2
    assert rows["ts"]["status"] == "unseen"
    assert rows["r"]["status"] == "unseen"
    assert report["summary"]["source_transition_types"] == 2


def test_cross_language_audit_respects_explicit_language_prefixes(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "transcriptions.csv").write_text(
        "name,ph_seq,ph_dur\ntake,ja/k ja/a,0.1 0.2\n", encoding="utf-8"
    )
    japanese = _phone_set("Japanese", ["k", "a"], ["K", "A"])
    mandarin = _phone_set("Mandarin", ["k", "a"], ["K", "A"])
    report = audit_cross_language_coverage(
        dataset, [mandarin, japanese], ["K", "A"], "mandarin", "japanese"
    )
    assert report["summary"]["covered_target_phones"] == 2


def test_cross_language_audit_reports_unknown_source_phone_without_guessing(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "transcriptions.csv").write_text(
        "name,ph_seq,ph_dur\ntake,unknown,0.1\n", encoding="utf-8"
    )
    source = _phone_set("Mandarin", ["a"], ["A"])
    target = _phone_set("Japanese", ["a"], ["A"])
    report = audit_cross_language_coverage(
        dataset, [source, target], ["A"], "mandarin", "japanese"
    )
    assert report["summary"]["unmapped_source_tokens"] == 1
    assert report["summary"]["covered_target_phones"] == 0
    assert report["unmapped_source_tokens"][0]["phone"] == "unknown"
