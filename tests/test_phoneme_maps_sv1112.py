from opensv_train.filename_labels import parse_filename_phones


def test_sv1112_chinese_uses_legacy_mandarin_labels():
    target = {"sil", "a", "A", "o", "@", "e", "7", "U", "u", "i", "i\\", "AU", "@U"}
    parsed = parse_filename_phones(
        "a A O @ e 7 U u i r\\ au ou_0.wav",
        target,
        source_system="sv1112-chinese",
    )
    assert parsed == [
        "sil", "a", "A", "o", "@", "e", "7", "U", "u", "i", "i\\", "AU", "@U", "sil"
    ]


def test_sv1112_cantonese_keeps_legacy_aspiration_and_vowel_contrasts():
    target = {
        "sil", "ts", "tsh", "p", "ph", "t", "th", "k", "kh", "kw", "kwh",
        "a", "6", "E", "e", "i", "I", "O", "o", "u", "U", "9", "8", "y",
        "m=", "N=", ":i", ":y", ":u", ":m", ":n", ":N", ":p}", ":t}", ":k}",
    }
    parsed = parse_filename_phones(
        "ts tsh p ph t th k kh kw kwh a 6 E e i I O o u U 9 8 y m= N= :i :y :u :m :n :N :p} :t} :k}_0.wav",
        target,
        source_system="sv1112-cantonese",
    )
    assert parsed[1:-1] == [
        "ts", "tsh", "p", "ph", "t", "th", "k", "kh", "kw", "kwh",
        "a", "6", "E", "e", "i", "I", "O", "o", "u", "U", "9", "8", "y",
        "m=", "N=", ":i", ":y", ":u", ":m", ":n", ":N", ":p}", ":t}", ":k}",
    ]


def test_sv1112_modes_do_not_guess_unknown_phones():
    import pytest

    with pytest.raises(ValueError, match="no verified mapping"):
        parse_filename_phones("unknown_0.wav", {"sil", "u", "n"}, source_system="sv1112-cantonese")
