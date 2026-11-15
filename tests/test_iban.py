from receipt_reader.iban import detect_bank_names, find_ibans, make_iban, mask, mod97_valid, parse_iban

from .conftest import IBAN_MELLAT, IBAN_MELLI


def test_known_valid_iban():
    # the example from the ECBS IBAN registry for Iran
    assert mod97_valid("IR062960000000100324200001")


def test_generated_ibans_are_valid():
    assert mod97_valid(IBAN_MELLAT)
    assert mod97_valid(IBAN_MELLI)


def test_single_digit_error_fails_checksum():
    bad = IBAN_MELLAT[:-1] + str((int(IBAN_MELLAT[-1]) + 1) % 10)
    assert not mod97_valid(bad)
    info = parse_iban(bad)
    assert info and info.checksum_valid is False


def test_transposition_fails_checksum():
    s = IBAN_MELLAT
    t = s[:20] + s[21] + s[20] + s[22:]
    if t != s:
        assert not mod97_valid(t)


def test_normalize_formats():
    raw24 = IBAN_MELLAT[2:]
    spaced = " ".join(IBAN_MELLAT[i:i + 4] for i in range(0, 26, 4))
    dashed = "-".join(IBAN_MELLAT[i:i + 4] for i in range(0, 26, 4))
    fa = IBAN_MELLAT.translate({ord(str(i)): "۰۱۲۳۴۵۶۷۸۹"[i] for i in range(10)})
    for raw in (IBAN_MELLAT, raw24, spaced, dashed, fa, IBAN_MELLAT.lower()):
        info = parse_iban(raw)
        assert info is not None, raw
        assert info.value == IBAN_MELLAT
        assert info.checksum_valid
        assert info.bank == "Mellat" and info.bank_code == "012"


def test_partial():
    for raw in ("IR**********8550", "...8550", "IR** **** **** 8550", "****8550"):
        info = parse_iban(raw)
        assert info is not None and info.partial, raw
        assert info.value == "...8550"
        assert info.checksum_valid is None


def test_bank_lookup_from_data_file():
    assert parse_iban(IBAN_MELLI).bank == "Melli"
    assert parse_iban(make_iban("0570000000000000000001")).bank == "Pasargad"
    assert "012" in detect_bank_names("بانک ملت شعبه مرکزی")


def test_find_in_text():
    found = find_ibans(f"به حساب شبا {IBAN_MELLAT} نزد بانک ملت")
    assert [i.value for _, i in found] == [IBAN_MELLAT]


def test_mask():
    assert mask(IBAN_MELLAT) == "*" * 22 + IBAN_MELLAT[-4:]
