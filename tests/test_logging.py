from receipt_reader.logging_utils import mask_sensitive

from .conftest import IBAN_MELLAT


def test_iban_masked():
    out = mask_sensitive(f"dest={IBAN_MELLAT}")
    assert IBAN_MELLAT not in out and out.endswith(IBAN_MELLAT[-4:])


def test_card_and_national_id_masked():
    out = mask_sensitive("card 6037-9918-1234-5678 nid ۰۰۱۲۳۴۵۶۷۸")
    assert "6037" not in out and "5678" in out
    assert "0012345678" not in out and out.endswith("5678")


def test_amounts_and_dates_untouched():
    s = "amount 450,000,000 date 1405/08/24"
    assert mask_sensitive(s) == s
