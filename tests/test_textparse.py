from receipt_reader.engines.easyocr_engine import _join_rtl
from receipt_reader.textparse import TextLine, parse_lines, repair_rtl_numbers

from .conftest import IBAN_MELLAT, IBAN_MELLI


def _parse(*texts, source="printed"):
    cands, hints = parse_lines([TextLine(t, source=source) for t in texts], "t")
    return cands, hints


def by(cands, field):
    return [c for c in cands if c.field == field]


def test_repair_rtl_split_numbers():
    assert repair_rtl_numbers("مبلغ: ریال 500,000,000 ,4") == "مبلغ: ریال 4,500,000,000"
    assert repair_rtl_numbers("000,000 ,500 ,4") == "4,500,000,000"
    assert repair_rtl_numbers("4,500,000, 000") == "4,500,000,000"
    assert repair_rtl_numbers("11: 42") == "11:42"
    assert repair_rtl_numbers("450,000,000 ریال") == "450,000,000 ریال"  # untouched


def test_easyocr_join_keeps_numbers_ltr():
    assert _join_rtl(["مبلغ:", "000,000", "4,500,", "ریال"]) == "مبلغ: 4,500,000,000 ریال"


def test_amount_with_label_and_currency():
    c, _ = _parse("مبلغ: ۴۵۰٬۰۰۰٬۰۰۰ ریال")
    a = by(c, "amount")
    assert [x.value for x in a] == [450_000_000] and a[0].currency == "rial"


def test_fee_and_balance_are_not_amounts():
    c, _ = _parse("کارمزد: 5,000 ریال", "مانده حساب: 1,250,000,000 ریال")
    assert by(c, "amount") == []


def test_amount_words_line():
    c, _ = _parse("مبلغ به حروف: چهارصد و پنجاه میلیون ریال")
    w = [x for x in by(c, "amount") if x.source == "words"]
    assert w and w[0].value == 450_000_000 and w[0].currency == "rial"


def test_iban_roles():
    c, _ = _parse(f"از حساب: {IBAN_MELLI}", f"به حساب شبا: {IBAN_MELLAT}")
    roles = {x.value: x.role for x in by(c, "iban")}
    assert roles == {IBAN_MELLI: "source", IBAN_MELLAT: "dest"}


def test_label_on_previous_line():
    c, _ = _parse("شماره پیگیری:", "140508241234567")
    t = by(c, "tracking_number")
    assert t and t[0].value == "140508241234567" and t[0].label == "شماره پیگیری"


def test_tracking_label_misread_is_canonicalized():
    c, _ = _parse("شماره بیگیری: 1234567890")
    assert by(c, "tracking_number")[0].label == "شماره پیگیری"


def test_account_number_not_tracking():
    c, _ = _parse("شماره حساب: 1234567890123")
    assert by(c, "tracking_number") == [] and by(c, "amount") == []


def test_date_and_time_on_separate_lines():
    c, _ = _parse("تاریخ: ۱۴۰۵/۰۸/۲۴", "ساعت: ۱۱:۴۲")
    assert by(c, "paid_at")[0].value == "1405/08/24 11:42"


def test_status_and_type_votes():
    _, h = _parse("انتقال پایا", "وضعیت: ناموفق")
    assert h.transfer_votes == {"paya": 1} and h.status_votes == {"failed": 1}


def test_beneficiary_name():
    c, _ = _parse("نام ذینفع: علی کریمی")
    assert by(c, "receiver_name")[0].value == "علی کریمی"


# ---- regressions found on the owner's real receipts ----
def test_no_digit_fragments_from_unreadable_tokens():
    c, _ = _parse("شماره ارجاع 13??517?461208833?9")
    assert by(c, "tracking_number") == []


def test_inherited_amount_label_needs_a_grouped_amount():
    c, _ = _parse("مبلغ انتقال", "198877665544 شماره ارجاع")
    assert by(c, "amount") == []
    c, _ = _parse("مبلغ انتقال", "7,315,000 ریال")
    assert [x.value for x in by(c, "amount")] == [7_315_000]


def test_leading_zero_is_not_an_amount():
    c, _ = _parse("مبلغ انتقال 09,000 ریال")
    assert by(c, "amount") == []


def test_bare_holder_label_without_section_is_not_assigned():
    c, _ = _parse("به نام ممدر کوثه")
    assert by(c, "receiver_name") == [] and by(c, "sender_name") == []
    c, _ = _parse("دارنده کارت کاوه صادقی")
    assert by(c, "receiver_name")[0].value == "کاوه صادقی"


def test_card_last4_survives_an_unreadable_group():
    c, _ = _parse("کارت مقصد 86-62??xX-XXXX-3307")
    assert [x.value for x in by(c, "receiver_card")] == ["XXXX-3307"]


def test_misread_label_line_does_not_inherit_previous_label():
    # «ارجاع» misread as «ارچاع»: the reference number must not become the «پیگیری» number
    c, _ = _parse("شماره پیگیری", "71234598765 شماره ارچاع")
    assert by(c, "tracking_number") == []
    c, _ = _parse("شماره پیگیری:", "381946")
    assert [x.value for x in by(c, "tracking_number")] == ["381946"]
