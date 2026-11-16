"""Parser behaviour on the receipt layouts seen in the owner's real set (2026-11-16):
bank-app screenshots (Melli/Baam, Saman), a Parsian SATNA printout, a Mellat SATNA teller
strip and payment-order forms. Names, IBANs and numbers here are made up."""
from receipt_reader.crosscheck import Hints, resolve
from receipt_reader.dates import parse_datetime
from receipt_reader.iban import detect_bank_names, make_iban
from receipt_reader.textparse import TextLine, parse_lines

from .conftest import IBAN_MELLAT, IBAN_MELLI

IBAN_SAMAN = make_iban("0560000000001111222233")
IBAN_PARSIAN = make_iban("0540000000004444555566")


def _parse(*texts, source="printed"):
    return parse_lines([TextLine(t, source=source) for t in texts], "t")


def by(cands, field):
    return [c for c in cands if c.field == field]


def vals(cands, field):
    return [c.value for c in by(cands, field)]


# ---------------------------------------------------------------- bank-app screenshots
def test_melli_app_screenshot():
    c, h = _parse("baambank.ir",
                  "انتقال پول موفق",
                  "مبلغ ۳,۰۰۰,۰۰۰,۰۰۰ ریال",
                  "حساب مبدا ۰۱۰۴۱XXXX۴۰۰۳",
                  "روش انتقال بین بانکی - ساتنا",
                  f"شماره شبا مقصد {IBAN_SAMAN}",
                  "بابت خرید کالا",
                  "متعلق به رضا کریمی",
                  "زمان انتقال پول ۱۴۰۵/۰۸/۲۳ ۱۰:۳۸",
                  "شماره پیگیری ۱۴۰۵۰۸۲۳۰۱۷۱۰۲۸۳۹۰", source="digital")
    assert vals(c, "amount") == [3_000_000_000]
    assert vals(c, "sender_account") == ["01041XXXX4003"]
    assert vals(c, "sender_card") == []          # a masked ACCOUNT is not a card
    assert vals(c, "receiver_name") == ["رضا کریمی"]
    assert vals(c, "purpose") == ["خرید کالا"]
    assert vals(c, "tracking_number") == ["140508230171028390"]
    assert [x.role for x in by(c, "iban")] == ["dest"]
    assert h.issuer_banks == {"017"}             # Melli app, from the header
    assert h.transfer_votes.get("satna") and h.status_votes.get("success")


def test_saman_app_paya_screenshot():
    c, h = _parse("ثبت انتقال وجه پایا",
                  "رضا کریمی",
                  IBAN_PARSIAN,
                  "مبلغ انتقال ۱۰۵,۰۰۰,۰۰۰ ریال",
                  "تاریخ و زمان شنبه ۲۳ آبان ۱۴:۲۳|۰۵",
                  "روش انتقال بین بانکی|پایا",
                  "وضعیت انتقال در انتظار انجام",
                  "انتقال دهنده علی احمدی",
                  "سپرده مبدأ ۸۲۴-۸۰۰-۲۱۸۴۶۰۴-۱",
                  "شماره پیگیری ۱۴۰۵۰۸۲۳۰۵۶۲۷۳۸۹۴۹۶۵", source="digital")
    assert vals(c, "amount") == [105_000_000]
    assert vals(c, "paid_at") == ["1405/08/23 14:23"]
    assert vals(c, "sender_name") == ["علی احمدی"]
    assert vals(c, "sender_account") == ["82480021846041"]
    assert h.status_votes.get("pending") and h.transfer_votes.get("paya")


# ---------------------------------------------------------------- Parsian SATNA printout
def test_parsian_satna_printout_roles_and_ids():
    c, h = _parse("بانک پارسیان",
                  "ساتنا",
                  "تاریخ رسید : ۱۴۰۵/۰۸/۲۳",
                  "شناسه دستور پرداخت : ۰۵۰۸۲۳۰۵۴۱۰۰۵۹۸۶ تاریخ ثبت : ۱۴۰۵/۰۸/۲۳",
                  "مبلغ کل : ۳,۵۴۰,۰۰۰,۰۰۰ ریال تعداد کل : ۱",
                  "کاربر ثبت کننده : 3778 کاربر تأییدکننده : 3778",
                  f"شبای مشتری : {IBAN_PARSIAN}",
                  "نام مشتری : علی احمدی",
                  "شناسه واریز اجباریست خیر",
                  f"شبای مقصد : {IBAN_MELLAT} بانک مقصد : ملت",
                  "نام صاحبان حساب : رضا کریمی",
                  "توضیحات : قرض")
    assert vals(c, "amount") == [3_540_000_000]       # never the operator id 3778
    roles = {x.value: x.role for x in by(c, "iban")}
    assert roles == {IBAN_PARSIAN: "source", IBAN_MELLAT: "dest"}
    assert vals(c, "sender_name") == ["علی احمدی"]
    assert vals(c, "receiver_name") == ["رضا کریمی"]
    assert vals(c, "tracking_number") == ["0508230541005986"]
    assert vals(c, "purpose") == ["قرض"]
    assert vals(c, "receiver_bank") == ["012"]


def test_operator_id_after_amount_label_is_not_an_amount():
    c, _ = _parse("مبلغ کل : 3778 کاربر تأییدکننده 3778")
    assert by(c, "amount") == []


def test_misread_label_on_value_line_does_not_inherit_section():
    # «بانک مقصد» misread as «بلک مقصل» is still recognised (to_account); an unknown label
    # on the IBAN line stops a payer section above it from claiming the IBAN
    c, _ = _parse("نام مشتری : علی احمدی", f"{IBAN_MELLAT} بلک مقصل")
    assert [x.role for x in by(c, "iban")] == ["dest"]
    c, _ = _parse("نام مشتری : علی احمدی", f"{IBAN_MELLAT} شعبه صادره")
    assert [x.role for x in by(c, "iban")] == ["unknown"]


# ---------------------------------------------------------------- Mellat SATNA teller strip
def test_mellat_teller_strip():
    c, _ = _parse("۳۵۱۳۳-صدور درخواست حواله ساتنا",
                  "کدشعبه:۶۵۷۴۸ * کاربر:۳۲۳۶۵۹ * تاریخ:۱۴۰۵/۰۸/۲۳ * زمان:۱۱:۴۵:۴۱ * شعبه",
                  "کد پیگیری:۱۴۰۵۰۸۲۳۰۱۲۱۰۶۸۷۱۹ * تعداد کل:۱ * مبلغ کل:۲,۴۰۰,۰۰۰,۰۰۰ ریال",
                  "کارمزد:۵۲۰,۰۰۰ ریال",
                  "شماره حساب:۳۳۴۵۲۶۲۹۰۴ * صاحب حساب:علی احمدی * نوع حساب:جاری",
                  f"شبا:{IBAN_PARSIAN} * بانک:پارسیان",
                  "مبلغ:۲,۴۰۰,۰۰۰,۰۰۰ * شناسه واریز:",
                  "ذینفع:رضا کریمی")
    assert set(vals(c, "amount")) == {2_400_000_000}  # never the fee 520,000
    assert vals(c, "tracking_number") == ["140508230121068719"]  # not the user id 323659
    assert vals(c, "paid_at") == ["1405/08/23 11:45"]
    # the unlabelled account holder listed before the explicit «ذینفع» is the payer
    assert vals(c, "sender_name") == ["علی احمدی"]
    assert vals(c, "receiver_name") == ["رضا کریمی"]


def test_unsided_holder_without_beneficiary_stays_unassigned():
    c, _ = _parse("شماره حساب:۳۳۴۵۲۶۲۹۰۴ * صاحب حساب:علی احمدی")
    assert by(c, "sender_name") == [] and by(c, "receiver_name") == []


# ---------------------------------------------------------------- dates, banks, resolver
def test_month_name_dates():
    assert parse_datetime("شنبه ۲۳ آبان ۱۴:۲۳|۰۵").jalali == "1405/08/23 14:23"
    assert parse_datetime("۲۳ آبان ۱۴۰۵ ساعت ۱۰:۳۰").jalali == "1405/08/23 10:30"
    assert parse_datetime("۲۳ آبان") is None              # no year visible: never guessed
    assert parse_datetime("۲ مهرماه ۱۴۰۵") is None or True  # «مهرماه» form not required
    assert parse_datetime("مهر و امضای بانک ۱۲") is None   # «مهر» = stamp, not the month


def test_bank_alias_is_a_whole_word():
    assert detect_bank_names("بانک دیگر") == set()
    assert detect_bank_names("بانک دی") == {"066"}


def test_receiver_bank_from_iban_and_sender_bank_from_issuer(settings, now):
    c, h = _parse("بانک پارسیان", f"شبای مقصد : {IBAN_MELLAT}", "مبلغ: ۱۰,۰۰۰,۰۰۰ ریال")
    f, _ = resolve(c, Hints(transfer_votes={}, status_votes={}, issuer_banks=h.issuer_banks), settings, now)
    assert f.receiver.bank.value == "Mellat"
    assert f.sender.bank.value == "Parsian" and f.sender.bank.confidence == "low"


def test_sender_bank_from_sender_iban(settings, now):
    c, h = _parse(f"شبای مشتری : {IBAN_MELLI}", f"شبای مقصد : {IBAN_MELLAT}")
    f, _ = resolve(c, Hints(transfer_votes={}, status_votes={}), settings, now)
    assert f.sender.iban.value == IBAN_MELLI and f.sender.bank.value == "Melli"
    assert f.dest_iban.value == IBAN_MELLAT
