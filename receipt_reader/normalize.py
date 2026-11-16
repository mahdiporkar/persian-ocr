"""Persian text normalization and low-level value parsing.

Everything here is pure and deterministic so it can be unit-tested without images.
"""
from __future__ import annotations

import re
from typing import Optional

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_DIGIT_TABLE = {ord(c): str(i) for i, c in enumerate(PERSIAN_DIGITS)}
_DIGIT_TABLE.update({ord(c): str(i) for i, c in enumerate(ARABIC_DIGITS)})

_CHAR_TABLE = {
    ord("ي"): "ی",
    ord("ى"): "ی",
    ord("ئ"): "ی",
    ord("ك"): "ک",
    ord("ة"): "ه",
    ord("ۀ"): "ه",
    ord("أ"): "ا",
    ord("إ"): "ا",
    ord("ٱ"): "ا",
    ord("ؤ"): "و",
    ord("٬"): ",",  # Arabic thousands separator
    ord("،"): ",",  # Arabic comma
    ord("٫"): ".",  # Arabic decimal separator
}

# zero-width and bidi control characters, tatweel and Arabic diacritics
_STRIP_RE = re.compile(
    "[‌‍‎‏‪-‮⁦-⁩﻿ـً-ٰٟ]"
)
_WS_RE = re.compile(r"[ \t  -​　]+")


def to_ascii_digits(text: str) -> str:
    return text.translate(_DIGIT_TABLE)


def normalize_text(text: Optional[str]) -> str:
    """Canonical form used for matching labels and parsing values.

    - Persian/Arabic digits -> ASCII
    - Arabic letter variants -> Persian (ي->ی, ك->ک ...)
    - ZWNJ / bidi marks / tatweel / diacritics removed (ZWNJ becomes nothing, so
      «می‌شود» and «میشود» compare equal)
    - runs of whitespace collapsed
    """
    if not text:
        return ""
    text = to_ascii_digits(text).translate(_CHAR_TABLE)
    text = _STRIP_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def normalize_name(text: Optional[str]) -> str:
    """Normalization for fuzzy person-name comparison."""
    t = normalize_text(text)
    t = re.sub(r"[^\w ]", " ", t)
    for honorific in ("آقای", "خانم", "اقای", "جناب", "سرکار"):
        t = re.sub(rf"(^| ){honorific}( |$)", " ", t)
    t = t.replace("آ", "ا")
    return " ".join(t.split())


_SEP_CHARS = ",. '’"


def parse_amount_digits(raw: Optional[str]) -> Optional[int]:
    """Parse an amount written in digits with optional thousands separators.

    Accepts "4,500,000,000", "۴٬۵۰۰٬۰۰۰٬۰۰۰", "4.500.000.000", "4 500 000 000",
    "450000000". Returns None when the string is ambiguous (e.g. "4,50,000",
    or a real decimal part) — we never guess.
    """
    if raw is None:
        return None
    s = normalize_text(raw)
    s = re.sub(r"(ریال|تومان|rial|rials|irr|toman|tomans)", "", s, flags=re.I).strip()
    s = s.strip("-–—:= ")
    if not s or not re.fullmatch(r"[0-9][0-9,.' ’]*", s):
        return None
    s = s.rstrip(_SEP_CHARS)
    # trailing ".00"/",00" style decimals: accept only all-zero fraction
    m = re.fullmatch(r"(.+?)[.,](0{1,2})", s)
    if m and re.search(r"[.,' ]\d{3}$", m.group(1)) is not None:
        s = m.group(1)
    if len(s) > 1 and s[0] == "0":
        return None  # "09,000" / "0450000": a leading zero is not an amount
    if not any(c in s for c in _SEP_CHARS):
        return int(s)
    groups = re.split(r"[,.' ’]+", s)
    if len(groups) < 2:
        return None
    if not (1 <= len(groups[0]) <= 3) or any(len(g) != 3 for g in groups[1:]):
        return None
    return int("".join(groups))


def digits_only(text: Optional[str]) -> str:
    return re.sub(r"\D", "", to_ascii_digits(text or ""))


def to_fa_digits(text: str) -> str:
    return text.translate({ord(str(i)): PERSIAN_DIGITS[i] for i in range(10)})


def format_fa_number(n: int) -> str:
    """4500000000 -> «۴٬۵۰۰٬۰۰۰٬۰۰۰» (Persian digits, Persian thousands separator)."""
    s = f"{n:,}".replace(",", "٬")
    return s.translate({ord(str(i)): PERSIAN_DIGITS[i] for i in range(10)})


# --- labels -----------------------------------------------------------------
# Each key is a canonical label id; values are normalized surface forms.
LABELS: dict[str, list[str]] = {
    "payer": ["واریز کننده", "واریزکننده", "پرداخت کننده", "فرستنده", "نام واریز کننده", "انتقال دهنده",
              "انتقال‌دهنده", "نام مشتری", "صاحب حساب مبدا"],
    "beneficiary": ["ذینفع", "ذی نفع", "گیرنده", "صاحب حساب مقصد", "نام ذینفع", "دریافت کننده", "متعلق به",
                    "صاحب سپرده مقصد"],
    # section markers: everything after them (until the other marker) is about that party
    "to_account": ["به حساب", "حساب مقصد", "شبای مقصد", "شماره شبای مقصد", "مقصد", "شبا مقصد", "واریز به",
                   "مقصل",  # common OCR misread of «مقصد» (not a Persian word)
                   "به کارت", "به بانک"],
    "from_account": ["از حساب", "حساب مبدا", "مبدا", "شبای مبدا", "حساب برداشت", "برداشت از", "از کارت",
                     "از بانک", "سپرده مبدا", "شبای مشتری", "حساب مشتری", "نبای مشتری"],
    "holder": ["دارنده کارت", "به نام", "صاحب حساب", "نام صاحب حساب", "صاحبان حساب"],
    "account_ref": ["به شماره", "شماره حساب", "حساب شماره"],
    "bank": ["بانک"],
    "iban": ["شماره شبا", "شبا", "iban", "sheba", "شناسه شبا"],
    "account_no": ["شماره حساب", "حساب شماره"],
    "card_no": ["شماره کارت", "کارت مقصد", "کارت مبدا", "از کارت", "به کارت"],
    "deposit_id": ["شناسه واریز", "شناسه پرداخت"],
    "terminal": ["شماره ترمینال", "ترمینال", "شماره پایانه"],
    "amount": ["مبلغ", "مبلغ واریزی", "مبلغ انتقال", "مبلغ تراکنش", "مبلغ ریال", "مبلغ به ریال", "amount"],
    "amount_words": ["مبلغ به حروف", "به حروف", "مبلغ حروف"],
    "rial": ["ریال", "rial", "irr"],
    "toman": ["تومان", "toman"],
    # OCR often confuses پ/ب and ی/ک at the word end: include the common misreads
    "tracking": ["شماره پیگیری", "کد پیگیری", "پیگیری", "بیگیری", "پیگیرک", "بیگیرک", "پیگیر", "tracking"],
    "reference": ["شماره مرجع", "کد مرجع", "مرجع", "reference", "ref"],
    "trace": ["شماره ارجاع", "کد رهگیری", "شماره رهگیری", "شماره تراکنش", "شناسه تراکنش", "شماره سند",
              "شناسه دستور پرداخت", "شماره دستور پرداخت"],
    "date": ["تاریخ", "date", "تاریخ تراکنش", "تاریخ انتقال"],
    "time": ["ساعت", "زمان", "time"],
    "satna": ["ساتنا", "satna"],
    "paya": ["پایا", "paya"],
    "internal": ["انتقال داخلی", "حساب به حساب", "انتقال وجه داخلی", "بین بانکی داخلی"],
    "card_to_card": ["کارت به کارت", "انتقال کارت", "card to card", "کارت مقصد", "به کارت"],
    "success": ["موفق", "انجام شد", "با موفقیت", "successful", "success", "تراکنش موفق", "انتقال یافت"],
    "failed": ["ناموفق", "نا موفق", "انجام نشد", "خطا", "failed", "unsuccessful", "رد شد", "برگشت"],
    "pending": ["در انتظار", "در حال پردازش", "در جریان", "pending", "ثبت شد"],
    "balance": ["مانده", "موجودی", "مانده حساب", "balance"],
    "fee": ["کارمزد", "fee", "هزینه"],
    "purpose": ["بابت", "توضیحات", "شرح انتقال"],
    # fields that carry numbers but are none of ours: a number right after them is theirs
    "other_field": ["کاربر", "کد شعبه", "شعبه صادره", "تعداد کل", "تعداد", "تلفن", "کد پستی", "شماره شناسنامه",
                    "شماره تماس", "کد اقتصادی"],
    "national_id": ["کد ملی", "شماره ملی", "شناسه ملی"],
}

_NORMALIZED_LABELS = {k: [normalize_text(v).lower() for v in vs] for k, vs in LABELS.items()}


def label_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, label id) of every label occurrence in already-normalized text, longest
    form first, non-overlapping, in text order."""
    t = text.lower()
    forms = sorted(((f, k) for k, fs in _NORMALIZED_LABELS.items() for f in fs if f), key=lambda x: -len(x[0]))
    taken = [False] * len(t)
    out = []
    for f, k in forms:
        pattern = rf"\b{re.escape(f)}\b" if f.isascii() else re.escape(f)
        for m in re.finditer(pattern, t):
            if any(taken[m.start():m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            out.append((m.start(), m.end(), k))
    return sorted(out)


def find_labels(text: str) -> set[str]:
    """Return the set of canonical label ids present in a line of text."""
    t = " ".join(normalize_text(text).lower().replace(":", " ").replace("：", " ").split())
    found = set()
    for key, forms in _NORMALIZED_LABELS.items():
        for f in forms:
            if not f:
                continue
            if f.isascii():
                if re.search(rf"\b{re.escape(f)}\b", t):
                    found.add(key)
                    break
            elif f in t:
                # «ناموفق» contains «موفق» — resolve below
                found.add(key)
                break
    if "failed" in found and "success" in found:
        stripped = t
        for f in _NORMALIZED_LABELS["failed"]:
            stripped = stripped.replace(f, " ")
        if not any(f in stripped for f in _NORMALIZED_LABELS["success"]):
            found.discard("success")
    if "amount_words" in found:
        found.add("amount")
    return found
