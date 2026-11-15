"""Parser for amounts written in Persian words (مبلغ به حروف).

    «چهارصد و پنجاه میلیون ریال»            -> 450_000_000
    «چهار میلیارد و پانصد میلیون ریال»     -> 4_500_000_000
    «یکصد و بیست و پنج هزار تومان»         -> 125_000
    «۴۵۰ میلیون ریال»                       -> 450_000_000   (mixed digits + scale words)

The parser is strict: if any part of the text is not a recognised number word,
connector or unit, it returns None instead of a partial value. A partial parse of
an amount is worse than no parse.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .normalize import normalize_text

_UNITS = {
    "صفر": 0, "یک": 1, "یه": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "پنح": 5,
    "شش": 6, "شیش": 6, "هفت": 7, "هشت": 8, "نه": 9,
}
_TEENS = {
    "ده": 10, "یازده": 11, "دوازده": 12, "سیزده": 13, "چهارده": 14, "پانزده": 15,
    "پونزده": 15, "شانزده": 16, "شونزده": 16, "هفده": 17, "هیفده": 17, "هجده": 18,
    "هیجده": 18, "نوزده": 19,
}
_TENS = {
    "بیست": 20, "سی": 30, "چهل": 40, "پنجاه": 50, "شصت": 60, "هفتاد": 70,
    "هشتاد": 80, "نود": 90,
}
_HUNDREDS = {
    "یکصد": 100, "صد": 100, "دویست": 200, "سیصد": 300, "چهارصد": 400, "پانصد": 500,
    "پونصد": 500, "ششصد": 600, "شیشصد": 600, "هفتصد": 700, "هشتصد": 800, "نهصد": 900,
}
_SCALES = {
    "هزار": 10**3, "میلیون": 10**6, "ملیون": 10**6, "میلیارد": 10**9, "ملیارد": 10**9,
    "بیلیون": 10**9, "تریلیون": 10**12, "تریلیارد": 10**15,
}
_SMALL = {**_UNITS, **_TEENS, **_TENS, **_HUNDREDS}
_ALL = {**_SMALL, **_SCALES}

_UNIT_WORDS = {"ریال": "rial", "تومان": "toman", "تومن": "toman"}
_FILLER = ["مبلغ", "به حروف", "حروف", "فقط", "تمام", "معادل", "عدد", "مبلغ به حروف", "بحروف"]

# vocabulary sorted longest-first for greedy segmentation of run-together words
_VOCAB = sorted(_ALL.keys(), key=len, reverse=True)


@dataclass(frozen=True)
class WordsAmount:
    value: int
    currency: Optional[str]  # "rial" | "toman" | None


def _tokenize(text: str) -> Optional[list[str]]:
    """Split into number tokens. Handles both spaced and run-together writing,
    e.g. «چهارصدوپنجاه». Returns None on unrecognised residue."""
    tokens: list[str] = []
    for chunk in text.split():
        if re.fullmatch(r"\d+", chunk):
            tokens.append(chunk)
            continue
        i = 0
        while i < len(chunk):
            for w in _VOCAB:
                if chunk.startswith(w, i):
                    tokens.append(w)
                    i += len(w)
                    break
            else:
                if chunk[i] == "و":  # connector «و» glued to neighbours
                    i += 1
                    continue
                return None
    return tokens


def parse_persian_number_words(text: Optional[str]) -> Optional[WordsAmount]:
    if not text:
        return None
    t = normalize_text(text)
    t = re.sub(r"[.,:;()\[\]«»\-_/]", " ", t)
    t = re.sub(r"(\d) (?=\d{3}\b)", r"\1", t)  # "450 000" -> "450000"

    currency = None
    for w, cur in _UNIT_WORDS.items():
        if w in t:
            if currency and currency != cur:
                return None  # both units mentioned: ambiguous
            currency = cur
            t = t.replace(w, " ")
    for f in sorted(_FILLER, key=len, reverse=True):
        t = t.replace(f, " ")
    # standalone connector
    t = re.sub(r"(^| )و(?= |$)", " ", t)
    t = " ".join(t.split())
    if not t:
        return None

    tokens = _tokenize(t)
    if not tokens:
        return None

    total = 0          # completed part (already multiplied by its scale)
    group = 0          # current group < 1000 (or a raw digit number)
    last_scale = None  # to reject "هزار میلیون هزار" style garbage
    seen_any = False
    for tok in tokens:
        if tok.isdigit():
            group += int(tok)
            seen_any = True
            continue
        if tok in _SCALES:
            scale = _SCALES[tok]
            if total and scale > _largest_scale(total):
                # e.g. «صد هزار میلیون»: everything so far is multiplied
                total = (total + group) * scale
            else:
                if last_scale is not None and scale >= last_scale and total:
                    return None
                total += (group or 1) * scale
            last_scale = scale
            group = 0
            seen_any = True
            continue
        val = _SMALL[tok]
        if tok == "صد" and 0 < group < 10:
            group *= 100  # «چهار صد»
        else:
            group += val
        seen_any = True
    if not seen_any:
        return None
    return WordsAmount(total + group, currency)


def _largest_scale(n: int) -> int:
    for s in sorted(_SCALES.values(), reverse=True):
        if n >= s:
            return s
    return 1
