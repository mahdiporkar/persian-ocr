"""Turn OCR text lines into labelled field candidates.

Used by the local OCR engines (Tesseract, EasyOCR). The VLM engine returns
structured candidates directly and does not go through this module, which keeps
the two readers independent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .dates import parse_datetime, _TIME_RE
from .iban import detect_bank_names, find_ibans
from .normalize import LABELS, find_labels, normalize_text, parse_amount_digits
from .numwords import parse_persian_number_words
from .schema import Candidate


@dataclass
class TextLine:
    text: str
    bbox: Optional[list[int]] = None  # [x, y, w, h]
    conf: Optional[float] = None       # 0..100
    source: str = "printed"            # printed | handwritten | digital
    pass_id: str = "full"


@dataclass
class ParseHints:
    transfer_votes: dict[str, int] = field(default_factory=dict)
    status_votes: dict[str, int] = field(default_factory=dict)
    bank_codes_named: set[str] = field(default_factory=set)
    full_text: str = ""


_AMOUNT_NUM_RE = re.compile(r"(?<![\d/:])\d{1,3}(?:[,.' ]\d{3})+(?![\d/:])|(?<![\d/:])\d{4,16}(?![\d/:])")
_LONG_DIGITS_RE = re.compile(r"(?<!\d)\d{6,30}(?!\d)")
_SCALE_WORD_RE = re.compile(r"(هزار|میلیون|ملیون|میلیارد|ملیارد|صد|ریال|تومان)")

_TRACKING_KEYS = ("tracking", "reference", "trace")


_MISREAD_CANON = {"بیگیری": "پیگیری", "پیگیرک": "پیگیری", "بیگیرک": "پیگیری", "پیگیر": "پیگیری"}


def _label_surface(text: str, key: str) -> Optional[str]:
    """The label as printed, including a leading «شماره»/«کد» (OCR misspellings fixed)."""
    t = normalize_text(text)
    for form in sorted(LABELS[key], key=len, reverse=True):
        f = normalize_text(form)
        m = re.search(rf"((?:شماره|کد|شماره ی)\s*)?{re.escape(f)}", t, flags=re.I)
        if m:
            word = _MISREAD_CANON.get(f, f)
            prefix = (m.group(1) or "").strip()
            if prefix and not word.startswith(prefix):
                return f"{prefix} {word}"
            return word
    return None


def _after_label(text: str, keys: tuple[str, ...]) -> str:
    """Text that follows the first label of the given kinds (':' trimmed)."""
    t = normalize_text(text)
    best = None
    for k in keys:
        for form in sorted(LABELS[k], key=len, reverse=True):
            f = normalize_text(form)
            i = t.lower().find(f.lower())
            if i >= 0 and (best is None or i < best[0]):
                best = (i, i + len(f))
                break
    if best is None:
        return t
    return t[best[1]:].lstrip(" :：-=\t")


def _currency_of(text: str) -> Optional[str]:
    labels = find_labels(text)
    if "rial" in labels and "toman" not in labels:
        return "rial"
    if "toman" in labels and "rial" not in labels:
        return "toman"
    return None


def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for a, b in spans:
        for i in range(a, b):
            chars[i] = " "
    return "".join(chars)


_SPLIT_LEAD_RE = re.compile(r"(?<![\d,])(\d{3}(?:,\d{3})+)\s+,\s*(\d{1,3})(?![\d,])")
_SPLIT_TAIL_RE = re.compile(r"(?<![\d,])(\d{1,3})\s*,\s+(\d{3}(?:,\d{3})+)(?![\d,])")


def repair_rtl_numbers(text: str) -> str:
    """Undo a common bidi artefact of OCR on RTL lines: the leading thousands group
    of an LTR number ends up after the rest, e.g. "500,000,000 ,4" for 4,500,000,000.
    Only the exact pattern (3-digit groups, then a comma and 1-3 digits) is repaired."""
    # a space after a thousands comma or a time colon: "4,500,000, 000", "11: 42"
    text = re.sub(r"(?<=\d),\s+(?=\d{3}(?!\d))", ",", text)
    text = re.sub(r"(?<=\d):\s+(?=\d{2}(?!\d))", ":", text)
    for _ in range(5):  # groups can be split more than once: "000,000 ,500 ,4"
        new = _SPLIT_LEAD_RE.sub(lambda m: f"{m.group(2)},{m.group(1)}", text)
        new = _SPLIT_TAIL_RE.sub(lambda m: f"{m.group(1)},{m.group(2)}", new)
        if new == text:
            break
        text = new
    return text


def parse_lines(lines: list[TextLine], engine: str) -> tuple[list[Candidate], ParseHints]:
    cands: list[Candidate] = []
    hints = ParseHints()
    hints.full_text = "\n".join(l.text for l in lines)
    hints.bank_codes_named = detect_bank_names(hints.full_text)

    pending_label: Optional[set[str]] = None   # label line with its value on the next line
    pending_label_text = ""
    date_cands_by_pass: dict[str, list[int]] = {}
    times_by_pass: dict[str, list[str]] = {}

    for ln in lines:
        raw = ln.text
        t = normalize_text(raw)
        if not t:
            continue
        labels = find_labels(t)
        ctx = set(labels)
        label_text = t
        if pending_label and not (labels - {"rial", "toman"}):
            ctx |= pending_label
            label_text = pending_label_text + " " + t
        src = ln.source

        def add(field_name, value, raw_text, **kw):
            cands.append(Candidate(field=field_name, value=value, raw_text=raw_text, source=src,
                                   engine=engine, pass_id=ln.pass_id, region=ln.bbox,
                                   ocr_conf=ln.conf, **kw))

        for key in ("satna", "paya", "internal", "card_to_card"):
            if key in labels:
                hints.transfer_votes[key] = hints.transfer_votes.get(key, 0) + 1
        for key in ("success", "failed", "pending"):
            if key in labels:
                hints.status_votes[key] = hints.status_votes.get(key, 0) + 1

        work = repair_rtl_numbers(t)
        consumed: list[tuple[int, int]] = []

        # ---- IBAN ----
        for raw_iban, info in find_ibans(work):
            if ctx & {"to_account", "beneficiary"}:
                role = "dest"
            elif ctx & {"from_account", "payer"}:
                role = "source"
            else:
                role = "unknown"
            add("iban", info.value, raw_iban, role=role)
            i = work.upper().find(raw_iban.upper())
            if i >= 0:
                consumed.append((i, i + len(raw_iban)))
        work = _strip_spans(work, consumed)

        # ---- date / time ----
        d = parse_datetime(work)
        if d:
            add("paid_at", d.jalali, d.raw)
            date_cands_by_pass.setdefault(ln.pass_id, []).append(len(cands) - 1)
            work = work.replace(d.raw.split(" ")[0], " ")
        tm = _TIME_RE.search(work)
        if tm:
            times_by_pass.setdefault(ln.pass_id, []).append(f"{int(tm.group(1)):02d}:{tm.group(2)}")
            work = work[: tm.start()] + " " * (tm.end() - tm.start()) + work[tm.end():]

        # ---- tracking / reference numbers ----
        tr_keys = [k for k in _TRACKING_KEYS if k in ctx]
        if tr_keys:
            label = _label_surface(label_text, tr_keys[0]) or tr_keys[0]
            for m in _LONG_DIGITS_RE.finditer(work):
                add("tracking_number", m.group(0), m.group(0), label=label)
                work = work[: m.start()] + " " * len(m.group(0)) + work[m.end():]

        # ---- amounts ----
        excluded = bool(ctx & {"balance", "fee"}) and "amount" not in labels
        if "fee" in labels or "balance" in labels:
            excluded = True
        has_amount_label = "amount" in ctx
        cur = _currency_of(label_text) or _currency_of(t)
        if not excluded and not (ctx & {"account_no", "card_no", "national_id", "iban"} and not has_amount_label):
            if has_amount_label or cur:
                for m in _AMOUNT_NUM_RE.finditer(work):
                    val = parse_amount_digits(m.group(0))
                    if val is None or val < 1000:
                        continue
                    add("amount", val, m.group(0).strip(), currency=cur)
            # amount in words
            if "amount_words" in ctx or (_SCALE_WORD_RE.search(t) and not re.search(r"\d", work)):
                tail = _after_label(t, ("amount_words", "amount"))
                w = parse_persian_number_words(tail)
                if w and w.value >= 1000:
                    cands.append(Candidate(field="amount", value=w.value, raw_text=tail, source="words",
                                           engine=engine, pass_id=ln.pass_id, region=ln.bbox,
                                           currency=w.currency, ocr_conf=ln.conf))

        # ---- destination bank named next to the destination account ----
        if ctx & {"to_account", "beneficiary"}:
            for code in detect_bank_names(t):
                add("dest_bank_name", code, t)

        # ---- names ----
        for key, fname in (("beneficiary", "dest_holder_name"), ("payer", "source_name")):
            if key in labels:
                tail = _after_label(t, (key,))
                tail = re.sub(r"[\d:：\-_/]+", " ", tail)
                for lbl in ("to_account", "from_account", "iban", "account_no"):
                    for form in LABELS[lbl]:
                        tail = tail.replace(normalize_text(form), " ")
                tail = " ".join(tail.split())
                if 2 <= len(tail) <= 60:
                    add(fname, tail, tail, role="dest" if key == "beneficiary" else "source")

        # label-only line: carry it to the next line
        value_found = bool(re.search(r"\d{4,}", t))
        if labels - {"rial", "toman"} and not value_found:
            pending_label = labels
            pending_label_text = t
        else:
            pending_label = None
            pending_label_text = ""

    # attach a stand-alone time to a date-only candidate of the same pass
    for pass_id, idxs in date_cands_by_pass.items():
        times = times_by_pass.get(pass_id) or []
        for i in idxs:
            c = cands[i]
            if len(str(c.value)) == 10 and times:
                p = parse_datetime(f"{c.value} {times[0]}")
                if p:
                    cands[i] = c.model_copy(update={"value": p.jalali, "raw_text": f"{c.raw_text} {times[0]}"})
    return cands, hints
