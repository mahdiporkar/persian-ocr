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
# tracking numbers: 5-30 digits, optionally hyphen-grouped ("123-456789-0"); never digits
# taken out of a hex transaction id such as "a1b2c3d4e5f6"
_LONG_DIGITS_RE = re.compile(r"(?<![\dA-Za-z\-])\d[\d\-]{3,40}\d(?![\dA-Za-z])")
# card numbers as printed: full or masked 16 digits, or a short masked form "**** 6120"
_CARD_RE = re.compile(r"(?<![\dA-Za-z])\d{4}[\s\-_]*[\dxX*•]{2,4}[\s\-_]*[\dxX*•]{4}[\s\-_]*\d{4}(?!\d)"
                      r"|[*xX•]{3,}[\s\-_:]*\d{4}(?!\d)")
_ACCOUNT_RE = re.compile(r"(?<![\dA-Za-z])\d[\d\-]{6,22}\d(?![\dA-Za-z])")
_SCALE_WORD_RE = re.compile(r"(هزار|میلیون|ملیون|میلیارد|ملیارد|صد|ریال|تومان)")

_TRACKING_KEYS = ("tracking", "reference", "trace")
SECTION_TTL = 4  # a «از …» / «به …» label applies to at most this many following lines

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
    return t[best[1]:].lstrip(" :：-=\t.")


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


def _blank(text: str, m: re.Match) -> str:
    return text[: m.start()] + " " * (m.end() - m.start()) + text[m.end():]


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


def _clean_name(text: str) -> Optional[str]:
    t = re.sub(r"[\d:：\-_/.,()|»«*]+", " ", text)
    for key in ("to_account", "from_account", "iban", "account_no", "account_ref", "holder", "card_no", "bank",
                "beneficiary", "payer"):
        for form in sorted(LABELS[key], key=len, reverse=True):
            t = t.replace(normalize_text(form), " ")
    t = " ".join(w for w in t.split() if re.search(r"[؀-ۿ]", w))
    return t if 2 <= len(t) <= 60 else None


def parse_lines(lines: list[TextLine], engine: str) -> tuple[list[Candidate], ParseHints]:
    cands: list[Candidate] = []
    hints = ParseHints()
    hints.full_text = "\n".join(l.text for l in lines)
    hints.bank_codes_named = detect_bank_names(hints.full_text)

    pending_label: Optional[set[str]] = None   # label line with its value on the next line
    pending_label_text = ""
    section: Optional[str] = None               # "sender" | "receiver", from «از …» / «به …» labels
    section_age = 0
    date_cands_by_pass: dict[str, list[int]] = {}
    times_by_pass: dict[str, list[str]] = {}

    for ln in lines:
        t = normalize_text(ln.text)
        if not t:
            continue
        labels = find_labels(t)
        ctx = set(labels)
        label_text = t
        # a label from the previous line applies only to a value-only line; a line with its own
        # (possibly misread, unrecognised) label text, e.g. «شماره ارچاع», must not inherit it
        own_words = [w for w in re.findall(r"[؀-ۿ]{3,}", t) if w not in ("ریال", "تومان")]
        if pending_label and not (labels - {"rial", "toman"}) and not own_words:
            ctx |= pending_label
            label_text = pending_label_text + " " + t
        src = ln.source

        # which party does this line talk about?
        is_from = bool(ctx & {"from_account", "payer"})
        is_to = bool(ctx & {"to_account", "beneficiary"})
        line_side = "sender" if is_from and not is_to else "receiver" if is_to and not is_from else None
        if line_side:
            section, section_age = line_side, 0
        else:
            section_age += 1
            if section_age > SECTION_TTL:
                section = None
        side = line_side or section

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
        # cards keep only their last 4 digits, so for card numbers an unreadable group
        # ("29-??22xX-XXXX-4821") only blanks that hyphen group
        work_cards = re.sub(r"[^\s:\-]*\?[^\s:\-]*", lambda m: " " * len(m.group(0)), work)
        # everywhere else a token with an unreadable character ('?') is unreadable as a whole: never
        # pick digit fragments out of it (e.g. "13??517?461208833?9" must not yield "72363666")
        work = re.sub(r"[^\s:]*\?[^\s:]*", lambda m: " " * len(m.group(0)), work)

        # ---- IBAN ----
        consumed: list[tuple[int, int]] = []
        for raw_iban, info in find_ibans(work, iban_label="iban" in ctx):
            role = {"receiver": "dest", "sender": "source"}.get(side or "", "unknown")
            add("iban", info.value, raw_iban, role=role)
            i = work.upper().find(raw_iban.upper())
            if i >= 0:
                consumed.append((i, i + len(raw_iban)))
        work = _strip_spans(work, consumed)
        work_cards = _strip_spans(work_cards, consumed)

        # ---- card numbers (downstream keeps only the last 4 digits) ----
        if "card_no" in ctx or re.search(r"[xX*•]{2}", work_cards):
            card_side = side
            if card_side is None and "card_no" in ctx:
                card_side = "sender"  # «شماره کارت» before any «مقصد» line is the payer's card
            for m in list(_CARD_RE.finditer(work_cards)):
                if card_side:
                    add(f"{card_side}_card", m.group(0).strip(), m.group(0).strip())
                work = _blank(work, m)

        # ---- date / time ----
        d = parse_datetime(work)
        if d:
            add("paid_at", d.jalali, d.raw)
            date_cands_by_pass.setdefault(ln.pass_id, []).append(len(cands) - 1)
            work = work.replace(d.raw.split(" ")[0], " ")
        tm = _TIME_RE.search(work)
        if tm:
            times_by_pass.setdefault(ln.pass_id, []).append(f"{int(tm.group(1)):02d}:{tm.group(2)}")
            work = _blank(work, tm)

        # ---- tracking / reference numbers ----
        tr_keys = [k for k in _TRACKING_KEYS if k in ctx]
        if tr_keys and not ctx & {"terminal", "deposit_id"}:
            label = _label_surface(label_text, tr_keys[0]) or tr_keys[0]
            for m in list(_LONG_DIGITS_RE.finditer(work)):
                digits = m.group(0).replace("-", "")
                if len(digits) >= 5:
                    add("tracking_number", digits, m.group(0), label=label)
                    work = _blank(work, m)

        # ---- account numbers of a party ----
        if side and ctx & {"account_ref", "account_no", "from_account", "to_account"} \
                and not ctx & {"amount", "tracking", "reference", "trace", "deposit_id"}:
            for m in list(_ACCOUNT_RE.finditer(work)):
                digits = m.group(0).replace("-", "")
                if 8 <= len(digits) <= 20:
                    add(f"{side}_account", digits, m.group(0))
                    work = _blank(work, m)

        # ---- amounts ----
        excluded = bool(labels & {"fee", "balance", "deposit_id", "terminal"})
        has_amount_label = "amount" in ctx
        amount_label_inherited = has_amount_label and "amount" not in labels
        cur = _currency_of(label_text) or _currency_of(t)
        if not excluded and not (ctx & {"account_no", "account_ref", "card_no", "national_id", "iban"}
                                 and not has_amount_label):
            if has_amount_label or cur:
                for m in _AMOUNT_NUM_RE.finditer(work):
                    if amount_label_inherited and not re.search(r"[,.' ]", m.group(0).strip()):
                        # label came from the previous line: accept only a properly grouped amount,
                        # not any long digit run (that is usually the next field's reference number)
                        continue
                    val = parse_amount_digits(m.group(0))
                    if val is None or val < 1000:
                        continue
                    add("amount", val, m.group(0).strip(), currency=cur,
                        label=_label_surface(label_text, "amount") if has_amount_label else None)
            # amount in words
            if "amount_words" in ctx or (_SCALE_WORD_RE.search(t) and not re.search(r"\d", work)):
                tail = _after_label(t, ("amount_words", "amount"))
                w = parse_persian_number_words(tail)
                if w and w.value >= 1000:
                    cands.append(Candidate(field="amount", value=w.value, raw_text=tail, source="words",
                                           engine=engine, pass_id=ln.pass_id, region=ln.bbox,
                                           currency=w.currency, ocr_conf=ln.conf, label="مبلغ به حروف"))

        # ---- bank named for a party (only on the line that names the side) ----
        if line_side and ("bank" in ctx or "card_no" in ctx):
            for code in detect_bank_names(t):
                add(f"{line_side}_bank", code, t)

        # ---- names ----
        name_side, key = None, None
        if "beneficiary" in labels:
            name_side, key = "receiver", "beneficiary"
        elif "payer" in labels:
            name_side, key = "sender", "payer"
        elif "holder" in labels:
            # «دارنده کارت» is the destination card's holder on Iranian receipts; «به نام» needs
            # a known «از …»/«به …» section, otherwise we cannot tell whose name it is
            if side:
                name_side, key = side, "holder"
            elif "دارنده" in t:
                name_side, key = "receiver", "holder"
        if name_side:
            name = _clean_name(_after_label(t, (key,)))
            if name:
                add(f"{name_side}_name", name, name, role="dest" if name_side == "receiver" else "source")

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
