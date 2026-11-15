"""Match a blindly-read receipt against the debtor's open lines and suggest a route.

Expected values never reach the readers: this module only runs after all
fields have been resolved from the image.
"""
from __future__ import annotations

from typing import Optional

from rapidfuzz import fuzz

from .iban import parse_iban
from .normalize import format_fa_number, normalize_name, to_fa_digits
from .schema import REQUIRED_FIELDS, ExpectedLine, Fields, LineMatch

# Score weights. The score is a ranking aid, not a probability.
W_AMOUNT = {"AMOUNT_EXACT": 0.50, "AMOUNT_PARTIAL": 0.30, "AMOUNT_OVER_REMAINING": 0.05}
W_IBAN = {"IBAN_FULL_MATCH": 0.40, "IBAN_LAST_DIGITS_MATCH": 0.25}
W_NAME = 0.10

# warnings that always send a receipt to a human
MANAGER_WARNINGS = {
    "MULTIPLE_AMOUNTS", "UNKNOWN_CURRENCY", "IBAN_MISMATCH", "NO_MATCHING_LINE",
    "AMBIGUOUS_MATCH", "STATUS_NOT_SUCCESS", "POSSIBLE_EDITING", "IBAN_CONFLICT",
    "DATE_CONFLICT", "TRACKING_CONFLICT", "AMOUNT_CONFLICT_UNRESOLVED", "NO_EXPECTED_LINES",
}


def _amount_reason(amount_rial: int, line: ExpectedLine) -> str:
    remaining = (line.remaining_toman if line.remaining_toman is not None else line.amount_toman) * 10
    if amount_rial == remaining:
        return "AMOUNT_EXACT"
    if amount_rial < remaining:
        return "AMOUNT_PARTIAL"
    return "AMOUNT_OVER_REMAINING"


def score_line(fields: Fields, line: ExpectedLine, name_threshold: int) -> LineMatch:
    reasons: list[str] = []
    score = 0.0
    currency_assumed = None

    # ---- amount (compared in rial) ----
    amount_ok = False
    a = fields.amount_rial.value
    if a is None:
        reasons.append("AMOUNT_UNREADABLE")
    else:
        readings = [(a, fields.currency)]
        if fields.currency == "unknown":
            # no unit printed: test the number both as rial and as toman
            readings = [(a, "rial"), (a * 10, "toman")]
        best = None
        for val, cur in readings:
            r = _amount_reason(val, line)
            if best is None or W_AMOUNT[r] > W_AMOUNT[best[0]]:
                best = (r, cur)
        reasons.append(best[0])
        score += W_AMOUNT[best[0]]
        amount_ok = best[0] in ("AMOUNT_EXACT", "AMOUNT_PARTIAL")
        if fields.currency == "unknown":
            currency_assumed = best[1]
            reasons.append(f"CURRENCY_ASSUMED_{best[1].upper()}")

    # ---- IBAN ----
    iban_ok = False
    exp = parse_iban(line.iban) if line.iban else None
    got = fields.dest_iban
    if not got.value or exp is None:
        reasons.append("IBAN_UNREADABLE" if not got.value else "IBAN_NOT_EXPECTED")
    elif not got.partial:
        if got.value == exp.value and got.checksum_valid:
            reasons.append("IBAN_FULL_MATCH")
            score += W_IBAN["IBAN_FULL_MATCH"]
            iban_ok = True
        elif got.value[-6:] == exp.value[-6:]:
            # checksum failed but the tail agrees: treat like a partial read
            reasons.append("IBAN_LAST_DIGITS_MATCH")
            score += W_IBAN["IBAN_LAST_DIGITS_MATCH"]
            iban_ok = True
        else:
            reasons.append("IBAN_MISMATCH")
    else:
        sfx = got.value.lstrip(".")
        if exp.value.endswith(sfx) and len(sfx) >= 4:
            reasons.append("IBAN_LAST_DIGITS_MATCH")
            score += W_IBAN["IBAN_LAST_DIGITS_MATCH"]
            iban_ok = True
        else:
            reasons.append("IBAN_MISMATCH")

    # ---- name (informational, small weight) ----
    if line.holder_name and fields.dest_holder_name.value:
        r = fuzz.token_sort_ratio(normalize_name(line.holder_name), normalize_name(fields.dest_holder_name.value))
        if r >= name_threshold:
            reasons.append("NAME_MATCH")
            score += W_NAME
        else:
            reasons.append("NAME_MISMATCH")

    fits = amount_ok and iban_ok
    return LineMatch(line_id=line.line_id, score=round(score, 3), fits=fits, reasons=reasons,
                     currency_assumed=currency_assumed)


def sender_question(fields: Fields, line_iban: Optional[str]) -> str:
    if fields.amount_toman is not None and fields.currency != "unknown":
        amount = f"{format_fa_number(fields.amount_toman)} تومان"
    elif fields.amount_rial.value is not None:
        amount = f"{format_fa_number(fields.amount_rial.value)} ریال"
    else:
        amount = "مبلغ نامشخص"
    tail = None
    if fields.dest_iban.value:
        tail = fields.dest_iban.value.lstrip(".")[-4:]
    elif line_iban:
        tail = line_iban[-4:]
    acct = f"به حساب …{to_fa_digits(tail)}" if tail else "به حسابی که خوانا نبود"
    return f"این فیش را {amount} {acct} خواندم. درست است؟"


def decide(fields: Fields, warnings: list[str], lines: list[ExpectedLine], name_threshold: int,
           image_flags: list[str]) -> tuple[list[LineMatch], Optional[str], str, Optional[str], list[str]]:
    """Return (matches, best_line_id, route, sender_question_fa, warnings)."""
    warnings = list(warnings)

    def warn(code):
        if code not in warnings:
            warnings.append(code)

    if "POSSIBLE_EDITING" in image_flags:
        warn("POSSIBLE_EDITING")

    matches = [score_line(fields, ln, name_threshold) for ln in lines]
    matches.sort(key=lambda m: -m.score)
    fitting = [m for m in matches if m.fits]
    best: Optional[LineMatch] = None

    if not lines:
        warn("NO_EXPECTED_LINES")
    elif not fitting:
        warn("NO_MATCHING_LINE")
        if any("IBAN_MISMATCH" in m.reasons and any(r in m.reasons for r in ("AMOUNT_EXACT", "AMOUNT_PARTIAL"))
               for m in matches):
            warn("IBAN_MISMATCH")
    elif len(fitting) > 1 and fitting[0].score == fitting[1].score:
        warn("AMBIGUOUS_MATCH")
    else:
        best = fitting[0]
    if fields.currency == "unknown" and fields.amount_rial.value is not None:
        warn("UNKNOWN_CURRENCY")

    confidences = {
        "amount_rial": fields.amount_rial.confidence,
        "dest_iban": fields.dest_iban.confidence,
        "paid_at": fields.paid_at.confidence,
        "tracking_number": fields.tracking_number.confidence,
    }
    any_conflict = any(v == "conflict" for v in confidences.values())

    if (any_conflict or fields.amount_rial.value is None or best is None
            or fields.document_status != "success"
            or any(w in MANAGER_WARNINGS for w in warnings)):
        route = "manager_review"
    elif (all(confidences[f] == "high" for f in REQUIRED_FIELDS)
          and fields.dest_iban.checksum_valid is True
          and "IBAN_FULL_MATCH" in best.reasons
          and len(fitting) == 1
          and fields.transfer_type != "unknown"):
        route = "match"
    else:
        route = "confirm_with_sender"

    question = None
    if route == "confirm_with_sender":
        line_iban = next((ln.iban for ln in lines if best and ln.line_id == best.line_id), None)
        question = sender_question(fields, line_iban)
    return matches, (best.line_id if best else None), route, question, warnings
