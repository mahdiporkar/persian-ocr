"""Cross-check candidates from all engines/passes and assign confidence levels.

The exact rules are documented in README.md ("Confidence rules"). Summary:

  high     — two independent reads agree (different engine, or same engine on a
             different crop) and at least one is printed/digital; or one
             printed/digital read that passes every validator of its field.
             For amounts a single read is never "high" unless the amount in words
             agrees: there is no other validator for an amount.
  medium   — one clean source and no conflict.
  low      — handwriting only, a failed validator, or unclear.
  conflict — sources disagree; every candidate is kept, no value is picked.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

from rapidfuzz import fuzz

from .config import Settings
from .dates import is_plausible, parse_datetime
from .iban import bank_for_code, parse_iban
from .normalize import digits_only, normalize_name
from .schema import (AmountField, Candidate, DateField, Fields, IbanField, TextField,
                     TrackingField)

TRUSTED = ("printed", "digital")
_LEVEL = {"conflict": -1, "low": 0, "medium": 1, "high": 2}


def _cap(level: str, cap: str) -> str:
    if level == "conflict":
        return level
    return level if _LEVEL[level] <= _LEVEL[cap] else cap


def _key(c: Candidate, crops_independent: bool = True) -> tuple[str, str]:
    return (c.engine, c.pass_id) if crops_independent else (c.engine, "")


def _independent(cs: Iterable[Candidate], crops_independent: bool = True) -> int:
    return len({_key(c, crops_independent) for c in cs})


def _pub(cs: Iterable[Candidate]) -> list[dict]:
    return [c.public() for c in cs]


@dataclass
class Hints:
    transfer_votes: dict[str, int]
    status_votes: dict[str, int]
    printed_strip_found: bool = False


class Resolver:
    def __init__(self, settings: Settings, now: datetime):
        self.s = settings
        self.now = now
        self.warnings: list[str] = []

    def _ind(self, cs: Iterable[Candidate]) -> int:
        return _independent(cs, self.s.same_engine_crops_independent)

    def warn(self, code: str) -> None:
        if code not in self.warnings:
            self.warnings.append(code)

    # ------------------------------------------------------------------ amount
    def amount(self, cands: list[Candidate]) -> tuple[AmountField, str, Optional[int]]:
        amts = [c for c in cands if c.field == "amount" and isinstance(c.value, int)]
        out = AmountField(candidates=_pub(amts))
        if not amts:
            return out, "unknown", None

        units = {c.currency for c in amts if c.currency in ("rial", "toman")}
        currency = next(iter(units)) if len(units) == 1 else "unknown"
        if len(units) > 1:  # both rial and toman were seen
            self.warn("AMOUNT_CONFLICT")

        words = [c for c in amts if c.source == "words"]
        digits = [c for c in amts if c.source != "words"]
        trusted = [c for c in digits if c.source in TRUSTED]
        hand = [c for c in digits if c.source == "handwritten"]
        word_vals = {c.value for c in words}

        value: Optional[int] = None
        conf = "low"
        tvals = {c.value for c in trusted}
        if len(tvals) > 1:
            self.warn("MULTIPLE_AMOUNTS")
            self.warn("AMOUNT_CONFLICT")
            conf = "conflict"
        elif len(tvals) == 1:
            value = next(iter(tvals))
            agree = [c for c in trusted if c.value == value]
            words_agree = value in word_vals
            if word_vals and not words_agree:
                self.warn("AMOUNT_CONFLICT")
                conf, value = "conflict", None
            else:
                if word_vals - {value}:
                    self.warn("AMOUNT_CONFLICT")
                conf = "high" if (self._ind(agree) >= 2 or words_agree) else "medium"
                if any(c.value != value for c in hand):
                    self.warn("AMOUNT_CONFLICT")
                    self.warn("HANDWRITTEN_AMOUNT_DISAGREES")
        else:
            hvals = {c.value for c in hand}
            if len(hvals) > 1:
                self.warn("AMOUNT_CONFLICT")
                conf = "conflict"
            elif len(hvals) == 1:
                value = next(iter(hvals))
                if word_vals and value not in word_vals:
                    self.warn("AMOUNT_CONFLICT")
                    conf, value = "conflict", None
                elif value in word_vals:
                    conf = "medium"  # handwriting-only is capped at medium
                else:
                    conf = "low"
            elif len(word_vals) == 1:
                value, conf = next(iter(word_vals)), "low"
            elif len(word_vals) > 1:
                self.warn("AMOUNT_CONFLICT")
                conf = "conflict"

        rial: Optional[int] = None
        if value is not None:
            rial = value * 10 if currency == "toman" else value
            if not (self.s.min_amount_rial <= rial <= self.s.max_amount_rial):
                conf = _cap(conf, "low")
        out.value = rial
        out.confidence = conf
        toman = rial // 10 if (rial is not None and rial % 10 == 0) else None
        return out, currency, toman

    # -------------------------------------------------------------------- IBAN
    def iban(self, cands: list[Candidate]) -> IbanField:
        ibs = [c for c in cands if c.field == "iban"]
        source_vals = {c.value for c in ibs if c.role == "source"}
        dest = [c for c in ibs if c.role == "dest"]
        unknown = [c for c in ibs if c.role in (None, "unknown") and c.value not in source_vals]
        pool = dest or unknown
        out = IbanField(candidates=_pub(ibs))
        if not pool:
            return out
        role_cap = "high" if dest else "medium"

        parsed = [(c, parse_iban(c.value)) for c in pool]
        parsed = [(c, p) for c, p in parsed if p is not None]
        full_valid = [(c, p) for c, p in parsed if not p.partial and p.checksum_valid]
        full_invalid = [(c, p) for c, p in parsed if not p.partial and not p.checksum_valid]
        partial = [(c, p) for c, p in parsed if p.partial]

        valid_vals = {p.value for _, p in full_valid}
        if len(valid_vals) > 1:
            self.warn("IBAN_CONFLICT")
            out.confidence = "conflict"
            return out
        if len(valid_vals) == 1:
            v = next(iter(valid_vals))
            p = next(p for _, p in full_valid)
            agree = [c for c, q in full_valid]
            for _, q in partial:
                if not v.endswith(q.value.lstrip(".")):
                    self.warn("IBAN_CONFLICT")
                    out.confidence = "conflict"
                    out.value = None
                    return out
            agree_trusted = [c for c in agree if c.source in TRUSTED]
            known_bank = bank_for_code(p.bank_code) is not None
            if agree_trusted and (self._ind(agree) >= 2 or known_bank):
                conf = "high"
            elif agree_trusted:
                conf = "medium"
            elif self._ind(agree) >= 2:
                conf = "medium"  # handwriting only, but two reads agree and checksum passes
            else:
                conf = "low"
            named = {c.value for c in cands if c.field == "dest_bank_name"}
            if named and p.bank_code not in named:
                self.warn("BANK_NAME_MISMATCH")
                conf = _cap(conf, "medium")
            out.value, out.partial, out.checksum_valid = v, False, True
            out.bank, out.bank_fa = p.bank, p.bank_fa
            out.confidence = _cap(conf, role_cap)
            return out
        if full_invalid:
            self.warn("INVALID_IBAN")
            v, _ = Counter(p.value for _, p in full_invalid).most_common(1)[0]
            p = next(p for _, p in full_invalid if p.value == v)
            out.value, out.checksum_valid, out.partial = v, False, False
            out.bank, out.bank_fa = p.bank, p.bank_fa
            out.confidence = "low"
            return out
        # partial only
        self.warn("PARTIAL_IBAN")
        sfx = sorted({p.value.lstrip(".") for _, p in partial}, key=len, reverse=True)
        longest = sfx[0]
        if any(not longest.endswith(s) for s in sfx[1:]):
            self.warn("IBAN_CONFLICT")
            out.confidence = "conflict"
            return out
        out.value, out.partial, out.checksum_valid = "..." + longest, True, None
        trusted = [c for c, _ in partial if c.source in TRUSTED]
        out.confidence = _cap("medium" if trusted else "low", role_cap)
        return out

    # -------------------------------------------------------------------- date
    def paid_at(self, cands: list[Candidate]) -> DateField:
        ds = []
        for c in cands:
            if c.field != "paid_at":
                continue
            p = parse_datetime(str(c.value))
            if p:
                ds.append((c, p))
        out = DateField(candidates=_pub(c for c, _ in ds))
        if not ds:
            return out
        trusted = [(c, p) for c, p in ds if c.source in TRUSTED]
        pool = trusted or ds
        by_day = defaultdict(list)
        for c, p in pool:
            by_day[p.iso[:10]].append((c, p))
        if len(by_day) > 1:
            self.warn("DATE_CONFLICT")
            out.confidence = "conflict"
            return out
        day_items = next(iter(by_day.values()))
        timed = [(c, p) for c, p in day_items if not p.date_only]
        times = {p.jalali for _, p in timed}
        if len(times) > 1:
            self.warn("DATE_CONFLICT")
            out.confidence = "conflict"
            return out
        c0, p0 = (timed or day_items)[0]
        if trusted and any(p.iso[:10] != p0.iso[:10] for c, p in ds if c.source == "handwritten"):
            self.warn("DATE_CONFLICT")
        out.jalali, out.iso = p0.jalali, p0.iso
        plausible = is_plausible(p0, self.now, self.s.date_max_age_days)
        out.plausible = plausible
        if trusted:
            conf = "high" if (self._ind(c for c, _ in day_items) >= 2 or not p0.date_only) else "medium"
        else:
            conf = "medium" if self._ind(c for c, _ in day_items) >= 2 else "low"
        if not plausible:
            self.warn("DATE_IMPLAUSIBLE")
            conf = "low"
        out.confidence = conf
        return out

    # ---------------------------------------------------------------- tracking
    _LABEL_PRIORITY = ("پیگیری", "مرجع", "رهگیری", "ارجاع", "تراکنش", "سند")

    def tracking(self, cands: list[Candidate]) -> TrackingField:
        ts = [c for c in cands if c.field == "tracking_number" and digits_only(str(c.value))]
        out = TrackingField(candidates=_pub(ts))
        if not ts:
            return out
        seen = {}
        for c in ts:
            v = digits_only(str(c.value))
            seen.setdefault((v, c.label), c)
        out.all = [{"value": v, "label": lbl} for (v, lbl) in seen]

        def prio(c: Candidate) -> int:
            lbl = c.label or ""
            for i, k in enumerate(self._LABEL_PRIORITY):
                if k in lbl:
                    return i
            return len(self._LABEL_PRIORITY)

        best_p = min(prio(c) for c in ts)
        group = [c for c in ts if prio(c) == best_p]
        trusted = [c for c in group if c.source in TRUSTED]
        pool = trusted or group
        vals = {digits_only(str(c.value)) for c in pool}
        if len(vals) > 1:
            self.warn("TRACKING_CONFLICT")
            out.confidence = "conflict"
            return out
        v = vals.pop()
        agree = [c for c in pool if digits_only(str(c.value)) == v]
        out.value, out.label = v, agree[0].label
        valid = 6 <= len(v) <= 30
        if trusted:
            conf = "high" if valid else "low"  # printed/digital + format validator
        else:
            conf = "medium" if (self._ind(agree) >= 2 and valid) else "low"
        out.confidence = conf
        return out

    # ------------------------------------------------------------------- names
    def text_field(self, cands: list[Candidate], name: str) -> TextField:
        cs = [c for c in cands if c.field == name and c.value]
        out = TextField(candidates=_pub(cs))
        if not cs:
            return out
        best = Counter(normalize_name(c.value) for c in cs).most_common(1)[0][0]
        rep = next(c for c in cs if normalize_name(c.value) == best)
        agree = [c for c in cs if fuzz.token_sort_ratio(normalize_name(c.value), best) >= self.s.name_match_threshold]
        disagree = [c for c in cs if c not in agree]
        out.value = rep.value
        if disagree and self._ind(disagree) >= 1 and self._ind(agree) == 1:
            out.confidence = "conflict"
        elif self._ind(agree) >= 2 and any(c.source in TRUSTED for c in agree):
            out.confidence = "high"
        elif any(c.source in TRUSTED for c in agree) or self._ind(agree) >= 2:
            out.confidence = "medium"
        else:
            out.confidence = "low"
        return out

    # ------------------------------------------------------------------- enums
    @staticmethod
    def _vote(cands: list[Candidate], field: str, votes: dict[str, int]) -> dict[str, int]:
        v = dict(votes)
        for c in cands:
            if c.field == field and c.value:
                v[c.value] = v.get(c.value, 0) + 2  # structured reading weighs more than a keyword
        return v

    def transfer_type(self, cands, hints: Hints) -> str:
        v = self._vote(cands, "transfer_type", hints.transfer_votes)
        v.pop("unknown", None)
        if not v:
            return "unknown"
        ranked = sorted(v.items(), key=lambda kv: -kv[1])
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return "unknown"
        return ranked[0][0]

    def document_status(self, cands, hints: Hints, tracking: TrackingField) -> str:
        v = self._vote(cands, "document_status", hints.status_votes)
        if v.get("failed"):
            return "failed"
        if v.get("pending"):
            return "pending"
        if v.get("success"):
            return "success"
        if (self.s.printed_strip_implies_success and hints.printed_strip_found
                and tracking.value and any(c.source == "printed" and c.pass_id == "strip"
                                           for c in cands if c.field == "tracking_number")):
            return "success"
        return "unknown"


def resolve(cands: list[Candidate], hints: Hints, settings: Settings, now: datetime) -> tuple[Fields, list[str]]:
    r = Resolver(settings, now)
    amount, currency, toman = r.amount(cands)
    f = Fields()
    f.amount_rial = amount
    f.currency = currency
    f.amount_toman = toman
    f.dest_iban = r.iban(cands)
    f.dest_bank = f.dest_iban.bank
    f.paid_at = r.paid_at(cands)
    f.tracking_number = r.tracking(cands)
    f.transfer_type = r.transfer_type(cands, hints)
    f.document_status = r.document_status(cands, hints, f.tracking_number)
    f.dest_holder_name = r.text_field(cands, "dest_holder_name")
    f.source_name = r.text_field(cands, "source_name")
    f.source_account = r.text_field(cands, "source_account")
    words = [c for c in cands if c.field == "amount" and c.source == "words"]
    if words:
        f.amount_words = TextField(value=words[0].raw_text, confidence="medium" if len(words) > 1 else "low",
                                   candidates=_pub(words))
    if f.amount_rial.value is not None and currency == "unknown":
        r.warn("UNKNOWN_CURRENCY")
    if f.document_status != "success":
        r.warn("STATUS_NOT_SUCCESS")
    return f, r.warnings
