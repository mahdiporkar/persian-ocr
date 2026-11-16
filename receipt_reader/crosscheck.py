"""Cross-check candidates from all engines and assign confidence levels.

Exact rules: README.md, "Confidence rules". Key principles (owner decisions):

* Independent evidence = DIFFERENT ENGINES. Two passes of the same engine (full page
  and zoomed strip) are one source of evidence.
* Any disagreement between amount readings is a conflict: no value is picked, every
  candidate is kept, the route becomes manager_review. A printed/digital reading may be
  offered as `suggested_value` for the human, never used for matching.
* document_status reports what the document says; it never claims settlement.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional

from rapidfuzz import fuzz

from .config import Settings
from .dates import is_plausible, parse_datetime
from .iban import bank_for_code, parse_iban
from .normalize import digits_only, normalize_name
from .schema import (AmountField, Candidate, DateField, Fields, IbanField, Party, TextField,
                     TrackingField)

TRUSTED = ("printed", "digital")
_LEVEL = {"conflict": -1, "low": 0, "medium": 1, "high": 2}

TRACKING_MIN_DIGITS = 5   # card-to-card tracking numbers are often 5-6 digits (leading zero dropped)
TRACKING_MAX_DIGITS = 30


def _cap(level: str, cap: str) -> str:
    if level == "conflict":
        return level
    return level if _LEVEL[level] <= _LEVEL[cap] else cap


def engines_of(cs: Iterable[Candidate]) -> int:
    """Number of independent sources = number of distinct engines."""
    return len({c.engine for c in cs})


def _pub(cs: Iterable[Candidate]) -> list[dict]:
    return [c.public() for c in cs]


@dataclass
class Hints:
    transfer_votes: dict[str, int]
    status_votes: dict[str, int]
    issuer_banks: set[str] = field(default_factory=set)


class Resolver:
    def __init__(self, settings: Settings, now: datetime):
        self.s = settings
        self.now = now
        self.warnings: list[str] = []

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

        def to_rial(v: int) -> int:
            return v * 10 if currency == "toman" else v

        words = [c for c in amts if c.source == "words"]
        trusted = [c for c in amts if c.source in TRUSTED]
        hand = [c for c in amts if c.source == "handwritten"]
        values = {c.value for c in amts}

        if len(values) > 1:
            # any disagreement -> conflict, whatever the sources
            self.warn("AMOUNT_CONFLICT")
            tvals = {c.value for c in trusted}
            if len(tvals) > 1:
                self.warn("MULTIPLE_AMOUNTS")
            if len(tvals) == 1:
                tv = next(iter(tvals))
                if any(c.value != tv for c in hand):
                    self.warn("HANDWRITTEN_AMOUNT_DISAGREES")
                out.suggested_value = to_rial(tv)
                out.suggestion_basis = ("the only printed/digital reading; other readings disagree "
                                        "(see candidates) — a human must confirm")
            out.confidence = "conflict"
            return out, currency, None

        v = next(iter(values))
        if trusted:
            conf = "high" if (engines_of(trusted) >= 2 or words) else "medium"
        elif hand:
            # handwriting only is capped at medium, even when the words agree
            conf = "medium" if (words or engines_of(hand) >= 2) else "low"
        else:
            conf = "low"  # words only
        rial = to_rial(v)
        if not (self.s.min_amount_rial <= rial <= self.s.max_amount_rial):
            conf = _cap(conf, "low")
        out.value = rial
        out.confidence = conf
        toman = rial // 10 if rial % 10 == 0 else None
        return out, currency, toman

    # -------------------------------------------------------------------- IBAN
    def iban(self, cands: list[Candidate]) -> IbanField:
        ibs = [c for c in cands if c.field == "iban"]
        source_vals = {c.value for c in ibs if c.role == "source"}
        dest = [c for c in ibs if c.role == "dest"]
        unknown = [c for c in ibs if c.role in (None, "unknown") and c.value not in source_vals]
        pool = dest or unknown
        out = IbanField(candidates=_pub(c for c in ibs if c.role != "source"))
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
            agree = [c for c, _ in full_valid]
            for _, q in partial:
                if not v.endswith(q.value.lstrip(".")):
                    self.warn("IBAN_CONFLICT")
                    out.confidence = "conflict"
                    return out
            agree_trusted = [c for c in agree if c.source in TRUSTED]
            known_bank = bank_for_code(p.bank_code) is not None
            if agree_trusted and (engines_of(agree) >= 2 or known_bank):
                conf = "high"   # mod-97 + known bank code are strong validators for one printed read
            elif agree_trusted or engines_of(agree) >= 2:
                conf = "medium"
            else:
                conf = "low"
            named = {c.value for c in cands if c.field == "receiver_bank"}
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
        # partial only: never checksum-validated
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
        if len({p.jalali for _, p in timed}) > 1:
            self.warn("DATE_CONFLICT")
            out.confidence = "conflict"
            return out
        if trusted and any(p.iso[:10] != day_items[0][1].iso[:10] for c, p in ds if c.source == "handwritten"):
            self.warn("DATE_CONFLICT")
            out.confidence = "conflict"
            return out
        c0, p0 = (timed or day_items)[0]
        out.jalali, out.iso = p0.jalali, p0.iso
        plausible = is_plausible(p0, self.now, self.s.date_max_age_days)
        out.plausible = plausible
        n_eng = engines_of(c for c, _ in day_items)
        if trusted:
            conf = "high" if n_eng >= 2 else "medium"
        else:
            conf = "medium" if n_eng >= 2 else "low"
        if not plausible:
            self.warn("DATE_IMPLAUSIBLE")
            conf = "low"
        out.confidence = conf
        return out

    # ---------------------------------------------------------------- tracking
    _LABEL_PRIORITY = ("پیگیری", "مرجع", "رهگیری", "ارجاع", "تراکنش", "سند")

    def tracking(self, cands: list[Candidate]) -> TrackingField:
        ts = [c for c in cands if c.field == "tracking_number"
              and TRACKING_MIN_DIGITS <= len(digits_only(str(c.value))) <= TRACKING_MAX_DIGITS]
        out = TrackingField(candidates=_pub(ts))
        if not ts:
            return out
        seen = {}
        for c in ts:
            seen.setdefault((digits_only(str(c.value)), c.label), c)
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
        # a tracking number has no checksum: one engine's read is never "high"
        if trusted:
            out.confidence = "high" if engines_of(agree) >= 2 else "medium"
        else:
            out.confidence = "medium" if engines_of(agree) >= 2 else "low"
        return out

    # ------------------------------------------------------------ text fields
    def text_field(self, cands: list[Candidate], name: str, key=normalize_name, display=None) -> TextField:
        cs = [c for c in cands if c.field == name and c.value]
        out = TextField(candidates=_pub(cs))
        if not cs:
            return out
        best = Counter(key(c.value) for c in cs).most_common(1)[0][0]
        rep = next(c for c in cs if key(c.value) == best)
        agree = [c for c in cs if fuzz.token_sort_ratio(key(c.value), best) >= self.s.name_match_threshold]
        disagree = [c for c in cs if c not in agree]
        out.value = display(rep.value) if display else rep.value
        if disagree and engines_of(agree) == 1 and engines_of(disagree) >= 1:
            out.confidence = "conflict"
            out.value = None  # readings disagree: keep all candidates, pick none
        elif engines_of(agree) >= 2 and any(c.source in TRUSTED for c in agree):
            out.confidence = "high"
        elif any(c.source in TRUSTED for c in agree) or engines_of(agree) >= 2:
            out.confidence = "medium"
        else:
            out.confidence = "low"
        return out

    def party(self, cands: list[Candidate], side: str) -> Party:
        def last4(v: str) -> str:
            return digits_only(v)[-4:]

        def bank_name(code: str) -> str:
            info = bank_for_code(code)
            return info["en"] if info else code

        return Party(
            name=self.text_field(cands, f"{side}_name"),
            account=self.text_field(cands, f"{side}_account", key=digits_only),
            card=self.text_field(cands, f"{side}_card", key=last4, display=lambda v: "…" + last4(v)),
            iban=(self.text_field([c for c in cands if c.field == "iban" and c.role == "source"], "iban",
                                  key=lambda v: v) if side == "sender" else TextField()),
            bank=self.text_field(cands, f"{side}_bank", key=lambda v: v, display=bank_name),
        )

    # ------------------------------------------------------------------- enums
    @staticmethod
    def _vote(cands: list[Candidate], field: str, votes: dict[str, int]) -> dict[str, int]:
        v = dict(votes)
        for c in cands:
            if c.field == field and c.value:
                v[c.value] = v.get(c.value, 0) + 2  # a structured reading weighs more than a keyword
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

    def document_status(self, cands, hints: Hints) -> str:
        """Only explicit words on the document count. A printed tracking number or a teller
        strip is NOT evidence of success (owner decision)."""
        v = self._vote(cands, "document_status", hints.status_votes)
        if v.get("failed"):
            return "failed"
        if v.get("pending"):
            return "pending"
        if v.get("success"):
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
    f.document_status = r.document_status(cands, hints)
    f.sender = r.party(cands, "sender")
    f.receiver = r.party(cands, "receiver")
    if f.dest_iban.value:
        f.receiver.iban = TextField(value=f.dest_iban.value, confidence=f.dest_iban.confidence)
    # a party's bank not named on the document: from its checksum-valid IBAN's bank code
    if f.receiver.bank.value is None and f.dest_iban.bank and f.dest_iban.checksum_valid:
        f.receiver.bank = TextField(value=f.dest_iban.bank, confidence=f.dest_iban.confidence,
                                    candidates=[{"value": f.dest_iban.bank, "source": "iban_bank_code"}])
    if f.sender.bank.value is None and f.sender.iban.value:
        info = parse_iban(f.sender.iban.value)
        if info and info.checksum_valid and info.bank:
            f.sender.bank = TextField(value=info.bank, confidence=f.sender.iban.confidence,
                                      candidates=[{"value": info.bank, "source": "iban_bank_code"}])
    # ... else the bank that issued the receipt (header/footer), which on bank-issued receipts
    # is the payer's bank: only ever "low", and only when exactly one bank is named there
    if f.sender.bank.value is None and len(hints.issuer_banks) == 1:
        info = bank_for_code(next(iter(hints.issuer_banks)))
        if info:
            f.sender.bank = TextField(value=info["en"], confidence="low",
                                      candidates=[{"value": info["en"], "source": "issuer_header"}])
    f.purpose = r.text_field(cands, "purpose")
    words = [c for c in cands if c.field == "amount" and c.source == "words"]
    if words:
        f.amount_words = TextField(value=words[0].raw_text, confidence="medium" if engines_of(words) > 1 else "low",
                                   candidates=_pub(words))
    if (f.amount_rial.value is not None or f.amount_rial.suggested_value is not None) and currency == "unknown":
        r.warn("UNKNOWN_CURRENCY")
    if f.document_status != "success":
        r.warn("STATUS_NOT_SUCCESS")
    return f, r.warnings
