"""Iranian IBAN (شبا) handling: normalization, ISO 13616 mod-97, bank lookup."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .normalize import normalize_text

IR_IBAN_LEN = 26  # "IR" + 24 digits
_DATA = Path(__file__).parent / "data" / "banks.json"


@dataclass(frozen=True)
class IbanInfo:
    value: str                 # "IR" + 24 digits, or "...8550" for partial
    partial: bool
    checksum_valid: Optional[bool]  # None for partial
    bank_code: Optional[str]
    bank: Optional[str]        # English name
    bank_fa: Optional[str]

    @property
    def last_digits(self) -> str:
        return re.sub(r"\D", "", self.value)[-6:]


def mod97_valid(iban: str) -> bool:
    """ISO 13616 / ISO 7064 mod-97-10 check."""
    s = iban.replace(" ", "").upper()
    if len(s) < 5 or not s[:2].isalpha():
        return False
    rearranged = s[4:] + s[:4]
    digits = "".join(str(int(c, 36)) for c in rearranged)
    if not digits.isdigit():
        return False
    return int(digits) % 97 == 1


def compute_check_digits(bban: str, country: str = "IR") -> str:
    """Check digits for a BBAN (used by tests and synthetic data generation)."""
    rearranged = bban + country + "00"
    digits = "".join(str(int(c, 36)) for c in rearranged)
    return f"{98 - int(digits) % 97:02d}"


def make_iban(bban24: str) -> str:
    assert len(bban24) == 22 or len(bban24) == 24
    bban = bban24[-22:]
    return "IR" + compute_check_digits(bban) + bban


@lru_cache(maxsize=1)
def bank_table() -> dict[str, dict]:
    data = json.loads(_DATA.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def bank_for_code(code: Optional[str]) -> Optional[dict]:
    if not code:
        return None
    return bank_table().get(code)


def detect_bank_names(text: str) -> set[str]:
    """Bank codes whose name/alias appears in the given (document) text."""
    t = normalize_text(text).lower()
    found = set()
    for code, info in bank_table().items():
        for alias in info.get("aliases", []):
            a = normalize_text(alias).lower()
            if a and (re.search(rf"\b{re.escape(a)}\b", t) if a.isascii() else a in t):
                found.add(code)
                break
    return found


_PARTIAL_RE = re.compile(r"(?:IR)?[\s\-]*[*xX•.…]{2,}[\s\-*xX•.…]*(\d{3,8})\s*$", re.I)


def parse_iban(raw: Optional[str]) -> Optional[IbanInfo]:
    """Normalize one IBAN-like string.

    Accepts "IR06 0120 0000 ...", "IR-06-0120...", "ir060120...", the 24 digits
    without the IR prefix, and masked/partial forms like "IR**********8550" or
    "...8550". Returns None when the string is not IBAN-shaped.
    """
    if not raw:
        return None
    s = normalize_text(raw).upper()
    s = s.replace("I R", "IR")
    compact = re.sub(r"[\s\-_/.]", "", s)
    m = _PARTIAL_RE.search(s) or _PARTIAL_RE.search(re.sub(r"[\s\-_/]", "", s))
    if re.fullmatch(r"IR\d{24}", compact):
        full = compact
    elif re.fullmatch(r"\d{24}", compact):
        full = "IR" + compact
    elif m:
        return IbanInfo(value="..." + m.group(1), partial=True, checksum_valid=None,
                        bank_code=None, bank=None, bank_fa=None)
    else:
        return None
    code = full[4:7]
    info = bank_for_code(code)
    return IbanInfo(
        value=full,
        partial=False,
        checksum_valid=mod97_valid(full),
        bank_code=code,
        bank=info["en"] if info else None,
        bank_fa=info["fa"] if info else None,
    )


# IBAN-shaped substrings inside free text (24 digits with optional IR and spacing)
_IBAN_IN_TEXT = re.compile(r"(?:IR[\s\-]*)?(?:\d[\s\-]*){24}(?!\d)", re.I)
# A masked IBAN must carry the IR prefix: masked card numbers look like "6037-70xx-xxxx-5190".
_PARTIAL_IR_IN_TEXT = re.compile(r"IR[\s\-]*[\d\s\-]{0,22}[*xX•]{2,}[\s\-*xX•\d]*\d", re.I)
_PARTIAL_BARE_IN_TEXT = re.compile(r"[*xX•]{4,}[\s\-]*\d{3,8}\b")


def find_ibans(text: str, iban_label: bool = False) -> list[tuple[str, IbanInfo]]:
    """Return (raw_substring, info) for every IBAN-like run in the text.

    A masked/partial IBAN is accepted only with the IR prefix, or without it when the
    line is explicitly labelled شبا/IBAN (iban_label=True). A partial IBAN never gets a
    checksum result (checksum_valid stays None)."""
    t = normalize_text(text)
    out: list[tuple[str, IbanInfo]] = []
    for m in _IBAN_IN_TEXT.finditer(t.upper()):
        raw = m.group(0).strip(" -")
        info = parse_iban(raw)
        if info:
            out.append((raw, info))
    if not out:
        pats = [_PARTIAL_IR_IN_TEXT] + ([_PARTIAL_BARE_IN_TEXT] if iban_label else [])
        for pat in pats:
            for m in pat.finditer(t):
                info = parse_iban(m.group(0))
                if info and info.partial:
                    out.append((m.group(0).strip(), info))
            if out:
                break
    return out


def mask(value: Optional[str], keep: int = 4) -> str:
    if not value:
        return ""
    v = re.sub(r"\s", "", value)
    if len(v) <= keep:
        return v
    return "*" * (len(v) - keep) + v[-keep:]
