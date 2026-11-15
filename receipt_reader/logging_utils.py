"""Logging with masking of IBANs, card numbers and national IDs (last 4 digits kept)."""
from __future__ import annotations

import json
import logging
import re

from .normalize import to_ascii_digits

_IBAN = re.compile(r"\bIR[\s\-]*((?:\d[\s\-]*){20})((?:\d[\s\-]*){3}\d)\b", re.I)
_LONG_DIGITS = re.compile(r"(?<!\d)(\d[\d\s\-]{8,40}\d)(?!\d)")


def _mask_digits(m: re.Match) -> str:
    s = m.group(0)
    d = re.sub(r"\D", "", s)
    if len(d) < 10:  # amounts, dates and short codes are not sensitive
        return s
    return "*" * (len(d) - 4) + d[-4:]


def mask_sensitive(text: str) -> str:
    t = to_ascii_digits(text)
    t = _IBAN.sub(lambda m: "IR" + "*" * 20 + re.sub(r"\D", "", m.group(2)), t)
    t = _LONG_DIGITS.sub(_mask_digits, t)
    return t


class MaskingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        record.msg = mask_sensitive(msg)
        record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": self.formatTime(record), "level": record.levelname, "logger": record.name,
             "msg": record.getMessage()}
        rid = getattr(record, "request_id", None)
        if rid:
            d["request_id"] = rid
        return json.dumps(d, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(JsonFormatter())
    h.addFilter(MaskingFilter())
    root.handlers = [h]
