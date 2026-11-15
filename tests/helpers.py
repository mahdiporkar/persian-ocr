from receipt_reader.crosscheck import Hints, resolve
from receipt_reader.matcher import decide
from receipt_reader.schema import Candidate, ExpectedLine


def C(field, value, source="printed", engine="a", pass_id="full", **kw):
    return Candidate(field=field, value=value, raw_text=str(value), source=source, engine=engine,
                     pass_id=pass_id, **kw)


def good_printed(iban, amount=4_500_000_000, tracking="123456789012", date="1405/08/24 11:42"):
    """Two engines reading the same clean printed strip."""
    out = []
    for eng in ("a", "b"):
        out += [
            C("amount", amount, engine=eng, currency="rial"),
            C("iban", iban, engine=eng, role="dest"),
            C("paid_at", date, engine=eng),
            C("tracking_number", tracking, engine=eng, label="شماره پیگیری"),
            C("transfer_type", "satna", engine=eng),
            C("document_status", "success", engine=eng),
        ]
    return out


def run(cands, settings, now, lines=(), flags=(), strip=True):
    hints = Hints(transfer_votes={}, status_votes={}, printed_strip_found=strip)
    fields, warnings = resolve(list(cands), hints, settings, now)
    matches, best, route, q, warnings = decide(fields, warnings, list(lines), settings.name_match_threshold,
                                               list(flags))
    return fields, matches, best, route, q, warnings


def line(line_id, iban, toman, remaining=None, name=None):
    return ExpectedLine(line_id=line_id, iban=iban, amount_toman=toman, remaining_toman=remaining,
                        holder_name=name)
