"""End-to-end on synthetic images through the real local OCR engine.

Synthetic images exercise the plumbing only. They say nothing about accuracy on
real receipts and must never be used to report accuracy.
"""
import os

import pytest

from receipt_reader.config import Settings
from receipt_reader.engines import build_engines
from receipt_reader.pipeline import ReceiptReader
from receipt_reader.schema import ExpectedLine

from .conftest import IBAN_MELLAT, IBAN_MELLI, NOW
from .synth import SatnaSpec, fonts_available, photograph, render_satna_form, render_screenshot, to_bytes

TESSDATA = os.environ.get("RR_TESSDATA_PREFIX") or os.environ.get("TESSDATA_PREFIX") or (
    os.path.expanduser("~/tessdata") if os.path.exists(os.path.expanduser("~/tessdata/fas.traineddata")) else None)


@pytest.fixture(scope="module")
def reader():
    s = Settings(_env_file=None, engines="tesseract", tessdata_prefix=TESSDATA)
    r = ReceiptReader(s, build_engines(s))
    ok, why = r.status["tesseract"]
    if not ok:
        pytest.skip(f"tesseract not available: {why}")
    if not fonts_available():
        pytest.skip("no Persian-capable font for synthetic rendering")
    return r


LINES = [ExpectedLine(line_id="L-17", iban=IBAN_MELLAT, amount_toman=450_000_000, holder_name="علی کریمی"),
         ExpectedLine(line_id="L-18", iban=IBAN_MELLI, amount_toman=450_000_000)]


def test_satna_form_printed_strip_beats_handwriting(reader):
    """The real failure: handwritten 1,500,000,000-style misreading must not be passed on."""
    spec = SatnaSpec(dest_iban=IBAN_MELLAT, handwritten_amount_rial=15_000_000_000)
    img = to_bytes(photograph(render_satna_form(spec)))
    res = reader.read(img, LINES, now=NOW, include_debug=True)
    f = res.fields
    assert res.debug["printed_strip_box"] is not None
    # owner decision: the disagreement is a conflict -> manager review; printed is only suggested
    assert f.amount_rial.value is None and f.amount_rial.confidence == "conflict"
    assert f.amount_rial.suggested_value == 4_500_000_000
    assert res.suggested_route == "manager_review"
    assert "HANDWRITTEN_AMOUNT_DISAGREES" in res.warnings
    assert any(c["value"] == 15_000_000_000 and c["source"] == "handwritten" for c in f.amount_rial.candidates)
    assert f.dest_iban.value == IBAN_MELLAT and f.dest_iban.checksum_valid
    assert f.paid_at.jalali == "1405/08/24 11:42"
    assert res.best_line_id is None  # no amount -> no line is picked


def test_satna_without_strip_handwriting_only(reader):
    spec = SatnaSpec(dest_iban=IBAN_MELLAT, handwritten_amount_rial=4_500_000_000, strip=False)
    res = reader.read(to_bytes(photograph(render_satna_form(spec))), LINES, now=NOW)
    # without a teller strip nothing can be "match": no printed source, no date, no tracking
    assert res.suggested_route != "match"


def test_screenshot(reader):
    img = render_screenshot(amount_rial=4_500_000_000, dest_iban=IBAN_MELLAT)
    res = reader.read(to_bytes(img, "PNG"), LINES, now=NOW)
    assert res.image.kind == "screenshot"
    assert res.fields.dest_iban.value == IBAN_MELLAT
    assert res.fields.tracking_number.value == "987654321012"
    assert res.suggested_route != "match" or res.best_line_id == "L-17"


def test_wrong_line_never_matches(reader):
    """Receipt pays Mellat, debtor only has a Melli line: must go to a human."""
    spec = SatnaSpec(dest_iban=IBAN_MELLAT)
    res = reader.read(to_bytes(photograph(render_satna_form(spec))), [LINES[1]], now=NOW)
    assert res.suggested_route == "manager_review"
    assert res.best_line_id is None
