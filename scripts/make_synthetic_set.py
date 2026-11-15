"""Generate a small synthetic labelled set in eval/synthetic/ (plumbing checks and the demo only).

    python scripts/make_synthetic_set.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receipt_reader.iban import make_iban  # noqa: E402
from tests.synth import SatnaSpec, photograph, render_satna_form, render_screenshot  # noqa: E402

OUT = ROOT / "eval" / "synthetic"
MELLAT = make_iban("0120000000001234568550")
MELLI = make_iban("0170000000003456781234")
SAMAN = make_iban("0560000000009876543210")
EVAL_AT = "2026-11-15T12:00:00+03:30"


def write(name, img, gt, fmt="JPEG"):
    OUT.mkdir(parents=True, exist_ok=True)
    ext = ".jpg" if fmt == "JPEG" else ".png"
    img.save(OUT / f"{name}{ext}", format=fmt, **({"quality": 88} if fmt == "JPEG" else {}))
    gt = {"synthetic": True, "evaluated_at": EVAL_AT, **gt}
    (OUT / f"{name}.json").write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    lines = [
        {"line_id": "L-17", "iban": MELLAT, "holder_name": "علی کریمی", "amount_toman": 450_000_000,
         "remaining_toman": 450_000_000},
        {"line_id": "L-18", "iban": MELLI, "holder_name": "سارا محمدی", "amount_toman": 200_000_000,
         "remaining_toman": 200_000_000},
    ]
    # 1. the real-world failure: handwritten amount wrong, printed strip right
    spec = SatnaSpec(dest_iban=MELLAT, handwritten_amount_rial=15_000_000_000)
    write("satna_handwriting_wrong", photograph(render_satna_form(spec), seed=1, angle=2.0), {
        "amount_rial": 4_500_000_000, "currency": "rial", "dest_iban": MELLAT, "paid_at_jalali": "1405/08/24 11:42",
        "tracking_number": "140508241234567", "transfer_type": "satna", "document_status": "success",
        "sources": {"amount_rial": "printed"}, "expected_lines": lines, "correct_line_id": "L-17"})
    # 2. clean form, everything agrees
    spec = SatnaSpec(dest_iban=MELLAT, tracking="140508249876543")
    write("satna_clean", photograph(render_satna_form(spec), seed=2, angle=-1.5), {
        "amount_rial": 4_500_000_000, "currency": "rial", "dest_iban": MELLAT, "paid_at_jalali": "1405/08/24 11:42",
        "tracking_number": "140508249876543", "transfer_type": "satna", "document_status": "success",
        "sources": {"amount_rial": "printed"}, "expected_lines": lines, "correct_line_id": "L-17"})
    # 3. form paying an account that is not one of the debtor's lines
    spec = SatnaSpec(dest_iban=SAMAN, printed_amount_rial=2_000_000_000, handwritten_amount_rial=2_000_000_000,
                     words="دو میلیارد ریال", bank_header="بانک سامان")
    write("satna_wrong_account", photograph(render_satna_form(spec), seed=3, angle=1.0), {
        "amount_rial": 2_000_000_000, "currency": "rial", "dest_iban": SAMAN, "paid_at_jalali": "1405/08/24 11:42",
        "tracking_number": "140508241234567", "transfer_type": "satna", "document_status": "success",
        "expected_lines": lines, "correct_line_id": None})
    # 4. mobile-banking PAYA screenshot
    write("paya_screenshot", render_screenshot(amount_rial=2_000_000_000, dest_iban=MELLI), {
        "amount_rial": 2_000_000_000, "currency": "rial", "dest_iban": MELLI, "paid_at_jalali": "1405/08/24 09:15",
        "tracking_number": "987654321012", "transfer_type": "paya", "document_status": "success",
        "sources": {"amount_rial": "digital"}, "expected_lines": lines, "correct_line_id": "L-18"}, fmt="PNG")
    # 5. failed transfer screenshot
    write("failed_screenshot", render_screenshot(amount_rial=4_500_000_000, dest_iban=MELLAT, status="ناموفق"), {
        "amount_rial": 4_500_000_000, "currency": "rial", "dest_iban": MELLAT, "paid_at_jalali": "1405/08/24 09:15",
        "tracking_number": "987654321012", "transfer_type": "paya", "document_status": "failed",
        "expected_lines": lines, "correct_line_id": None}, fmt="PNG")
    print(f"wrote synthetic set to {OUT}")


if __name__ == "__main__":
    main()
