import json
from pathlib import Path

from scripts.label_helper import check_one

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_labels_are_valid():
    for p in (ROOT / "eval" / "synthetic").glob("*.json"):
        assert check_one(json.loads(p.read_text(encoding="utf-8"))) == [], p.name


def test_catches_label_mistakes():
    gt = json.loads((ROOT / "eval" / "synthetic" / "satna_clean.json").read_text(encoding="utf-8"))
    gt["dest_iban"] = gt["dest_iban"][:-1] + ("0" if gt["dest_iban"][-1] != "0" else "1")
    gt["amount_rial"] = 4.5e9
    gt["correct_line_id"] = "nope"
    errs = " | ".join(check_one(gt))
    assert "mod-97" in errs and "positive integer" in errs and "not in expected_lines" in errs
