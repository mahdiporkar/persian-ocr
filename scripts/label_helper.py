"""Helpers for building the real labelled set in eval/real/.

    python scripts/label_helper.py strip-gps eval/real       # remove GPS location from images (in place)
    python scripts/label_helper.py todo eval/real            # write <stem>.todo.json for unlabelled images
    python scripts/label_helper.py check eval/real           # validate every <stem>.json

`todo` runs the reader and puts its output under "_reader_suggestion" ONLY. The ground-truth
fields stay null: fill them from the image itself, not from the suggestion (copying the
reader's answer would make the evaluation measure nothing). Rename to <stem>.json when done.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receipt_reader.dates import parse_datetime  # noqa: E402
from receipt_reader.iban import parse_iban  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
TRUTH_KEYS = ["amount_rial", "currency", "dest_iban", "paid_at_jalali", "tracking_number", "transfer_type",
              "document_status"]
ENUMS = {
    "currency": {"rial", "toman", None},
    "transfer_type": {"satna", "paya", "internal", "card_to_card", "unknown", None},
    "document_status": {"success", "failed", "pending", "unknown", None},
}
SOURCES = {"printed", "handwritten", "digital", "words"}


def images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)


def cmd_strip_gps(folder: Path) -> int:
    from PIL import Image

    n = 0
    for p in images(folder):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".webp"):
            continue
        img = Image.open(p)
        exif = img.getexif()
        if 0x8825 not in exif:  # GPSInfo
            continue
        del exif[0x8825]
        buf = io.BytesIO()
        fmt = "JPEG" if p.suffix.lower() in (".jpg", ".jpeg") else "WEBP"
        # keep orientation / Software / Make / Model: the reader uses them
        img.save(buf, format=fmt, exif=exif.tobytes(), quality=95)
        p.write_bytes(buf.getvalue())
        n += 1
        print(f"removed GPS from {p.name}")
    print(f"{n} file(s) changed")
    return 0


def cmd_todo(folder: Path) -> int:
    from receipt_reader.config import Settings
    from receipt_reader.engines import build_engines
    from receipt_reader.pipeline import ReceiptReader

    s = Settings()
    reader = ReceiptReader(s, build_engines(s))
    for p in images(folder):
        if p.with_suffix(".json").exists() or p.with_suffix(".todo.json").exists():
            continue
        res = reader.read(p.read_bytes(), [])
        f = res.fields
        todo = {
            **{k: None for k in TRUTH_KEYS},
            "sources": {"amount_rial": None},
            "expected_lines": [],
            "correct_line_id": None,
            "evaluated_at": None,
            "notes": "",
            "_reader_suggestion": {
                "amount_rial": f.amount_rial.value, "currency": f.currency, "dest_iban": f.dest_iban.value,
                "paid_at_jalali": f.paid_at.jalali, "tracking_number": f.tracking_number.value,
                "transfer_type": f.transfer_type, "document_status": f.document_status,
                "warnings": res.warnings,
            },
        }
        p.with_suffix(".todo.json").write_text(json.dumps(todo, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {p.with_suffix('.todo.json').name}")
    return 0


def check_one(gt: dict) -> list[str]:
    errs = []
    for k in TRUTH_KEYS:
        if k not in gt:
            errs.append(f"missing key {k} (use null if it is not on the receipt)")
    a = gt.get("amount_rial")
    if a is not None and (not isinstance(a, int) or isinstance(a, bool) or a <= 0):
        errs.append("amount_rial must be a positive integer (rial, no separators, no float)")
    iban = gt.get("dest_iban")
    if iban:
        info = parse_iban(iban)
        if info is None:
            errs.append(f"dest_iban not IBAN-shaped: {iban}")
        elif not info.partial and not info.checksum_valid:
            errs.append(f"dest_iban fails the mod-97 checksum: {iban} (typo in the label?)")
    d = gt.get("paid_at_jalali")
    if d and parse_datetime(d) is None:
        errs.append(f"paid_at_jalali not parseable: {d}")
    t = gt.get("tracking_number")
    if t is not None and not str(t).isdigit():
        errs.append("tracking_number must be digits only (as a string)")
    for k, allowed in ENUMS.items():
        if gt.get(k) not in allowed:
            errs.append(f"{k} must be one of {sorted(x for x in allowed if x)}")
    for k, v in (gt.get("sources") or {}).items():
        if v is not None and v not in SOURCES:
            errs.append(f"sources.{k} must be one of {sorted(SOURCES)}")
    ids = [ln.get("line_id") for ln in gt.get("expected_lines", [])]
    for ln in gt.get("expected_lines", []):
        if not isinstance(ln.get("amount_toman"), int) or ln["amount_toman"] <= 0:
            errs.append(f"line {ln.get('line_id')}: amount_toman must be a positive integer")
        if ln.get("iban"):
            info = parse_iban(ln["iban"])
            if info is None or (not info.partial and not info.checksum_valid):
                errs.append(f"line {ln.get('line_id')}: invalid IBAN {ln['iban']}")
    if "correct_line_id" in gt and gt["correct_line_id"] is not None and gt["correct_line_id"] not in ids:
        errs.append(f"correct_line_id {gt['correct_line_id']} is not in expected_lines")
    if not gt.get("expected_lines"):
        errs.append("expected_lines is empty: the dangerous-error rate needs the lines the bot would send")
    return errs


def cmd_check(folder: Path) -> int:
    bad = total = 0
    for p in images(folder):
        gt_path = p.with_suffix(".json")
        if not gt_path.exists():
            print(f"UNLABELLED {p.name}")
            bad += 1
            continue
        total += 1
        try:
            gt = json.loads(gt_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"BAD JSON   {gt_path.name}: {e}")
            bad += 1
            continue
        errs = check_one(gt)
        if errs:
            bad += 1
            print(f"PROBLEMS   {gt_path.name}")
            for e in errs:
                print(f"   - {e}")
    print(f"{total} labelled, {bad} with problems")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["strip-gps", "todo", "check"])
    ap.add_argument("folder", type=Path)
    a = ap.parse_args(argv)
    return {"strip-gps": cmd_strip_gps, "todo": cmd_todo, "check": cmd_check}[a.command](a.folder)


if __name__ == "__main__":
    raise SystemExit(main())
