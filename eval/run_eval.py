"""Evaluate the receipt reader on a labelled set.

    python eval/run_eval.py eval/real                 # the owner's private real receipts
    python eval/run_eval.py eval/synthetic --engines tesseract

Each receipt is an image plus a ground-truth JSON with the same stem (see eval/README.md).

Reports per field: exact-match rate, null rate, wrong rate and the source confusion;
plus the two numbers that matter most:

  * dangerous-error rate — receipts routed "match" whose amount or IBAN (or chosen
    line) is wrong. Target: 0.
  * auto-match share     — receipts routed "match".

Accuracy is only meaningful on REAL receipts. If any sample is marked synthetic the
report says so in its first line and must not be quoted as accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from receipt_reader.config import Settings  # noqa: E402
from receipt_reader.dates import TEHRAN, parse_datetime  # noqa: E402
from receipt_reader.engines import build_engines  # noqa: E402
from receipt_reader.iban import parse_iban  # noqa: E402
from rapidfuzz import fuzz  # noqa: E402

from receipt_reader.normalize import digits_only, normalize_name  # noqa: E402
from receipt_reader.pipeline import ReceiptReader  # noqa: E402
from receipt_reader.schema import ExpectedLine  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
FIELDS = ["amount_rial", "currency", "dest_iban", "paid_at", "tracking_number", "transfer_type", "document_status"]
PARTY_FIELDS = [f"{side}_{k}" for side in ("sender", "receiver") for k in ("name", "card", "account", "bank")]
CONF_FIELDS = {"amount_rial": "amount_rial", "dest_iban": "dest_iban", "paid_at": "paid_at",
               "tracking_number": "tracking_number"}


def predicted(res) -> dict:
    f = res.fields
    return {
        "amount_rial": f.amount_rial.value,
        "currency": f.currency if f.currency != "unknown" else None,
        "dest_iban": f.dest_iban.value,
        "paid_at": f.paid_at.jalali,
        "tracking_number": f.tracking_number.value,
        "transfer_type": f.transfer_type if f.transfer_type != "unknown" else None,
        "document_status": f.document_status if f.document_status != "unknown" else None,
        **{f"{side}_{k}": _party_value(getattr(getattr(f, side), k).value, k)
           for side in ("sender", "receiver") for k in ("name", "card", "account", "bank")},
    }


def _party_value(v, kind):
    if v is None:
        return None
    if kind == "card":
        return digits_only(v)[-4:] or None
    if kind == "account":
        return digits_only(v)
    if kind == "name":
        return normalize_name(v)
    return v


def same(k: str, a, b) -> bool:
    if a is None or b is None:
        return a is b
    if k.endswith("_name"):
        return fuzz.token_sort_ratio(a, b) >= 90
    return a == b


def truth_of(gt: dict) -> dict:
    iban = gt.get("dest_iban")
    if iban:
        info = parse_iban(iban)
        iban = info.value if info else iban
    paid = gt.get("paid_at_jalali")
    if paid:
        p = parse_datetime(paid)
        paid = p.jalali if p else paid
    return {
        "amount_rial": gt.get("amount_rial"),
        "currency": gt.get("currency"),
        "dest_iban": iban,
        "paid_at": paid,
        "tracking_number": digits_only(gt["tracking_number"]) if gt.get("tracking_number") else None,
        "transfer_type": gt.get("transfer_type"),
        "document_status": gt.get("document_status"),
        **{f"{side}_{k}": _party_value((gt.get(side) or {}).get(k), k)
           for side in ("sender", "receiver") for k in ("name", "card", "account", "bank")},
    }


def labelled(gt: dict, k: str) -> bool:
    if k in PARTY_FIELDS:
        side, sub = k.split("_", 1)
        return sub in (gt.get(side) or {})
    return k in gt or (k == "paid_at" and "paid_at_jalali" in gt)


def chosen_source(res, field: str):
    f = getattr(res.fields, field, None)
    if f is None or not hasattr(f, "candidates"):
        return None
    val = f.value if hasattr(f, "value") else getattr(f, "jalali", None)
    for c in sorted(f.candidates, key=lambda c: c["source"] not in ("printed", "digital")):
        if c["value"] == val:
            return c["source"]
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--engines", default=None, help="comma-separated; default from config")
    ap.add_argument("--out", type=Path, default=Path("eval/reports"))
    ap.add_argument("--now", default=None, help="ISO time to evaluate date plausibility against "
                                                "(default: each sample's 'evaluated_at' or the real now)")
    args = ap.parse_args(argv)

    settings = Settings()
    reader = ReceiptReader(settings, build_engines(settings))
    engines = args.engines.split(",") if args.engines else None

    samples = sorted(p for p in args.dataset.iterdir() if p.suffix.lower() in IMAGE_EXT)
    if not samples:
        print(f"no images in {args.dataset}")
        return 2

    per_field = {k: Counter() for k in FIELDS + PARTY_FIELDS}
    confident_wrong = []
    source_conf = defaultdict(Counter)  # field -> (truth_source, predicted_source) counts
    routes = Counter()
    dangerous, rows = [], []
    synthetic = 0
    t0 = time.time()

    for img in samples:
        gt_path = img.with_suffix(".json")
        if not gt_path.exists():
            print(f"skip {img.name}: no ground truth")
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        synthetic += bool(gt.get("synthetic"))
        lines = [ExpectedLine(**ln) for ln in gt.get("expected_lines", [])]
        now = datetime.fromisoformat(args.now or gt["evaluated_at"]) if (args.now or gt.get("evaluated_at")) else None
        if now and now.tzinfo is None:
            now = now.replace(tzinfo=TEHRAN)
        res = reader.read(img.read_bytes(), lines, engines=engines, now=now)
        pred, truth = predicted(res), truth_of(gt)
        for k in FIELDS + PARTY_FIELDS:
            if not labelled(gt, k):
                continue
            ok = same(k, pred[k], truth[k])
            if not ok and k == "tracking_number" and pred[k] and gt.get("reference_numbers"):
                # a correct number read under ANOTHER label of the same receipt is still correct,
                # provided the reader reports that label
                lbl = res.fields.tracking_number.label or ""
                ok = any(v == pred[k] and lbl.split()[-1] in L for L, v in gt["reference_numbers"].items())
            if ok:
                per_field[k]["exact"] += 1           # includes "correctly empty"
            elif pred[k] is None:
                per_field[k]["null"] += 1            # missed (truth has a value)
            else:
                per_field[k]["wrong"] += 1           # wrong or invented value
                conf = getattr(res.fields, CONF_FIELDS[k]).confidence if k in CONF_FIELDS else None
                if conf == "high":
                    confident_wrong.append({"image": img.name, "field": k, "pred": pred[k], "truth": truth[k]})
            per_field[k]["n"] += 1
        for k, src in (gt.get("sources") or {}).items():
            source_conf[k][(src, chosen_source(res, k) or "none")] += 1

        routes[res.suggested_route] += 1
        is_dangerous = False
        if res.suggested_route == "match":
            wrong_amount = pred["amount_rial"] != truth["amount_rial"]
            wrong_iban = pred["dest_iban"] != truth["dest_iban"]
            wrong_line = "correct_line_id" in gt and res.best_line_id != gt["correct_line_id"]
            is_dangerous = wrong_amount or wrong_iban or wrong_line
            if is_dangerous:
                dangerous.append({"image": img.name, "wrong_amount": wrong_amount, "wrong_iban": wrong_iban,
                                  "wrong_line": wrong_line})
        rows.append({"image": img.name, "route": res.suggested_route, "best_line_id": res.best_line_id,
                     "expected_line": gt.get("correct_line_id"), "dangerous": is_dangerous,
                     "warnings": res.warnings, "engines": res.engines, "ms": res.processing_ms,
                     "pred": pred, "truth": truth})
        print(f"{img.name:40s} route={res.suggested_route:20s} dangerous={is_dangerous} {res.processing_ms}ms")

    n = len(rows)
    report = {
        "dataset": str(args.dataset),
        "synthetic_samples": synthetic,
        "is_real_measurement": synthetic == 0,
        "n": n,
        "dangerous_error_rate": len(dangerous) / n if n else None,
        "dangerous_errors": dangerous,
        "auto_match_share": routes["match"] / n if n else None,
        "confident_wrong_values": confident_wrong,
        "routes": dict(routes),
        "fields": {k: {"n": c["n"], "exact_rate": c["exact"] / c["n"] if c["n"] else None,
                       "null_rate": c["null"] / c["n"] if c["n"] else None,
                       "wrong_rate": c["wrong"] / c["n"] if c["n"] else None} for k, c in per_field.items()},
        "source_confusion": {k: {f"{a}->{b}": v for (a, b), v in c.items()} for k, c in source_conf.items()},
        "seconds": round(time.time() - t0, 1),
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_json = args.out / f"eval-{args.dataset.name}-{stamp}.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    if synthetic:
        print(f"*** {synthetic}/{n} SYNTHETIC samples — this is a plumbing check, NOT an accuracy measurement ***")
    print(f"receipts: {n}   dangerous errors: {len(dangerous)} ({report['dangerous_error_rate']:.1%})   "
          f"auto-match: {routes['match']} ({report['auto_match_share']:.1%})   routes: {dict(routes)}")
    print(f"wrong values reported with confidence=high: {len(confident_wrong)} {confident_wrong}")
    print("exact = correct value or correctly empty; null = missed; wrong = wrong or invented value")
    print(f"{'field':18s} {'n':>4s} {'exact':>7s} {'null':>7s} {'wrong':>7s}")
    for k, v in report["fields"].items():
        if v["n"]:
            print(f"{k:18s} {v['n']:4d} {v['exact_rate']:7.1%} {v['null_rate']:7.1%} {v['wrong_rate']:7.1%}")
    for k, v in report["source_confusion"].items():
        print(f"source[{k}]: {v}")
    print(f"report: {out_json}")
    return 1 if dangerous else 0


if __name__ == "__main__":
    raise SystemExit(main())
