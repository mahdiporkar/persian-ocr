"""Build a video from REAL runs of the service on a labelled folder.

Starts the API with uvicorn, calls /v1/health and /v1/receipts/read over HTTP for each
receipt, renders the actual responses next to the ground truth and joins the slides with
ffmpeg.

    python scripts/make_demo_video.py                                  # synthetic set -> demo/
    python scripts/make_demo_video.py --dataset eval/real \\
        --out eval/real/video/real-receipts-test.mp4 --private           # owner's receipts (never commit)

--private adds a "contains real customer data" banner to every slide. Accuracy is only
meaningful on real receipts; the synthetic set shows behaviour only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.run_eval import FIELDS, PARTY_FIELDS, labelled, predicted, same, truth_of  # noqa: E402
from receipt_reader.schema import ReadResponse  # noqa: E402
from tests.synth import FONT_CANDIDATES, MONO_CANDIDATES, _font, fa  # noqa: E402

W, H = 1920, 1080
BG = (247, 247, 244)
INK = (28, 28, 30)
MUTED = (110, 110, 118)
OK = (22, 128, 61)
BAD = (190, 40, 40)
WARN = (191, 122, 0)
ROUTE_COLORS = {"match": OK, "confirm_with_sender": WARN, "manager_review": BAD}
PORT = 8765
KEY = "demo-key"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
SHOW = ["amount_rial", "currency", "dest_iban", "paid_at", "tracking_number", "document_status",
        "transfer_type", "sender_card", "sender_bank", "sender_account", "receiver_name", "receiver_card",
        "receiver_bank", "receiver_account"]
CONF = ("amount_rial", "dest_iban", "paid_at", "tracking_number")


def has_fa(s: str) -> bool:
    return any("؀" <= ch <= "ۿ" for ch in s)


def txt(d, xy, s, size=30, fill=INK, mono=False, anchor_right=False, max_w=None):
    font = _font(MONO_CANDIDATES if mono else FONT_CANDIDATES, size)
    x, y = xy
    max_w = max_w or (W - x - 40 if not anchor_right else x - 40)
    if not has_fa(s) and d.textlength(s, font=font) > max_w:
        words, line = s.split(" "), ""
        for w in words:
            cand = (line + " " + w).strip()
            if d.textlength(cand, font=font) > max_w and line:
                d.text((x, y), line, font=font, fill=fill)
                y += int(size * 1.35)
                line = w
            else:
                line = cand
        s = line
    t = fa(s) if has_fa(s) else s
    if anchor_right:
        x -= d.textlength(t, font=font)
    d.text((x, y), t, font=font, fill=fill)
    return y + int(size * 1.4)


def cell(d, xy, value, size=24, fill=INK):
    """Draw a value that may be Persian (shaped right-to-left) or latin/digits."""
    font = _font(FONT_CANDIDATES, size)
    parts = value.split("  ")
    x, y = xy
    for i, part in enumerate(parts):
        t = fa(part) if has_fa(part) else part
        d.text((x, y), t, font=font, fill=fill if i == 0 else MUTED)
        x += d.textlength(t + "  ", font=font)


def banner(img, private):
    if private:
        d = ImageDraw.Draw(img)
        d.rectangle((0, H - 44, W, H), fill=(120, 0, 0))
        txt(d, (30, H - 40), "PRIVATE — contains real customer receipt data. Do not publish or commit.", 26,
            (255, 255, 255))
    return img


def slide_text(title, lines, sub=None, private=False):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 12), fill=(30, 64, 120))
    y = txt(d, (120, 100), title, 58)
    if sub:
        y = txt(d, (120, y + 6), sub, 30, MUTED)
    y += 26
    for ln in lines:
        s, kw = ln if isinstance(ln, tuple) else (ln, {})
        y = txt(d, (140, y), s, kw.get("size", 30), kw.get("fill", INK), kw.get("mono", False))
    return banner(img, private)


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def slide_receipt(name, img_path, resp, gt, private):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    rec = Image.open(img_path).convert("RGB")
    rec.thumbnail((620, 960))
    img.paste(rec, (30, 70))
    d.rectangle((30, 70, 30 + rec.width, 70 + rec.height), outline=(200, 200, 200), width=2)
    txt(d, (30, 20), f"{name}  ({resp['image']['kind']})", 26, MUTED)

    x = 690
    f = resp["fields"]
    route = resp["suggested_route"]
    d.rounded_rectangle((x, 24, x + 1190, 96), radius=14, fill=ROUTE_COLORS[route])
    txt(d, (x + 24, 34), f"suggested_route: {route}", 40, (255, 255, 255))

    res = ReadResponse(**resp)
    pred = predicted(res)
    truth = truth_of(gt) if gt else None

    hdr = _font(FONT_CANDIDATES, 22)
    y = 112
    d.text((x, y), "field", font=hdr, fill=MUTED)
    d.text((x + 245, y), "read by the service  (confidence)", font=hdr, fill=MUTED)
    d.text((x + 790, y), "ground truth", font=hdr, fill=MUTED)
    y += 32
    for k in SHOW:
        p = pred[k]
        if k == "amount_rial" and p is None and f["amount_rial"]["suggested_value"]:
            pshow = f"conflict, suggested {f['amount_rial']['suggested_value']:,}"
        else:
            pshow = _fmt(p)
        if k in CONF:
            pshow += f"  ({f[k]['confidence']})"
        elif "_" in k and k.split("_", 1)[0] in ("sender", "receiver"):
            side, sub = k.split("_", 1)
            v = f[side][sub]
            if v["value"] or v["confidence"] == "conflict":
                pshow += f"  ({v['confidence']})"
        if k == "tracking_number" and f["tracking_number"]["label"]:
            pshow += f"  {f['tracking_number']['label']}"
        if truth is None or not labelled(gt, k):
            mark, mc, tshow = "", MUTED, "(not labelled)"
        else:
            tshow = _fmt(truth[k])
            if same(k, p, truth[k]):
                mark, mc = "OK", OK
            elif p is None:
                mark, mc = "missed", WARN
            else:
                mark, mc = "WRONG", BAD
        d.text((x, y), k, font=_font(FONT_CANDIDATES, 23), fill=INK)
        cell(d, (x + 245, y), pshow[:52], 23)
        cell(d, (x + 790, y), tshow[:26], 23, MUTED)
        d.text((x + 1110, y), mark, font=_font(FONT_CANDIDATES, 23), fill=mc)
        y += 36
    y += 6
    cands = f["amount_rial"]["candidates"]
    if cands:
        y = txt(d, (x, y), "amount candidates (all kept, with evidence):", 21, MUTED)
        for c in cands[:3]:
            y = txt(d, (x + 20, y), f"• {c['value']:,}  {c['source']}  {c['engine']}/{c.get('pass', 'full')}  "
                                    f"raw='{c['raw_text'][:24]}'", 20, MUTED)
    if resp["warnings"]:
        y = txt(d, (x, y + 4), "warnings: " + ", ".join(resp["warnings"]), 21, (160, 60, 0), max_w=1180)
    y = txt(d, (x, max(y + 4, 930)), f"{resp['processing_ms']} ms · engines: " +
            ", ".join(f"{k}={v.split(':')[0]}" for k, v in resp["engines"].items()) +
            " · settlement confirmed by this service: no", 20, MUTED, max_w=1180)
    if gt and gt.get("notes"):
        txt(d, (x, y), "note: " + gt["notes"][:170], 19, MUTED, max_w=1180)
    return banner(img, private)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "eval" / "synthetic")
    ap.add_argument("--out", type=Path, default=ROOT / "demo" / "receipt-reader-demo.mp4")
    ap.add_argument("--engines", default=os.environ.get("RR_ENGINES", "vlm,tesseract,easyocr"))
    ap.add_argument("--private", action="store_true", help="real customer data: banner on every slide")
    ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)

    ffmpeg = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg not found")
    frames = a.out.parent / (a.out.stem + "-frames")
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)

    env = dict(os.environ, RR_API_KEYS=KEY, RR_ENGINES=a.engines, PYTHONIOENCODING="utf-8",
               RR_LOG_LEVEL="WARNING", RR_REQUEST_TIMEOUT_S=os.environ.get("RR_REQUEST_TIMEOUT_S", "300"))
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "receipt_reader.api:app_factory", "--factory",
                               "--port", str(PORT), "--log-level", "warning"], cwd=ROOT, env=env)
    slides, results, summary_rows = [], {}, []
    stats = {"n": 0, "dangerous": 0, "amount_ok": 0, "amount_wrong": 0}
    try:
        base = f"http://127.0.0.1:{PORT}"
        for _ in range(120):
            try:
                health = httpx.get(base + "/v1/health", timeout=2).json()
                break
            except Exception:
                time.sleep(0.5)
        else:
            sys.exit("server did not start")

        tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, env=env,
                               capture_output=True, text=True, encoding="utf-8")
        test_summary = [ln for ln in tests.stdout.splitlines() if ln.strip()][-1]

        samples = sorted(p for p in a.dataset.iterdir() if p.suffix.lower() in IMAGE_EXT)
        title = a.title or ("Receipt Reader — test on real receipts" if a.private else
                            "Receipt Reader — demo on synthetic receipts")
        slides.append((slide_text(title, [
            "Reads a receipt photo blind, cross-checks every reading, then suggests a route.",
            "It never approves a payment and never confirms settlement.",
            "",
            f"Dataset: {a.dataset.name}  ({len(samples)} images)",
            f"Engines requested: {a.engines}   (cloud VLM allowed: {health['cloud_vlm_allowed']})",
            "Each slide: the live API response next to the ground truth (OK / missed / WRONG).",
            "",
            ("Real receipts, small set: numbers describe THIS set only, not general accuracy." if a.private else
             "Synthetic images: behaviour only, not accuracy."), ], private=a.private), 7))
        eng_lines = [f"{n}: {'loaded' if e['loaded'] else 'not loaded'} — {e['detail']}"
                     for n, e in health["engines"].items()]
        slides.append((slide_text("GET /v1/health", eng_lines + [
            "", f"cloud_vlm_allowed = {health['cloud_vlm_allowed']}  (images leave the server only when true)",
            "", "$ pytest -q", (test_summary, {"mono": True}),
        ], sub="real response from the running service", private=a.private), 6))

        for path in samples:
            gt_path = path.with_suffix(".json")
            gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else None
            data = {"include_debug": "false"}
            if gt and gt.get("expected_lines"):
                data["expected_lines"] = json.dumps(gt["expected_lines"], ensure_ascii=False)
            with open(path, "rb") as fh:
                r = httpx.post(base + "/v1/receipts/read", headers={"X-API-Key": KEY},
                               files={"image": (path.name, fh, "image/jpeg")}, data=data, timeout=400)
            r.raise_for_status()
            resp = r.json()
            results[path.name] = resp
            print(f"{path.name}: {resp['suggested_route']} ({resp['processing_ms']} ms)")
            slides.append((slide_receipt(path.name, path, resp, gt, a.private), 8))
            res = ReadResponse(**resp)
            p = predicted(res)
            t = truth_of(gt) if gt else None
            stats["n"] += 1
            if t:
                if p["amount_rial"] is not None:
                    stats["amount_ok" if p["amount_rial"] == t["amount_rial"] else "amount_wrong"] += 1
                if res.suggested_route == "match" and (p["amount_rial"] != t["amount_rial"]
                                                       or p["dest_iban"] != t["dest_iban"]
                                                       or res.best_line_id != gt.get("correct_line_id")):
                    stats["dangerous"] += 1
            summary_rows.append(f"{path.name:9s} {resp['suggested_route']:16s} amount read "
                                f"{_fmt(p['amount_rial']):>13s}  truth {_fmt(t['amount_rial']) if t else '?':>13s}")

        routes = {}
        for resp in results.values():
            routes[resp["suggested_route"]] = routes.get(resp["suggested_route"], 0) + 1
        slides.append((slide_text("Summary — this set only", [(s, {"mono": True, "size": 23}) for s in summary_rows] + [
            "",
            f"routes: {routes}",
            f"dangerous errors (match with wrong amount / IBAN / line): {stats['dangerous']}",
            f"amounts read: {stats['amount_ok']} correct, {stats['amount_wrong']} wrong, "
            f"{stats['n'] - stats['amount_ok'] - stats['amount_wrong']} not read (null)",
        ], private=a.private), 14))
    finally:
        server.terminate()
        server.wait(timeout=10)

    concat = frames / "list.txt"
    with open(concat, "w", encoding="utf-8") as fh:
        for i, (img, secs) in enumerate(slides):
            pth = frames / f"s{i:02d}.png"
            img.save(pth)
            fh.write(f"file '{pth.name}'\nduration {secs}\n")
        fh.write(f"file 's{len(slides) - 1:02d}.png'\n")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-vf", "fps=25,format=yuv420p", "-c:v", "libx264", "-crf", "23", "-movflags", "+faststart",
                    str(a.out)], check=True)
    a.out.with_suffix(".responses.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    print(f"wrote {a.out} ({a.out.stat().st_size // 1024} KB, {sum(s for _, s in slides)} s)")


if __name__ == "__main__":
    main()
