"""Build demo/receipt-reader-demo.mp4 from REAL runs of the service on the synthetic set.

Starts the API with uvicorn, calls /v1/health and /v1/receipts/read over HTTP for each
synthetic receipt, renders the actual responses into slides and joins them with ffmpeg.

    python scripts/make_demo_video.py            # needs ffmpeg on PATH (or FFMPEG=...)

The synthetic set demonstrates behaviour only; it is not an accuracy measurement.
"""
from __future__ import annotations

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

from tests.synth import FONT_CANDIDATES, MONO_CANDIDATES, _font, fa  # noqa: E402

W, H = 1920, 1080
BG = (247, 247, 244)
INK = (28, 28, 30)
MUTED = (110, 110, 118)
ROUTE_COLORS = {"match": (22, 128, 61), "confirm_with_sender": (191, 122, 0), "manager_review": (190, 40, 40)}
FRAMES = ROOT / "demo" / "frames"
OUT = ROOT / "demo" / "receipt-reader-demo.mp4"
PORT = 8765
KEY = "demo-key"


def has_fa(s: str) -> bool:
    return any("؀" <= ch <= "ۿ" for ch in s)


def txt(d, xy, s, size=30, fill=INK, mono=False, anchor_right=False, max_w=None):
    font = _font(MONO_CANDIDATES if mono else FONT_CANDIDATES, size)
    x, y = xy
    max_w = max_w or (W - x - 40 if not anchor_right else x - 40)
    if not has_fa(s) and d.textlength(s, font=font) > max_w:
        # wrap on spaces
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
    return y + int(size * 1.45)


def slide_text(title, lines, sub=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 12), fill=(30, 64, 120))
    y = txt(d, (120, 140), title, 64)
    if sub:
        y = txt(d, (120, y + 10), sub, 34, MUTED)
    y += 40
    for ln in lines:
        if isinstance(ln, tuple):
            s, kw = ln
        else:
            s, kw = ln, {}
        y = txt(d, (140, y), s, kw.get("size", 34), kw.get("fill", INK), kw.get("mono", False))
    return img


def slide_receipt(name, img_path, resp, truth):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    rec = Image.open(img_path).convert("RGB")
    rec.thumbnail((860, 940))
    img.paste(rec, (40, 70))
    d.rectangle((40, 70, 40 + rec.width, 70 + rec.height), outline=(200, 200, 200), width=2)
    txt(d, (40, 20), f"input: {name}  ({resp['image']['kind']}, synthetic)", 26, MUTED)

    x = 960
    y = 40
    f = resp["fields"]
    route = resp["suggested_route"]
    d.rounded_rectangle((x, y, x + 900, y + 90), radius=16, fill=ROUTE_COLORS[route])
    txt(d, (x + 30, y + 18), f"suggested_route: {route}", 44, (255, 255, 255))
    y += 120
    a = f["amount_rial"]
    y = txt(d, (x, y), f"amount_rial = {a['value']:,}  [{a['confidence']}]" if a["value"] is not None
            else f"amount_rial = null  [{a['confidence']}]", 32)
    for c in a["candidates"][:5]:
        v = f"{c['value']:,}"
        y = txt(d, (x + 30, y), f"• {v:>16s}  {c['source']:<11s} {c['engine']}/{c.get('pass', 'full')}", 24, MUTED,
                mono=True)
    y = txt(d, (x, y + 6), f"currency = {f['currency']}   amount_toman = "
            f"{f['amount_toman']:,}" if f["amount_toman"] else f"currency = {f['currency']}   amount_toman = null", 28)
    ib = f["dest_iban"]
    y = txt(d, (x, y + 6), f"dest_iban = {ib['value'] or 'null'}", 28, mono=False)
    y = txt(d, (x + 30, y), f"checksum_valid={ib['checksum_valid']}  bank={ib['bank']}  [{ib['confidence']}]", 26, MUTED)
    p = f["paid_at"]
    y = txt(d, (x, y + 6), f"paid_at = {p['jalali']}  →  {p['iso']}  [{p['confidence']}]", 26)
    t = f["tracking_number"]
    y = txt(d, (x, y + 6), f"tracking = {t['value']}  [{t['confidence']}]", 28)
    y = txt(d, (x, y + 6), f"transfer_type = {f['transfer_type']}   document_status = {f['document_status']}", 28)
    y += 14
    for m in resp["matches"]:
        y = txt(d, (x, y), f"{m['line_id']}: score {m['score']}  fits={m['fits']}", 28)
        y = txt(d, (x + 30, y), ", ".join(m["reasons"]), 22, MUTED)
    y += 8
    if resp["warnings"]:
        y = txt(d, (x, y), "warnings:", 28)
        y = txt(d, (x + 30, y), ", ".join(resp["warnings"]), 22, (160, 60, 0))
    if resp.get("sender_question_fa"):
        y = txt(d, (x + 900, y + 10), resp["sender_question_fa"], 30, ROUTE_COLORS["confirm_with_sender"],
                anchor_right=True)
    y = txt(d, (x, max(y + 10, 960)), f"truth: amount {truth['amount_rial']:,}, line {truth.get('correct_line_id')}   "
            f"·  {resp['processing_ms']} ms  ·  engines: " + ", ".join(f"{k}={v}" for k, v in resp["engines"].items()), 20,
            MUTED)
    return img


def main():
    ffmpeg = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg not found")
    if FRAMES.exists():
        shutil.rmtree(FRAMES)
    FRAMES.mkdir(parents=True)

    env = dict(os.environ, RR_API_KEYS=KEY, RR_ENGINES=os.environ.get("RR_ENGINES", "vlm,tesseract,easyocr"),
               RR_REQUEST_TIMEOUT_S=os.environ.get("RR_REQUEST_TIMEOUT_S", "300"),
               PYTHONIOENCODING="utf-8", RR_LOG_LEVEL="WARNING")
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "receipt_reader.api:app_factory", "--factory",
                               "--port", str(PORT), "--log-level", "warning"], cwd=ROOT, env=env)
    try:
        base = f"http://127.0.0.1:{PORT}"
        for _ in range(60):
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

        slides = []
        slides.append((slide_text("Receipt Reader — settlement bot component", [
            "Reads a photo of an Iranian bank receipt (فیش) and suggests which open line it pays.",
            "It never approves anything. A receipt image is not proof that money arrived.",
            "",
            "1. read blind (expected values never reach OCR / VLM)",
            "2. collect candidates per field: printed / handwritten / digital / words",
            "3. cross-check: digits vs words, printed vs handwriting, IBAN mod-97, date",
            "4. match against the debtor's lines → match / confirm_with_sender / manager_review",
            "",
            ("Demo images are SYNTHETIC: this shows behaviour, not accuracy.", {"fill": (190, 40, 40)}),
        ]), 7))
        eng_lines = [f"{n}: {'loaded' if e['loaded'] else 'not loaded'} — {e['detail']}"
                     for n, e in health["engines"].items()]
        slides.append((slide_text("GET /v1/health", eng_lines + [
            "", f"cloud_vlm_allowed = {health['cloud_vlm_allowed']}  (images leave the server only when true)",
            "", "This run: " + ", ".join(n for n, e in health["engines"].items() if e["loaded"])
            + " (local only; images never left the machine)",
            "", "$ pytest -q", (test_summary, {"mono": True}),
        ], sub="real response from the running service"), 6))

        samples = ["satna_handwriting_wrong", "satna_clean", "satna_wrong_account", "paya_screenshot",
                   "failed_screenshot"]
        captions = {
            "satna_handwriting_wrong": "Handwritten amount is wrong (15,000,000,000); teller strip says 4,500,000,000",
            "satna_clean": "Clean SATNA form: strip, handwriting and words agree",
            "satna_wrong_account": "Valid receipt, but to an account that is not one of the debtor's lines",
            "paya_screenshot": "Mobile-banking PAYA screenshot",
            "failed_screenshot": "Screenshot of a FAILED transfer",
        }
        results = {}
        for name in samples:
            path = next((ROOT / "eval" / "synthetic").glob(name + ".*[gG]"))
            truth = json.loads((ROOT / "eval" / "synthetic" / f"{name}.json").read_text(encoding="utf-8"))
            with open(path, "rb") as fh:
                r = httpx.post(base + "/v1/receipts/read", headers={"X-API-Key": KEY},
                               files={"image": (path.name, fh, "image/jpeg")},
                               data={"expected_lines": json.dumps(truth["expected_lines"], ensure_ascii=False),
                                     "include_debug": "true"}, timeout=120)
            r.raise_for_status()
            resp = r.json()
            results[name] = resp
            slides.append((slide_text(captions[name], [
                f"POST /v1/receipts/read   image={path.name}",
                "expected_lines = " + ", ".join(f"{l['line_id']} ({l['amount_toman']:,} toman, …{l['iban'][-4:]})"
                                                for l in truth["expected_lines"]),
            ], sub="sending the photo and the debtor's open lines"), 3))
            slides.append((slide_receipt(name, path, resp, truth), 9))

        summary = []
        for name, resp in results.items():
            summary.append(f"{name:28s} → {resp['suggested_route']}")
        slides.append((slide_text("What happened", summary + [
            "",
            "• wrong handwriting is kept as a candidate but the printed strip wins (HANDWRITTEN_AMOUNT_DISAGREES)",
            "• two engines disagree (e.g. tracking number, a mangled strip amount) → conflict → a human looks",
            "• wrong account / failed status / unknown currency → manager_review, never match",
            "",
            ("Next: measure on the owner's real receipts (python eval/run_eval.py eval/real)", {"fill": (30, 64, 120)}),
        ]), 9))
    finally:
        server.terminate()
        server.wait(timeout=10)

    concat = FRAMES / "list.txt"
    with open(concat, "w", encoding="utf-8") as fh:
        for i, (img, secs) in enumerate(slides):
            p = FRAMES / f"s{i:02d}.png"
            img.save(p)
            fh.write(f"file '{p.name}'\nduration {secs}\n")
        fh.write(f"file 's{len(slides) - 1:02d}.png'\n")
    OUT.parent.mkdir(exist_ok=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-vf", "fps=25,format=yuv420p", "-c:v", "libx264", "-crf", "23", "-movflags", "+faststart",
                    str(OUT)], check=True)
    (ROOT / "demo" / "demo-responses.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {sum(s for _, s in slides)} s)")


if __name__ == "__main__":
    main()
