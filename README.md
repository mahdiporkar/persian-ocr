# Receipt Reader (فیش‌خوان) — settlement bot component

Reads a photo of an Iranian bank transfer receipt (فیش) and tells the settlement bot which of
the debtor's open lines it pays, and how sure it is. It is **one component** of the Telegram
settlement bot for the آبشده business: it reads, cross-checks and suggests. The bot and a human
manager make every decision.

> A receipt image is not proof that money arrived. This service never "verifies" or "confirms"
> a transaction; confirmation comes later from the person who receives the money.

```
image ──► validate + orient ──► crop/deskew ──► find printed strip ──┐
                                                                     │
          ┌──────────── read BLIND (no expected values) ─────────────┘
          ▼                                   ▼
   Engine A: vision-language model      Engine B: local OCR (Tesseract fas)
   full image + zoomed strip crop       full page + 2x zoomed strip crop
          └──────────────► candidates per field ◄──────────┘
                                  │  (value, raw_text, source, engine, pass, region)
                                  ▼
             cross-check: digits vs words, printed vs handwritten,
             IBAN mod-97, bank code vs bank name, date plausibility,
             independent reads agree → confidence high / medium / low / conflict
                                  ▼
             match against expected_lines (rial internally) → suggested_route
```

* Stateless: images are processed in memory and never stored. Returns `sha256`, a perceptual hash
  and the normalized tracking number so the bot's database can detect duplicates.
* Money is integers only (rial and toman). No floats anywhere.
* 100–150 receipts/day on one CPU server: every receipt is read by all configured engines.

---

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                    # set RR_API_KEYS at least
uvicorn receipt_reader.api:app_factory --factory --port 8080
# open http://localhost:8080/docs
```

Docker:

```bash
docker build -t receipt-reader .
docker run --env-file .env -p 8080:8080 receipt-reader
```

Tests (unit + API + end-to-end on synthetic images through real Tesseract):

```bash
pytest -q
```

### Installing the engines

**Tesseract (Engine B)** — needs Tesseract 5 and the Persian model.

* Debian/Ubuntu: `apt install tesseract-ocr`, then put
  [`fas.traineddata` (tessdata_best)](https://github.com/tesseract-ocr/tessdata_best/raw/main/fas.traineddata)
  in the tessdata folder (the Dockerfile does this).
* Windows: `winget install UB-Mannheim.TesseractOCR`, download `fas.traineddata` into a folder
  (e.g. `%USERPROFILE%\tessdata`, together with `eng.traineddata`) and set
  `RR_TESSDATA_PREFIX` to it.
* Check with `GET /v1/health` → `engines.tesseract.loaded`.

**EasyOCR (optional second local engine, Persian `fa` model)** — `pip install -r requirements-optional.txt`
(pulls PyTorch CPU, ~1 GB on disk; downloads its models on first use). Enable with
`RR_ENGINES=vlm,tesseract,easyocr`. It exists so the eval can compare it against Tesseract.

**VLM (Engine A)** — one adapter, two transports, chosen by `RR_VLM_PROVIDER`:

| Provider | Where images go | Setup |
|---|---|---|
| `anthropic` | Anthropic's cloud API | `RR_ALLOW_CLOUD_VLM=true`, `RR_VLM_API_KEY`, `RR_VLM_MODEL=claude-opus-5-5` |
| `openai_compatible` | your own server (vLLM, Ollama, LM Studio, llama.cpp) | `RR_VLM_BASE_URL=http://host:8000/v1`, `RR_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct` (or similar) |

Example self-hosted model with vLLM on a GPU box:
`vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 16384` then point `RR_VLM_BASE_URL` at it.
An `openai_compatible` endpoint on a private/loopback address counts as self-hosted; a public one
counts as cloud (override with `RR_VLM_SELF_HOSTED`).

### Cloud or not — decision needed from the owner

**`RR_ALLOW_CLOUD_VLM` is `false` by default: no receipt image leaves the server unless you
switch it on.** This is your decision; the trade-off:

| | Cloud VLM (e.g. Claude) | Self-hosted open VLM | Local OCR only |
|---|---|---|---|
| Handwriting, varied layouts | best expected (to be measured) | good for 7B+ models (to be measured) | weak on handwriting |
| Privacy | images (IBANs, names, amounts) are sent to a third party | stay on your hardware | stay on the server |
| Reachability | the provider must be reachable from the server's location and accept payment from it; sanctions/filtering can break it without notice | no external dependency | none |
| Cost | per-image API cost (small at 150/day) | a GPU server (≈16–24 GB VRAM for a 7B VL model) or slow CPU inference | free |

Without any VLM the service still runs on Tesseract alone, but it will route far fewer receipts
to `match` (handwriting is mostly `low`), so more work goes to humans.

---

## API

`POST /v1/receipts/read` — multipart form. Header `X-API-Key` required. Optional `X-Request-ID`
(echoed back; generated otherwise).

| field | |
|---|---|
| `image` | JPG / PNG / WEBP (HEIC when `pillow-heif` is installed). Type is checked from the bytes, not the filename. |
| `expected_lines` | JSON array: `[{"line_id","iban","holder_name","amount_toman","remaining_toman"}]` |
| `engines` | optional, e.g. `vlm,tesseract` (default `RR_ENGINES`) |
| `include_debug` | `true` to get OCR text, strip box, skew etc. |

`GET /v1/health` — status and which engines are loaded (with the reason when not).

```bash
curl -s http://localhost:8080/v1/receipts/read \
  -H "X-API-Key: $RR_KEY" \
  -F image=@receipt.jpg \
  -F 'expected_lines=[{"line_id":"L-17","iban":"IR440120000000001234568550","holder_name":"علی کریمی","amount_toman":450000000,"remaining_toman":450000000}]'

curl -s http://localhost:8080/v1/health
```

Response (abridged) — the real output for `eval/synthetic/satna_clean.jpg`, Tesseract only:

```json
{
  "receipt_id": "…", "request_id": "…",
  "image": {"sha256": "…", "phash": "c16a3ed03ac46eb5", "kind": "unknown", "flags": []},
  "fields": {
    "amount_rial": {"value": 4500000000, "confidence": "high", "candidates": [
      {"value": 4500000000, "source": "handwritten", "engine": "tesseract", "raw_text": "4,500,000,000"},
      {"value": 4500000000, "source": "printed", "engine": "tesseract", "raw_text": "4,500,000,000"},
      {"value": 4500000000, "source": "printed", "engine": "tesseract", "raw_text": "4,500,000,000", "pass": "strip"}]},
    "currency": "rial", "amount_toman": 450000000,
    "dest_iban": {"value": "IR440120000000001234568550", "checksum_valid": true, "bank": "Mellat", "confidence": "high"},
    "paid_at": {"jalali": "1405/08/24 11:42", "iso": "2026-11-15T11:42:00+03:30", "plausible": true, "confidence": "high"},
    "tracking_number": {"value": "1140508249876543", "label": "شماره پیگیری", "confidence": "high"},
    "transfer_type": "satna", "document_status": "success"
  },
  "matches": [
    {"line_id": "L-17", "score": 1.0, "fits": true, "reasons": ["AMOUNT_EXACT", "IBAN_FULL_MATCH", "NAME_MATCH"]},
    {"line_id": "L-18", "score": 0.05, "fits": false, "reasons": ["AMOUNT_OVER_REMAINING", "IBAN_MISMATCH", "NAME_MISMATCH"]}
  ],
  "best_line_id": "L-17",
  "suggested_route": "match",
  "sender_question_fa": null,
  "warnings": [],
  "engines": {"vlm": "skipped: no VLM provider configured (RR_VLM_PROVIDER)", "tesseract": "ok"},
  "processing_ms": 1790
}
```

Note the tracking number: the printed one is `140508249876543`; both Tesseract passes added the
same stray leading `1` (see *Known limitations*). Amount and IBAN were right.

Errors: `{"error": CODE, "detail": "...", "request_id": "..."}` with codes `UNAUTHORIZED` (401),
`BAD_EXPECTED_LINES` (400), `EMPTY_FILE` (400), `FILE_TOO_LARGE` / `IMAGE_TOO_LARGE` (413),
`UNSUPPORTED_TYPE` / `HEIC_NOT_SUPPORTED` (415), `CORRUPT_IMAGE` / `IMAGE_TOO_SMALL` (422),
`TIMEOUT` (504), `NO_API_KEY_CONFIGURED` (503). An engine failing never fails the request; it is
reported in `engines`.

### How the bot should call it

1. Debtor posts a photo → bot downloads the largest size Telegram offers (not the thumbnail).
2. Bot sends it with **all of that debtor's open lines** (`remaining_toman` = what is still owed).
3. Bot checks `image.sha256` / `image.phash` / `tracking_number_normalized` against its own DB for
   duplicates (this service does not remember anything).
4. Bot applies its own policy on top of `suggested_route` (e.g. in trial mode, treat `match` as
   `confirm_with_sender`):
   * `match` → mark the line as *receipt received* (not *paid*), wait for the receiver's confirmation.
   * `confirm_with_sender` → post `sender_question_fa` to the debtor, e.g.
     «این فیش را ۴۵۰٬۰۰۰٬۰۰۰ تومان به حساب …۸۵۵۰ خواندم. درست است؟»
   * `manager_review` → forward the photo, `warnings` and `matches` to the manager.
5. Log `request_id` with the bot's own record.

---

## Rules

### Fields and currency

Required: `amount_rial`, `dest_iban`, `paid_at`, `tracking_number`, `transfer_type`,
`document_status`. Also: `dest_holder_name`, `dest_bank` (from the IBAN bank code via
[`banks.json`](receipt_reader/data/banks.json)), `source_name`, `source_account`, `amount_words`.

* The unit is taken from the document (ریال / تومان next to the amount or in the words). If none is
  visible, `currency: "unknown"` (+ `UNKNOWN_CURRENCY`) — never guessed. The matcher then tests the
  number both as rial and as toman and reports which reading fit (`CURRENCY_ASSUMED_*`).
* `amount_toman = amount_rial // 10` only when it divides exactly, else `null`.
* Balances (مانده), fees (کارمزد), card/account/national-ID numbers are never taken as amounts or
  tracking numbers. A source IBAN (از حساب / مبدا) is never used as the destination.
* Unreadable → `null` + a warning. The VLM is told to write `?` for unreadable characters and such
  readings are dropped, not repaired.

### Candidate sources

Every reading is kept as a candidate with `source`:

* `printed` — teller strip/stamp, printed slip. Tesseract/EasyOCR: on a paper form where a strip was
  detected, text **inside** the strip is `printed` and text **outside** it is `handwritten`; with no
  strip, lines with OCR confidence ≥ `RR_TESSERACT_PRINTED_MIN_CONF` are `printed`, others
  `handwritten`. The VLM labels the source itself.
* `handwritten`, `digital` (anything on a screenshot), `words` (amount parsed from مبلغ به حروف).

Two candidates are **independent** if they come from different engines, or from the same engine on a
different crop (full page vs zoomed strip). The second case can be switched off with
`RR_SAME_ENGINE_CROPS_INDEPENDENT=false` (see limitations).

### Confidence rules (exact)

`trusted` = `printed` or `digital`.

**amount_rial**
1. Two different trusted values → `conflict` (+ `MULTIPLE_AMOUNTS`, `AMOUNT_CONFLICT`), value `null`.
2. One trusted value V:
   * a words reading exists and none equals V → `conflict`, value `null`;
   * else `high` if ≥ 2 independent trusted reads of V **or** the words equal V; otherwise `medium`
     (a single read of an amount has no other validator, so it is never `high` alone);
   * a handwritten reading ≠ V → V still wins, + `AMOUNT_CONFLICT` and `HANDWRITTEN_AMOUNT_DISAGREES`.
3. No trusted value: two different handwritten values → `conflict`; one handwritten value H:
   words ≠ H → `conflict`; words = H → `medium` (handwriting-only is capped at medium); no words → `low`.
4. Only words → `low`.
5. Outside `[RR_MIN_AMOUNT_RIAL, RR_MAX_AMOUNT_RIAL]` → at most `low`.

**dest_iban** — pool = readings with role `dest`, else role `unknown` that are not also a source IBAN.
1. Two different checksum-valid IBANs → `conflict` (`IBAN_CONFLICT`). A partial reading whose digits
   are not a suffix of the full IBAN → `conflict`.
2. One valid IBAN: `high` if read from a trusted source and (≥ 2 independent reads, or its bank code
   is known); `medium` if trusted otherwise; handwriting only → `medium` with ≥ 2 independent reads,
   else `low`. Bank named next to the destination ≠ bank of the code → `BANK_NAME_MISMATCH`, cap
   `medium`. Role not labelled on the document → cap `medium`.
3. Only checksum-failing IBANs → `low` + `INVALID_IBAN`; never used for an exact match.
4. Only partial (last digits) → `PARTIAL_IBAN`, `medium` if trusted else `low`; never `match`.

**paid_at** — trusted readings preferred. Different days, or different times on the same day →
`conflict` (`DATE_CONFLICT`). Trusted: `high` with ≥ 2 independent reads or when it has a time,
else `medium`; handwriting only: `medium` with ≥ 2 reads, else `low`. In the future (15 min
tolerance) or older than `RR_DATE_MAX_AGE_DAYS` (3) → `DATE_IMPLAUSIBLE`, `low`.

**tracking_number** — all numbers are returned in `all` with their own label. The primary one is
chosen by label priority (پیگیری › مرجع › رهگیری › ارجاع › تراکنش › سند). Two different trusted
values for that label → `conflict` (`TRACKING_CONFLICT`). Trusted + 6–30 digits → `high`; handwriting
only: `medium` with ≥ 2 reads, else `low`.

**transfer_type / document_status** — keyword votes (ساتنا، پایا، کارت به کارت … / موفق، ناموفق،
در انتظار) plus the VLM's reading (double weight). Any "failed" → `failed`; then `pending`; then
`success`. A paper form whose detected teller strip carries a printed tracking number counts as
`success` (`RR_PRINTED_STRIP_IMPLIES_SUCCESS`, an assumption the owner should confirm).

### Matching and route

Read blind first: expected values are never in any OCR/VLM input. Then, per line, in rial:

| reason | weight |
|---|---|
| `AMOUNT_EXACT` (= remaining) | 0.50 |
| `AMOUNT_PARTIAL` (< remaining) | 0.30 |
| `AMOUNT_OVER_REMAINING` | 0.05 |
| `IBAN_FULL_MATCH` (checksum valid) | 0.40 |
| `IBAN_LAST_DIGITS_MATCH` (partial, ≥ 4 digits, or invalid checksum with same last 6) | 0.25 |
| `IBAN_MISMATCH` | 0 |
| `NAME_MATCH` / `NAME_MISMATCH` (Persian-normalized fuzzy ≥ 85) | 0.10 / 0 |

The score only ranks lines. A line **fits** when the amount is exact or partial **and** the IBAN is
not a mismatch. Two fitting lines with equal scores → `AMBIGUOUS_MATCH`, no line picked.

* `match` — amount, IBAN, date and tracking number all `high`; IBAN checksum valid; exactly one
  fitting line, with `IBAN_FULL_MATCH`; status `success`; transfer type known.
* `manager_review` — any `conflict`; no amount; no fitting line (`NO_MATCHING_LINE`, `IBAN_MISMATCH`);
  `AMBIGUOUS_MATCH`; `MULTIPLE_AMOUNTS`; `UNKNOWN_CURRENCY`; status not `success`;
  `POSSIBLE_EDITING`; no expected lines sent; no engine available.
* `confirm_with_sender` — everything else (one line fits but something is `medium`/`low`, partial
  IBAN, two lines fit with different scores …), with the Persian question.

### Warning codes

`LOW_IMAGE_QUALITY`, `AMOUNT_CONFLICT`, `HANDWRITTEN_AMOUNT_DISAGREES`, `MULTIPLE_AMOUNTS`,
`UNKNOWN_CURRENCY`, `INVALID_IBAN`, `PARTIAL_IBAN`, `IBAN_MISMATCH`, `IBAN_CONFLICT`,
`BANK_NAME_MISMATCH`, `DATE_IMPLAUSIBLE`, `DATE_CONFLICT`, `TRACKING_CONFLICT`,
`STATUS_NOT_SUCCESS`, `NO_MATCHING_LINE`, `AMBIGUOUS_MATCH`, `POSSIBLE_EDITING`,
`NO_EXPECTED_LINES`, `NO_ENGINE_AVAILABLE`.

Image flags (signals only, never a rejection): `EDITING_SOFTWARE_IN_EXIF`, `VLM_SUSPECTED_EDITING`
(both raise `POSSIBLE_EDITING`), `SCREENSHOT_OF_SCREENSHOT`, `PHOTO_OF_SCREEN`, `BLURRY`,
`BAD_EXPOSURE`.

---

## Image handling

EXIF orientation fix → reject empty / non-image (magic bytes) / too small / too large / decompression
bombs → screenshot vs photo detection → (photos) find the sheet and warp it flat, deskew ±15° →
upscale small images → CLAHE + denoise for local OCR. The VLM gets the orientation-fixed original
(downscaled only above 2000 px).

**Printed strip:** a rectangular frame in the lower half containing ≥ 2 text lines (teller strips and
stamps are usually boxed); otherwise the longest run of lines in the lower half with machine-print
geometry (uniform glyph height, straight baseline). The strip is re-read as a 2× zoomed crop by
Tesseract, and the VLM re-reads its own strip box (or ours) as a second pass.

## Normalization

Persian/Arabic digits → ASCII; ي→ی, ك→ک, ة→ه …; ZWNJ, bidi marks, tatweel and diacritics removed;
thousands separators `,` `٬` `.` space; IBAN with spaces/dashes/no `IR`/masked; Jalali ↔ Gregorian
(original kept, ISO with `+03:30`); OCR bidi repair for split numbers (`500,000,000 ,4` →
`4,500,000,000`); common OCR misreads of labels (بیگیری → پیگیری). Persian number words parser in
[`numwords.py`](receipt_reader/numwords.py) — strict: any unknown word → no value.

## Security

* IBANs, card numbers and national IDs are masked in logs (last 4 digits kept); logs are JSON.
* Images are never written to disk; nothing is stored between requests.
* File size limit (`RR_MAX_UPLOAD_MB`), type checked from content, pixel-count limit, request timeout.
* API keys compared in constant time. Config only from environment / `.env` (see `.env.example`).

## Evaluation

See [`eval/README.md`](eval/README.md). `python eval/run_eval.py eval/real` reports the
dangerous-error rate (target 0), the auto-match share, and per-field exact/null/wrong rates and
source confusion. Synthetic images (`eval/synthetic/`, `tests/synth.py`) are for tests and the demo
only.

## Demo

[`demo/receipt-reader-demo.mp4`](demo/receipt-reader-demo.mp4) (82 s) — generated by
`python scripts/make_demo_video.py`: it starts the API with uvicorn, calls `/v1/health` and
`/v1/receipts/read` over HTTP for each synthetic receipt, and renders the real responses
(Tesseract + EasyOCR, no VLM). The raw responses are in
[`demo/demo-responses.json`](demo/demo-responses.json). It shows behaviour, not accuracy.

## Known limitations

* **No accuracy has been measured yet.** All numbers so far come from synthetic images. Engine
  choice, the VLM prompt and the thresholds must be tuned on the owner's real receipts.
* Tesseract cannot tell handwriting from print; the inside/outside-strip rule is a heuristic. If no
  strip is detected on a paper form, high-confidence handwriting may be tagged `printed`.
* Two Tesseract passes over the same pixels can repeat the same systematic misreading (seen on the
  synthetic form: a stray leading `1` in the tracking number in both passes, reported `high`). With
  a second engine (EasyOCR, on the same synthetic set) it became a `TRACKING_CONFLICT` instead. If
  only one engine runs in production, consider `RR_SAME_ENGINE_CROPS_INDEPENDENT=false`.
* EasyOCR on CPU takes ~10–25 s per receipt and sometimes mangles numbers on the zoomed strip
  (`4,` → `ا4`); those become conflicts (→ `manager_review`), not wrong values.
* Strip detection is heuristic (box frame or line regularity); unboxed, skewed or faint stamps may be
  missed.
* Status of paper forms relies on `RR_PRINTED_STRIP_IMPLIES_SUCCESS`.
* Card-to-card and ATM slips are handled by the generic parser only.
* The editing/tampering flags are weak signals.
* Region boxes are in the coordinates of the image each engine saw (VLM: oriented original;
  Tesseract: cropped/deskewed/upscaled page).

## Layout

```
receipt_reader/
  api.py            FastAPI app (auth, request IDs, errors, timeouts)
  pipeline.py       image → engines (parallel) → cross-check → match
  imaging.py        validation, orientation, crop/deskew, strip detection, hashes, flags
  engines/          base interface, vlm.py (Anthropic + OpenAI-compatible), tesseract.py,
                    easyocr_engine.py (optional), scripted.py (tests)
  textparse.py      OCR lines → labelled candidates (local engines)
  crosscheck.py     confidence levels and conflicts
  matcher.py        line scoring, route, Persian sender question
  normalize.py numwords.py iban.py dates.py   pure helpers
  data/banks.json   IBAN bank codes
eval/               labelled-set format, run_eval.py, synthetic set
tests/              unit, API, VLM adapter (no network), e2e with Tesseract
scripts/            synthetic set + demo video generators
```
