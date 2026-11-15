# Receipt Reader (فیش‌خوان) — settlement bot component

[![tests](https://github.com/mahdiporkar/persian-ocr/actions/workflows/tests.yml/badge.svg)](https://github.com/mahdiporkar/persian-ocr/actions/workflows/tests.yml)

Reads a photo of an Iranian bank transfer receipt (فیش) and tells the settlement bot which of
the debtor's open lines it pays, and how sure it is. It is **one component** of the Telegram
settlement bot for the آبشده business: it reads, cross-checks and suggests. The bot and a human
manager make every decision.

> A receipt image is not proof that money arrived. This service never "verifies" or "confirms"
> a transaction; confirmation comes later from the person who receives the money. Every response
> carries `settlement.confirmed_by_this_service: false`; `document_status` only reports what the
> document itself says.

### Owner decisions (2026-11-16) and where they are enforced

| Decision | Enforced in |
|---|---|
| Cloud VLM optional, **off by default**; no image leaves the server without `RR_ALLOW_CLOUD_VLM=true` + a provider config | `engines/vlm.py` `available()`; `anthropic` SDK only in `requirements-cloud.txt`; test `test_local_mode_never_imports_a_cloud_sdk` |
| No real-world accuracy claims beyond what is measured | README "Measured on real receipts" states set size and caveats |
| A printed tracking number / teller strip does **not** mean success; status ≠ settlement | `crosscheck.document_status` (explicit words only), `settlement` block in every response |
| Two passes of one engine are **one** source of evidence | `crosscheck.engines_of()` counts distinct engines only |
| Extraction independent of matching | `ReceiptReader.extract()` takes no expected values; `POST /v1/receipts/extract`; test that fields are identical with/without lines |
| Sender and receiver info when visible | `fields.sender` / `fields.receiver` (name, account, card last 4, IBAN, bank) |
| Keep all conflicting amount candidates and their evidence | `amount_rial.candidates` (value, raw text, source, engine, pass, label, region) |
| Any amount conflict → `manager_review`, even if printed looks more reliable | `amount_rial.value = null`, `confidence = conflict`; printed only as `suggested_value`; `AMOUNT_CONFLICT` forces `manager_review` |
| A masked IBAN is never checksum-validated | `partial: true`, `checksum_valid: null`, never `IBAN_FULL_MATCH`; masked card numbers are never taken as IBANs |
| Swagger documents all parameters and schemas | descriptions on every form field, header and model field; test `test_openapi_documents_every_parameter_and_schema` |
| Local OCR without any cloud dependency | `requirements.txt` has no cloud SDK; Tesseract/EasyOCR run offline |

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
             different ENGINES agree → confidence high / medium / low / conflict
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

Docker (verified: builds, runs as non-root, serves `/v1/health` and `/v1/receipts/read`; image ≈ 1 GB
with Tesseract and the Persian model; add `easyocr` yourself if wanted — it pulls PyTorch):

```bash
docker build -t receipt-reader .
docker run --env-file .env -p 8080:8080 receipt-reader
```

Tests (unit + API + image handling + end-to-end on synthetic images through real Tesseract). They
run on every push via GitHub Actions (Ubuntu + Tesseract `fas`), followed by the eval harness on the
synthetic set, which fails the build on any dangerous error:

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

**VLM (Engine A)** — one adapter, two transports, chosen by `RR_VLM_PROVIDER`. The cloud SDK is
not in the core requirements: `pip install -r requirements-cloud.txt` only if you enable it.

| Provider | Where images go | Setup |
|---|---|---|
| `anthropic` | Anthropic's cloud API | `RR_ALLOW_CLOUD_VLM=true`, `RR_VLM_API_KEY`, `RR_VLM_MODEL=claude-opus-5-5` |
| `openai_compatible` | your own server (vLLM, Ollama, LM Studio, llama.cpp) | `RR_VLM_BASE_URL=http://host:8000/v1`, `RR_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct` (or similar) |

Example self-hosted model with vLLM on a GPU box:
`vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 16384` then point `RR_VLM_BASE_URL` at it.
An `openai_compatible` endpoint on a private/loopback address counts as self-hosted; a public one
counts as cloud (override with `RR_VLM_SELF_HOSTED`).

### Cloud or not

Owner decision: a cloud VLM is allowed as an **optional** provider and stays **disabled by default**
(`RR_ALLOW_CLOUD_VLM=false`). No receipt image is sent anywhere unless it is explicitly configured.
The trade-off:

| | Cloud VLM (e.g. Claude) | Self-hosted open VLM | Local OCR only |
|---|---|---|---|
| Handwriting, varied layouts | best expected (to be measured) | good for 7B+ models (to be measured) | weak on handwriting |
| Privacy | images (IBANs, names, amounts) are sent to a third party | stay on your hardware | stay on the server |
| Reachability | the provider must be reachable from the server's location and accept payment from it; sanctions/filtering can break it without notice | no external dependency | none |
| Cost | per-image API cost (small at 150/day) | a GPU server (≈16–24 GB VRAM for a 7B VL model) or slow CPU inference | free |

Without any VLM the service still runs on local OCR alone. On the owner's real receipts this
currently means **every receipt goes to `manager_review`** (see "Measured on real receipts").

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

`POST /v1/receipts/extract` — the blind read only (same `image`, `engines`, `include_debug`); no
expected lines are accepted and no route is produced.

`GET /v1/health` — status and which engines are loaded (with the reason when not).

Swagger UI at `/docs`, OpenAPI at `/openapi.json` — every parameter and schema field is described.

```bash
curl -s http://localhost:8080/v1/receipts/read \
  -H "X-API-Key: $RR_KEY" \
  -F image=@receipt.jpg \
  -F 'expected_lines=[{"line_id":"L-17","iban":"IR440120000000001234568550","holder_name":"علی کریمی","amount_toman":450000000,"remaining_toman":450000000}]'

curl -s http://localhost:8080/v1/health
```

Response shape (abridged, illustrative values):

```json
{
  "receipt_id": "…", "request_id": "…",
  "image": {"sha256": "…", "phash": "c16a3ed03ac46eb5", "kind": "photo", "flags": []},
  "fields": {
    "amount_rial": {
      "value": null, "confidence": "conflict",
      "suggested_value": 4500000000,
      "suggestion_basis": "the only printed/digital reading; other readings disagree (see candidates) — a human must confirm",
      "candidates": [
        {"value": 15000000000, "source": "handwritten", "engine": "tesseract", "raw_text": "15,000,000,000", "currency": "rial"},
        {"value": 4500000000, "source": "words", "engine": "tesseract", "raw_text": "چهار میلیارد و پانصد میلیون ریال", "label": "مبلغ به حروف"},
        {"value": 4500000000, "source": "printed", "engine": "easyocr", "raw_text": "4,500,000,000", "label": "مبلغ", "region": [1057, 908, 495, 31]}]},
    "currency": "rial", "amount_toman": null,
    "dest_iban": {"value": "IR440120000000001234568550", "partial": false, "checksum_valid": true, "bank": "Mellat", "confidence": "high"},
    "paid_at": {"jalali": "1405/08/24 11:42", "iso": "2026-11-15T11:42:00+03:30", "plausible": true, "confidence": "high"},
    "tracking_number": {"value": "140508241234567", "label": "شماره پیگیری", "all": [{"value": "140508241234567", "label": "شماره پیگیری"}], "confidence": "high"},
    "transfer_type": "satna",
    "document_status": "unknown",
    "sender":   {"name": {"value": "رضا احمدی", "confidence": "medium"}, "card": {"value": null}, "account": {"value": null}, "iban": {"value": null}, "bank": {"value": null}},
    "receiver": {"name": {"value": "علی کریمی", "confidence": "high"}, "card": {"value": null}, "account": {"value": null}, "iban": {"value": "IR440120000000001234568550", "confidence": "high"}, "bank": {"value": null}}
  },
  "settlement": {"confirmed_by_this_service": false, "note": "Settlement must be confirmed outside this service …"},
  "matches": [{"line_id": "L-17", "score": 0.4, "fits": false, "reasons": ["AMOUNT_UNREADABLE", "IBAN_FULL_MATCH", "NAME_MATCH"]}],
  "best_line_id": null,
  "suggested_route": "manager_review",
  "sender_question_fa": null,
  "warnings": ["AMOUNT_CONFLICT", "HANDWRITTEN_AMOUNT_DISAGREES", "STATUS_NOT_SUCCESS", "NO_MATCHING_LINE"],
  "engines": {"vlm": "skipped: no VLM provider configured (RR_VLM_PROVIDER)", "tesseract": "ok", "easyocr": "ok"},
  "processing_ms": 18000
}
```

Errors: `{"error": CODE, "detail": "...", "request_id": "..."}` with codes `UNAUTHORIZED` (401),
`BAD_EXPECTED_LINES` (400), `EMPTY_FILE` (400), `FILE_TOO_LARGE` / `IMAGE_TOO_LARGE` (413),
`UNSUPPORTED_TYPE` / `HEIC_NOT_SUPPORTED` (415), `CORRUPT_IMAGE` / `IMAGE_TOO_SMALL` (422),
`TIMEOUT` (504), `NO_API_KEY_CONFIGURED` (503). An engine failing never fails the request; it is
reported in `engines`.

### How the bot should call it

1. Debtor posts a photo → bot downloads the largest size Telegram offers (not the thumbnail).
2. Bot sends it with **all of that debtor's open lines** (`remaining_toman` = what is still owed).
3. Bot checks `image.sha256` / `image.phash` / `tracking_number_normalized` against its own DB for
   duplicates (this service does not remember anything). Use the tracking number for duplicate
   decisions **only when its confidence is `high`** (two engines agree): single-engine reads were
   wrong on real receipts (see below); `phash` is the safer signal for re-sent screenshots.
4. Bot applies its own policy on top of `suggested_route` (e.g. in trial mode, treat `match` as
   `confirm_with_sender`):
   * `match` → mark the line as *receipt received* (not *paid*), wait for the receiver's
     confirmation — settlement is never confirmed by this service.
   * `confirm_with_sender` → post `sender_question_fa` to the debtor, e.g.
     «این فیش را ۴۵۰٬۰۰۰٬۰۰۰ تومان به حساب …۸۵۵۰ خواندم. درست است؟»
   * `manager_review` → forward the photo, `warnings` and `matches` to the manager.
5. Log `request_id` with the bot's own record.

---

## Rules

### Fields and currency

Required: `amount_rial`, `dest_iban`, `paid_at`, `tracking_number`, `transfer_type`,
`document_status`. Also: `dest_bank` (from the IBAN bank code via
[`banks.json`](receipt_reader/data/banks.json)), `amount_words`, and the two parties:

* `sender` / `receiver`: `name`, `account`, `card` (last 4 digits only, `…4821`), `iban`, `bank` — each
  with confidence and candidates, filled only when visible. The side comes from the document's own
  labels: «از حساب / از کارت / مبدا / برداشت از / واریز کننده» → sender; «به حساب / به کارت / مقصد /
  واریز به / ذینفع / دارنده کارت» → receiver. «به نام / به شماره» belong to the section they appear in
  (within 4 lines); with no section they are not assigned.

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

Two candidates are **independent only if they come from different engines** (owner decision). The
same engine reading the full page and the zoomed strip is one source of evidence.

### Confidence rules (exact)

`trusted` = `printed` or `digital`.

**amount_rial**
1. **Any two readings with different values** (digits from any source, or the amount in words) →
   `conflict`, `value: null`, `AMOUNT_CONFLICT` (+ `MULTIPLE_AMOUNTS` when two printed/digital values
   differ, + `HANDWRITTEN_AMOUNT_DISAGREES` when handwriting differs from the single printed value).
   If exactly one printed/digital value exists it is returned as `suggested_value` for the human,
   never used for matching. Route: always `manager_review`.
2. All readings agree on V:
   * printed/digital: `high` if read by ≥ 2 engines **or** the amount in words agrees; else `medium`
     (one engine's read of an amount has no other validator);
   * handwriting only: `medium` if the words agree or ≥ 2 engines agree, else `low`;
   * words only: `low`.
3. Outside `[RR_MIN_AMOUNT_RIAL, RR_MAX_AMOUNT_RIAL]` → at most `low`.

**dest_iban** — pool = readings with role `dest`, else role `unknown` that are not also a source IBAN.
1. Two different checksum-valid IBANs → `conflict` (`IBAN_CONFLICT`). A partial reading whose digits
   are not a suffix of the full IBAN → `conflict`.
2. One valid IBAN: `high` if read from a trusted source and (≥ 2 engines, or its bank code is known —
   mod-97 + a known bank code are strong validators); `medium` if trusted otherwise; handwriting only →
   `medium` with ≥ 2 engines, else `low`. Bank named next to the destination ≠ bank of the code → `BANK_NAME_MISMATCH`, cap
   `medium`. Role not labelled on the document → cap `medium`.
3. Only checksum-failing IBANs → `low` + `INVALID_IBAN`; never used for an exact match.
4. Only partial (masked, last digits) → `PARTIAL_IBAN`, `checksum_valid: null` (never validated),
   `medium` if trusted else `low`; can only give `IBAN_LAST_DIGITS_MATCH`, never `match`. A masked IBAN
   is accepted only with the `IR` prefix or on a line labelled شبا — masked card numbers
   (`6037-70xx-xxxx-5190`) are never read as IBANs.

**paid_at** — trusted readings preferred. Different days, or different times on the same day, or a
handwritten date that differs from the printed one → `conflict` (`DATE_CONFLICT`). Trusted: `high`
with ≥ 2 engines, else `medium`; handwriting only: `medium` with ≥ 2 engines, else `low`. In the future (15 min
tolerance) or older than `RR_DATE_MAX_AGE_DAYS` (3) → `DATE_IMPLAUSIBLE`, `low`.

**tracking_number** — all numbers are returned in `all` with their own label. The primary one is
chosen by label priority (پیگیری › مرجع › رهگیری › ارجاع › تراکنش › سند). 5–30 digits (card-to-card
tracking numbers are often 5 digits); digits inside hex transaction ids are ignored. Two different
values for that label → `conflict` (`TRACKING_CONFLICT`). `high` only when ≥ 2 engines agree (a
tracking number has no checksum); one engine → `medium` (printed/digital) or `low`.

**sender / receiver fields** — same engine rule; readings from two engines that disagree → `conflict`
with `value: null` (all readings kept).

**transfer_type / document_status** — keyword votes (ساتنا، پایا، کارت به کارت … / موفق، ناموفق،
در انتظار، ثبت شده، انتقال یافت) plus the VLM's reading (double weight). Any "failed" → `failed`;
then `pending` («ثبت شده» = submitted, e.g. PAYA); then `success`. **Only explicit words count**: a
printed tracking number or a teller strip never makes a document `success` (owner decision).

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
* `manager_review` — any `conflict`; **any amount disagreement** (`AMOUNT_CONFLICT`,
  `HANDWRITTEN_AMOUNT_DISAGREES`, `MULTIPLE_AMOUNTS`); no amount; no fitting line (`NO_MATCHING_LINE`,
  `IBAN_MISMATCH`); `AMBIGUOUS_MATCH`; `UNKNOWN_CURRENCY`; status not `success`;
  `POSSIBLE_EDITING`; no expected lines sent; no engine available.
* `confirm_with_sender` — everything else (one line fits but something is `medium`/`low`, partial
  IBAN, two lines fit with different scores …), with the Persian question.

### Warning codes

`LOW_IMAGE_QUALITY`, `AMOUNT_CONFLICT`, `HANDWRITTEN_AMOUNT_DISAGREES`, `MULTIPLE_AMOUNTS`,
`UNKNOWN_CURRENCY`, `INVALID_IBAN`, `PARTIAL_IBAN`, `IBAN_MISMATCH`, `IBAN_CONFLICT`,
`BANK_NAME_MISMATCH`, `DATE_IMPLAUSIBLE`, `DATE_CONFLICT`, `TRACKING_CONFLICT`,
`STATUS_NOT_SUCCESS`, `NO_MATCHING_LINE`, `AMBIGUOUS_MATCH`, `POSSIBLE_EDITING`,
`NO_EXPECTED_LINES`, `NO_ENGINE_AVAILABLE`, `LOW_IMAGE_QUALITY`.

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
`4,500,000,000`); common OCR misreads of labels (بیگیری → پیگیری). A token containing an
unreadable character (`?`) is discarded whole — digits are never picked out of it. Amounts with a
leading zero (`09,000`) are rejected. A label on one line applies to the next line only if that line
holds nothing but a value. Persian number words parser in
[`numwords.py`](receipt_reader/numwords.py) — strict: any unknown word → no value.

## Security

* IBANs, card numbers and national IDs are masked in logs (last 4 digits kept); logs are JSON.
* Images are never written to disk; nothing is stored between requests.
* File size limit (`RR_MAX_UPLOAD_MB`), type checked from content, pixel-count limit, request timeout.
* API keys compared in constant time. Config only from environment / `.env` (see `.env.example`).

## Evaluation

See [`eval/README.md`](eval/README.md). `python eval/run_eval.py eval/real` reports the
dangerous-error rate (target 0), the auto-match share, wrong values reported with confidence `high`,
per-field exact/missed/wrong rates (including sender/receiver fields) and source confusion. Synthetic
images (`eval/synthetic/`, `tests/synth.py`) are for tests and the demo only.

## Measured on real receipts (2026-11-16)

**Set:** 13 images supplied by the owner — 7 Pasargad-app card-to-card screenshots (2 of them the
same receipt), 2 copies of one Pasargad PAYA screenshot, 1 Mellat-app internal transfer, 1 sample
paper deposit form with example data, 2 photographed thermal slips. Images are low resolution
(e.g. 540×1170). Ground truth was **transcribed by the assistant from the images and has not yet been
verified by the owner**. The images and labels are private (`eval/real/`, git-ignored).

**This is not a general accuracy figure.** 13 images, mostly one app's layout, almost no paper
SATNA/PAYA forms (the main real-world case), and no VLM was run (cloud is off; no self-hosted model).

Engines: Tesseract 5 `fas+eng` + EasyOCR 1.7 `fa`, CPU only.

| | Tesseract + EasyOCR |
|---|---|
| Dangerous errors (`match` with wrong amount / IBAN / line) | **0 / 13** |
| Routed `match` automatically | 0 / 13 (all `manager_review`) |
| Wrong values reported with confidence `high` | **0** |
| Amount read correctly | 0 / 12 that show an amount (none read; **0 wrong**) |
| Document status correct | 12 / 13 |
| Receiver card last 4 correct | 6 / 9 |
| Sender / receiver bank correct | 7 / 9 · 8 / 11 |
| Tracking number | 0 read correctly, 3 wrong (`medium`, Tesseract-only reads), 8 missed, 2 correctly empty |
| Time per receipt | ~15–35 s (EasyOCR on CPU dominates) |

**What this shows:** off-the-shelf local OCR cannot read the digits of these banking-app fonts.
EasyOCR outputs Arabic «٥» both for 5 and for the circle-shaped Persian zero «۰», so its digits are
treated as unreadable (`RR_EASYOCR_AMBIGUOUS_FIVE=true`); Tesseract turns Persian digits into Latin
letters or drops zeros. The safety rules held — nothing wrong was passed on as certain — but nothing
is automated either. Reading amounts reliably needs a VLM (self-hosted, or cloud if the owner enables
it) or a recognizer trained on these fonts; that has to be measured on this set next.

Fixed because of this set: masked card numbers taken as partial IBANs; digit fragments picked out
of unreadable tokens; a reference number inheriting the previous line's «پیگیری» label after an OCR
misread; an amount label carried onto the next line's reference number; `09,000` read as 9,000; «به
نام» with no section assigned to the receiver; 5-digit tracking numbers rejected.

## Demo

[`demo/receipt-reader-demo.mp4`](demo/receipt-reader-demo.mp4) — generated by
`python scripts/make_demo_video.py`: it starts the API with uvicorn, calls `/v1/health` and
`/v1/receipts/read` over HTTP for each synthetic receipt, and shows each live response next to the
ground truth (Tesseract + EasyOCR, no VLM). It shows behaviour, not accuracy. The same script makes
the private video of the real set:
`python scripts/make_demo_video.py --dataset eval/real --out eval/real/video/real-receipts-test.mp4 --private`
(never commit it: it contains customer data).

## Known limitations

* **Local OCR does not read the amounts on the owner's real receipts** (0 of 12 in the set above).
  Without a VLM, every receipt is expected to go to `manager_review`. The next step is measuring a
  self-hosted VLM (e.g. Qwen2.5-VL) — and the cloud VLM only if the owner enables it — on `eval/real`.
* The ground truth of the real set was written by the assistant and must be checked by the owner.
* Single-engine tracking numbers can be wrong (seen on the real set: a dropped digit, and hyphens read as zeros); they are `medium`. Use tracking numbers for duplicate decisions only when `high`.
* Tesseract cannot tell handwriting from print; on paper forms the inside/outside-teller-strip rule is
  a heuristic. Strip detection (boxed frame or regular print lines) may miss unboxed or faint stamps.
* Paper forms without explicit status words get `document_status: unknown` → `manager_review` (owner
  decision: a tracking number is not proof of success).
* Card-to-card receipts have no IBAN; expected lines are IBAN-based, so they never `fit` a line
  (→ `manager_review`). Matching by card last-4 would need a `card_last4` on the bot's lines.
* EasyOCR on CPU is slow (~15–35 s per receipt). The editing/tampering flags are weak signals.
* Region boxes are in the coordinates of the image each engine saw (VLM: oriented original;
  Tesseract/EasyOCR: cropped/deskewed/upscaled page).

## Layout

```
receipt_reader/
  api.py            FastAPI app (auth, request IDs, errors, timeouts, OpenAPI docs)
  pipeline.py       extract(): image → engines (parallel) → cross-check   |   match(): lines → route
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
