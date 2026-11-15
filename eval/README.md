# Evaluation

Engine choice, prompts and thresholds are decided **only** from numbers measured here
on real, labelled receipts.

## Folders

| Folder | What | In git? |
|---|---|---|
| `eval/real/` | The owner's private real receipts + ground truth | **No** (git-ignored) |
| `eval/synthetic/` | Generated images for plumbing checks and the demo | Yes |
| `eval/reports/` | JSON reports written by `run_eval.py` | No |

Synthetic images are for unit and integration tests only. **A number measured on
`eval/synthetic/` is not an accuracy figure and must never be quoted as one.** The
script prints a banner when any sample is marked `"synthetic": true`.

## Labelled-set format

One image plus one JSON file with the same stem:

```
eval/real/
  0001.jpg
  0001.json
  0002.png
  0002.json
```

```json
{
  "amount_rial": 4500000000,
  "currency": "rial",
  "dest_iban": "IR440120000000001234568550",
  "paid_at_jalali": "1405/08/24 11:42",
  "tracking_number": "140508241234567",
  "transfer_type": "satna",
  "document_status": "success",
  "sources": {"amount_rial": "printed"},
  "expected_lines": [
    {"line_id": "L-17", "iban": "IR440120000000001234568550", "holder_name": "علی کریمی",
     "amount_toman": 450000000, "remaining_toman": 450000000}
  ],
  "correct_line_id": "L-17",
  "evaluated_at": "2026-11-15T12:00:00+03:30",
  "notes": "handwritten amount on the form says 1,500,000,000"
}
```

* Use `null` for a field that is genuinely not on the receipt (the reader should return `null` too).
  Leave a key out if you did not label it.
* `dest_iban`: the full IBAN, or the visible part as `"...8550"` when the receipt only shows the end.
* `sources`: where the *correct* value is on the document (`printed` / `handwritten` / `digital`) —
  used for the source-confusion table.
* `expected_lines` / `correct_line_id`: what the bot would have sent and which line the receipt
  really pays (`null` if none). Needed for the dangerous-error rate.
* `evaluated_at`: the "now" for the date-plausibility rule (receipts are evaluated long after they
  were taken). Defaults to the real current time.

## Run

```bash
python eval/run_eval.py eval/real                        # engines from RR_ENGINES
python eval/run_eval.py eval/real --engines tesseract    # one engine alone
python eval/run_eval.py eval/real --engines vlm,tesseract
```

## What it reports

* **Dangerous-error rate** (most important; goal **0**): receipts routed `match` whose amount or
  IBAN is wrong, or whose chosen line is not `correct_line_id`. The script exits non-zero if any.
* **Auto-match share**: receipts routed `match`.
* Per field: exact-match rate, `null` rate, wrong rate.
* Source confusion: labelled source of the right value → source of the value the reader chose.
* Route distribution, per-receipt rows, engine status and timings (in the JSON report).

## Collecting the real set

```bash
python scripts/label_helper.py strip-gps eval/real   # remove GPS location (keeps orientation/Software EXIF)
python scripts/label_helper.py todo eval/real        # <stem>.todo.json skeletons for unlabelled images
python scripts/label_helper.py check eval/real       # validate labels before running the eval
```

`todo` puts the reader's own output under `_reader_suggestion` and leaves every ground-truth field
`null`. Fill the fields **from the image**, not from the suggestion — copying the reader's answer
makes the evaluation measure nothing. Rename to `<stem>.json` when done. `check` catches typos
(an IBAN that fails mod-97, a float amount, a bad date, a `correct_line_id` not in the lines).

* 50 receipts is enough to start; 150+ before trusting thresholds. Keep the real mix (mostly paper
  SATNA/PAYA forms) and deliberately include the hard ones: wrong handwriting, rial/toman unclear,
  failed transfers, blurry photos, two lines with the same IBAN.
* Strip EXIF location data before sharing images.
