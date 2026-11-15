"""VLM adapter with an injected transport: no network, checks blind reading + normalization."""
import io

from PIL import Image

from receipt_reader.config import Settings
from receipt_reader.engines.vlm import PROMPT_FULL, PROMPT_STRIP, RESPONSE_SCHEMA, VLMEngine
from receipt_reader.pipeline import ReceiptReader
from receipt_reader.schema import ExpectedLine

from .conftest import IBAN_MELLAT, NOW

FULL = {
    "document_kind": "paper_form", "transfer_type": "satna", "document_status": "unknown",
    "printed_strip_present": True, "printed_strip_box": [100, 600, 900, 950], "suspected_editing": False,
    "notes": "",
    "readings": [
        {"field": "amount", "raw_text": "۱٬۵۰۰٬۰۰۰٬۰۰۰", "source": "handwritten", "role": "unknown", "label": "مبلغ",
         "unit": "rial", "box": None},
        {"field": "amount_words", "raw_text": "چهارصد و پنجاه میلیون ریال", "source": "handwritten",
         "role": "unknown", "label": "مبلغ به حروف", "unit": "rial", "box": None},
        {"field": "amount", "raw_text": "450,000,000", "source": "printed", "role": "unknown", "label": "مبلغ",
         "unit": "rial", "box": [500, 700, 800, 720]},
        {"field": "iban", "raw_text": " ".join(IBAN_MELLAT[i:i + 4] for i in range(0, 26, 4)), "source": "printed",
         "role": "dest", "label": "شبا", "unit": "none", "box": None},
        {"field": "paid_at", "raw_text": "۱۴۰۵/۰۸/۲۴ ۱۱:۴۲", "source": "printed", "role": "unknown", "label": "تاریخ",
         "unit": "none", "box": None},
        {"field": "tracking_number", "raw_text": "140508241234567", "source": "printed", "role": "unknown",
         "label": "شماره پیگیری", "unit": "none", "box": None},
        {"field": "tracking_number", "raw_text": "1405?8241234567", "source": "handwritten", "role": "unknown",
         "label": "شماره پیگیری", "unit": "none", "box": None},
    ],
}
STRIP = dict(FULL, readings=[r for r in FULL["readings"] if r["source"] == "printed"])


def _jpeg():
    """A noisy 'photo' (not a flat screenshot) so the strip re-read path runs."""
    import numpy as np
    arr = np.random.default_rng(1).integers(150, 255, (900, 1200, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def test_prompt_never_contains_expected_values():
    prompts = []

    def transport(jpeg, prompt):
        prompts.append(prompt)
        return FULL if prompt == PROMPT_FULL else STRIP

    s = Settings(_env_file=None, engines="vlm", vlm_provider="anthropic")
    reader = ReceiptReader(s, {"vlm": VLMEngine(s, transport=transport)})
    lines = [ExpectedLine(line_id="L-SECRET", iban=IBAN_MELLAT, amount_toman=33_333_333, holder_name="نام محرمانه")]
    res = reader.read(_jpeg(), lines, now=NOW)
    assert prompts == [PROMPT_FULL, PROMPT_STRIP]
    for p in prompts:
        assert "333" not in p and "L-SECRET" not in p and "محرمانه" not in p and IBAN_MELLAT not in p
    f = res.fields
    # handwritten 1.5B disagrees with printed/words 450M -> conflict; printed is only suggested
    assert f.amount_rial.value is None and f.amount_rial.confidence == "conflict"
    assert f.amount_rial.suggested_value == 450_000_000
    assert "HANDWRITTEN_AMOUNT_DISAGREES" in res.warnings
    assert f.dest_iban.value == IBAN_MELLAT and f.dest_iban.confidence == "high"  # mod-97 + known bank
    assert f.tracking_number.value == "140508241234567"  # the '?' reading was dropped, not guessed
    assert f.tracking_number.confidence == "medium"  # full + strip of ONE engine = one source
    assert f.transfer_type == "satna"
    assert res.suggested_route == "manager_review"


def test_cloud_vlm_disabled_by_default():
    s = Settings(_env_file=None, vlm_provider="anthropic")
    ok, why = VLMEngine(s).available()
    assert not ok and "RR_ALLOW_CLOUD_VLM" in why


def test_self_hosted_vlm_allowed_without_cloud_switch():
    s = Settings(_env_file=None, vlm_provider="openai_compatible", vlm_base_url="http://127.0.0.1:8000/v1")
    eng = VLMEngine(s)
    assert eng.is_cloud() is False
    assert eng.available()[0]


def test_public_openai_compatible_endpoint_counts_as_cloud():
    s = Settings(_env_file=None, vlm_provider="openai_compatible", vlm_base_url="https://api.example.com/v1",
                 vlm_self_hosted=False)
    assert VLMEngine(s).is_cloud() and not VLMEngine(s).available()[0]


def test_schema_is_strict():
    assert RESPONSE_SCHEMA["additionalProperties"] is False
    item = RESPONSE_SCHEMA["properties"]["readings"]["items"]
    assert set(item["required"]) == set(item["properties"])
