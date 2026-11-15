"""Engine A: vision-language model adapter.

One adapter, two transports:
  * provider "anthropic"          — Claude via the official Anthropic SDK (cloud)
  * provider "openai_compatible"  — any self-hosted open model behind an
                                    OpenAI-compatible /chat/completions endpoint
                                    (vLLM, Ollama, LM Studio, llama.cpp server …),
                                    e.g. Qwen2.5-VL.

The prompt never contains expected values. The model is asked to transcribe
what it sees, per field, with the source of each reading (printed /
handwritten / digital), and Python does all normalization and validation.
"""
from __future__ import annotations

import base64
import ipaddress
import json
import logging
import re
import socket
from typing import Any, Optional
from urllib.parse import urlparse

from PIL import Image

from ..config import Settings
from ..dates import parse_datetime
from ..iban import parse_iban
from ..imaging import LoadedImage, Prepared, encode_jpeg, to_pil
from ..normalize import digits_only, parse_amount_digits, normalize_text
from ..numwords import parse_persian_number_words
from ..schema import Candidate
from .base import Engine, EngineOutput

log = logging.getLogger("receipt_reader.vlm")

FIELDS = ["amount", "amount_words", "iban", "paid_at", "tracking_number",
          "sender_name", "sender_account", "sender_card", "sender_bank",
          "receiver_name", "receiver_account", "receiver_card", "receiver_bank"]

_NULLABLE_STR = {"type": ["string", "null"]}
_BOX = {"type": ["array", "null"], "items": {"type": "integer"}}

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_kind": {"type": "string", "enum": ["paper_form", "screenshot", "atm_slip", "other"]},
        "transfer_type": {"type": "string", "enum": ["satna", "paya", "internal", "card_to_card", "unknown"]},
        "document_status": {"type": "string", "enum": ["success", "failed", "pending", "unknown"]},
        "printed_strip_present": {"type": "boolean"},
        "printed_strip_box": _BOX,
        "suspected_editing": {"type": "boolean"},
        "readings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "enum": FIELDS},
                    "raw_text": {"type": "string"},
                    "source": {"type": "string", "enum": ["printed", "handwritten", "digital"]},
                    "role": {"type": "string", "enum": ["dest", "source", "unknown"]},
                    "label": _NULLABLE_STR,
                    "unit": {"type": "string", "enum": ["rial", "toman", "none"]},
                    "box": _BOX,
                },
                "required": ["field", "raw_text", "source", "role", "label", "unit", "box"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["document_kind", "transfer_type", "document_status", "printed_strip_present",
                 "printed_strip_box", "suspected_editing", "readings", "notes"],
    "additionalProperties": False,
}

PROMPT_FULL = """You are transcribing an Iranian bank transfer receipt (فیش واریز) for an audit system.
It may be a paper SATNA/PAYA form (فرم ساتنا / پایا) photographed with a phone, with handwritten
fields and a machine-printed strip or stamp added by the bank teller; or a mobile/internet-banking
screenshot; or an ATM slip.

Transcribe — do not interpret or correct. Report every reading you can see for these fields:
- amount: an amount written in digits (مبلغ). Copy the digits exactly as written, separators included.
  Do NOT report balances (مانده / موجودی) or fees (کارمزد) as amounts.
- amount_words: the amount written in Persian words (مبلغ به حروف), copied exactly.
- iban: any IBAN / شبا. role=dest for the destination/beneficiary account (به حساب، ذینفع، مقصد),
  role=source for the payer's account (از حساب، مبدا). Copy digits exactly; masked digits as '*'.
- paid_at: the transaction date and time as printed (تاریخ / ساعت), copied exactly.
- tracking_number: every tracking/reference number (شماره پیگیری، کد پیگیری، شماره مرجع، کد رهگیری)
  with its label copied into `label`. Account numbers and card numbers are NOT tracking numbers.
- sender_* = the payer side (از حساب، از کارت، مبدا، برداشت از، واریز کننده);
  receiver_* = the payee side (به حساب، به کارت، مقصد، واریز به، ذینفع، دارنده کارت).
  sender_name / receiver_name: holder names; sender_account / receiver_account: account numbers
  (not IBANs); sender_card / receiver_card: card numbers exactly as printed, masked digits as 'x';
  sender_bank / receiver_bank: bank names written for that side. Only report what is visible.

If the same field appears more than once (e.g. handwritten on the form AND printed on the teller
strip), report EACH occurrence as a separate reading. Never merge or choose between them.

For every reading set `source`:
- "printed": machine-printed (teller strip, stamp, printed slip),
- "handwritten": written by hand,
- "digital": text in a screenshot.
Set `unit` to rial/toman only if that word is visible next to that reading, else "none".
`box` is [x0, y0, x1, y1] of the reading in 0-1000 normalized image coordinates, or null.
If a character is unreadable, write '?' in its place rather than guessing.

Also report: document_kind, transfer_type (satna/paya/internal/card_to_card/unknown),
document_status (success/failed/pending/unknown — only from explicit words such as موفق / ناموفق /
در انتظار / ثبت شده; a teller's printed strip or a tracking number alone counts as unknown),
whether a machine-printed teller strip
or stamp is present and its box, and suspected_editing=true only for clear visual signs of digital
editing (mismatched fonts in one line, pasted digits, misaligned background).
"""

PROMPT_STRIP = """This image is a zoomed crop of the machine-printed strip or stamp that a bank teller
added to an Iranian transfer form. Transcribe every field you can see on it, exactly as printed,
using the same rules: amount (digits), amount_words, iban (role dest unless the label says source),
paid_at, tracking_number (with label), names. All readings here have source "printed" unless clearly
handwritten. Do not guess unreadable characters; write '?' instead."""


def _is_private_host(url: Optional[str]) -> bool:
    if not url:
        return False
    host = urlparse(url).hostname or ""
    if host in ("localhost",) or host.endswith(".local") or host.endswith(".internal"):
        return True
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(host))
        return ip.is_private or ip.is_loopback
    except Exception:
        return False


class VLMEngine(Engine):
    name = "vlm"

    def __init__(self, settings: Settings, transport=None):
        self.s = settings
        self._transport = transport  # injectable for tests: (image_bytes, prompt) -> dict

    # ---------------------------------------------------------------- gating
    def is_cloud(self) -> bool:
        if self.s.vlm_provider == "anthropic":
            return True
        if self.s.vlm_provider == "openai_compatible":
            if self.s.vlm_self_hosted is not None:
                return not self.s.vlm_self_hosted
            return not _is_private_host(self.s.vlm_base_url)
        return False

    def available(self) -> tuple[bool, str]:
        if self._transport is not None:
            return True, "ok (injected transport)"
        p = self.s.vlm_provider
        if p in ("", "none"):
            return False, "no VLM provider configured (RR_VLM_PROVIDER)"
        if self.is_cloud() and not self.s.allow_cloud_vlm:
            return False, "cloud VLM disabled: images may not leave the server (RR_ALLOW_CLOUD_VLM=false)"
        if p == "anthropic":
            try:
                import anthropic  # noqa: F401
            except Exception:
                return False, "anthropic SDK not installed"
            return True, f"ok ({self.s.vlm_model}, cloud)"
        if p == "openai_compatible":
            if not self.s.vlm_base_url:
                return False, "RR_VLM_BASE_URL is required for openai_compatible"
            return True, f"ok ({self.s.vlm_model} @ {urlparse(self.s.vlm_base_url).hostname})"
        return False, f"unknown provider {p}"

    # ------------------------------------------------------------- transport
    def _call(self, jpeg: bytes, prompt: str) -> dict:
        if self._transport is not None:
            return self._transport(jpeg, prompt)
        if self.s.vlm_provider == "anthropic":
            return self._call_anthropic(jpeg, prompt)
        return self._call_openai_compatible(jpeg, prompt)

    def _call_anthropic(self, jpeg: bytes, prompt: str) -> dict:
        import anthropic

        client = anthropic.Anthropic(api_key=self.s.vlm_api_key or None, timeout=self.s.vlm_timeout_s,
                                     max_retries=1)
        resp = client.beta.messages.create(
            model=self.s.vlm_model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={
                "effort": self.s.vlm_effort,
                "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
            },
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                 "data": base64.standard_b64encode(jpeg).decode()}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("VLM declined the request")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("VLM output truncated")
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)

    def _call_openai_compatible(self, jpeg: bytes, prompt: str) -> dict:
        import httpx

        url = self.s.vlm_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.s.vlm_api_key}"} if self.s.vlm_api_key else {}
        body = {
            "model": self.s.vlm_model,
            "temperature": 0,
            "max_tokens": 4096,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "receipt", "schema": RESPONSE_SCHEMA, "strict": True}},
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
                    {"type": "text", "text": prompt + "\nAnswer with JSON only."},
                ],
            }],
        }
        r = httpx.post(url, json=body, headers=headers, timeout=self.s.vlm_timeout_s)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
        return json.loads(content)

    # ------------------------------------------------------------------ read
    def read(self, loaded: LoadedImage, prep: Prepared) -> EngineOutput:
        out = EngineOutput(engine=self.name)
        img = prep.oriented
        jpeg = encode_jpeg(img, self.s.vlm_max_side)
        data = self._call(jpeg, PROMPT_FULL)
        out.debug["full"] = data
        default_src = "digital" if loaded.kind == "screenshot" else None
        self._collect(out, data, "full", img.size, default_src)

        if data.get("suspected_editing"):
            out.suspected_editing = True
        if self.s.vlm_strip_reread and loaded.kind != "screenshot":
            crop = None
            box = data.get("printed_strip_box") if data.get("printed_strip_present") else None
            if box and len(box) == 4:
                W, H = img.size
                x0, y0, x1, y1 = [int(v) for v in box]
                px0, py0 = max(0, int(x0 / 1000 * W) - 20), max(0, int(y0 / 1000 * H) - 20)
                px1, py1 = min(W, int(x1 / 1000 * W) + 20), min(H, int(y1 / 1000 * H) + 20)
                if px1 - px0 > 40 and py1 - py0 > 20:
                    crop = img.crop((px0, py0, px1, py1))
            elif prep.strip_box:
                x, y, w, h = prep.strip_box
                crop = to_pil(prep.document[max(0, y - 10): y + h + 10, max(0, x - 10): x + w + 10])
            if crop is not None:
                if crop.width < 1200:
                    f = 1200 / crop.width
                    crop = crop.resize((int(crop.width * f), int(crop.height * f)), Image.LANCZOS)
                sdata = self._call(encode_jpeg(crop, self.s.vlm_max_side), PROMPT_STRIP)
                out.debug["strip"] = sdata
                self._collect(out, sdata, "strip", None, "printed")
                out.printed_strip_box = prep.strip_box
        return out

    def _collect(self, out: EngineOutput, data: dict, pass_id: str, size, default_src: Optional[str]) -> None:
        tt = data.get("transfer_type")
        if tt and tt != "unknown":
            out.candidates.append(Candidate(field="transfer_type", value=tt, raw_text=tt, source="digital"
                                            if default_src == "digital" else "printed", engine=self.name,
                                            pass_id=pass_id))
        st = data.get("document_status")
        if st and st != "unknown":
            out.candidates.append(Candidate(field="document_status", value=st, raw_text=st,
                                            source=default_src or "printed", engine=self.name, pass_id=pass_id))
        for r in data.get("readings", []) or []:
            for c in reading_to_candidates(r, self.name, pass_id, size, default_src):
                out.candidates.append(c)


def _box_px(box, size) -> Optional[list[int]]:
    if not box or len(box) != 4 or size is None:
        return None
    W, H = size
    x0, y0, x1, y1 = box
    return [int(x0 / 1000 * W), int(y0 / 1000 * H), int((x1 - x0) / 1000 * W), int((y1 - y0) / 1000 * H)]


def reading_to_candidates(r: dict, engine: str, pass_id: str, size, default_src: Optional[str]) -> list[Candidate]:
    """Normalize one VLM reading in Python. The model's own normalization is not trusted."""
    field = r.get("field")
    raw = (r.get("raw_text") or "").strip()
    if not raw or "?" in raw and field in ("amount", "iban", "tracking_number"):
        return []  # unreadable characters: do not invent a value
    src = r.get("source") or default_src or "printed"
    if default_src == "digital":
        src = "digital"
    unit = r.get("unit")
    currency = unit if unit in ("rial", "toman") else None
    common = dict(raw_text=raw, engine=engine, pass_id=pass_id, region=_box_px(r.get("box"), size),
                  label=r.get("label"))
    t = normalize_text(raw)
    if currency is None:
        if "ریال" in t:
            currency = "rial"
        elif "تومان" in t:
            currency = "toman"

    if field == "amount":
        v = parse_amount_digits(raw)
        if v is None:
            return []
        return [Candidate(field="amount", value=v, source=src, currency=currency, **common)]
    if field == "amount_words":
        w = parse_persian_number_words(raw)
        if w is None:
            return []
        return [Candidate(field="amount", value=w.value, source="words", currency=w.currency or currency, **common)]
    if field == "iban":
        info = parse_iban(raw)
        if info is None:
            return []
        return [Candidate(field="iban", value=info.value, source=src, role=r.get("role") or "unknown", **common)]
    if field == "paid_at":
        p = parse_datetime(raw)
        if p is None:
            return []
        return [Candidate(field="paid_at", value=p.jalali, source=src, **common)]
    if field == "tracking_number":
        d = digits_only(raw)
        if not (5 <= len(d) <= 30):
            return []
        return [Candidate(field="tracking_number", value=d, source=src, **common)]
    side_role = "dest" if field.startswith("receiver_") else "source"
    if field in ("sender_name", "receiver_name"):
        return [Candidate(field=field, value=raw, source=src, role=side_role, **common)]
    if field in ("sender_account", "receiver_account"):
        d = digits_only(raw)
        return [Candidate(field=field, value=d, source=src, role=side_role, **common)] if len(d) >= 6 else []
    if field in ("sender_card", "receiver_card"):
        d = digits_only(raw)
        return [Candidate(field=field, value=raw, source=src, role=side_role, **common)] if len(d) >= 4 else []
    if field in ("sender_bank", "receiver_bank"):
        from ..iban import detect_bank_names
        return [Candidate(field=field, value=code, source=src, role=side_role, **common)
                for code in detect_bank_names(raw)]
    return []
