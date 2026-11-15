"""One test (or more) per owner decision / requirement. Layouts copy real receipts' structure
(Pasargad app card-to-card, Mellat app, Melli thermal slip) with made-up values."""
import io
import json
import subprocess
import sys

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from receipt_reader.api import create_app
from receipt_reader.config import Settings
from receipt_reader.engines.base import EngineOutput
from receipt_reader.engines.scripted import ScriptedEngine
from receipt_reader.iban import find_ibans, make_iban
from receipt_reader.pipeline import ReceiptReader
from receipt_reader.schema import ExpectedLine
from receipt_reader.textparse import TextLine, parse_lines

from .conftest import IBAN_MELLAT, NOW
from .helpers import C, good_printed, line, run


def _parse(*texts, source="digital"):
    cands, hints = parse_lines([TextLine(t, source=source) for t in texts], "t")
    return {(c.field, str(c.value)) for c in cands}, cands, hints


# ---------------------------------------------------------------- sender / receiver
def test_card_to_card_parties_pasargad_layout():
    got, cands, hints = _parse(
        "شماره تراکنش 9ab3c41d07f2",
        "شماره کارت 5022-29xx-xxxx-4821",
        "بانک مبدا پاسارگاد",
        "کارت مقصد 6219-86xx-xxxx-3307",
        "بانک مقصد سامان",
        "دارنده کارت کاوه صادقی",
        "مبلغ انتقال 36,500,000 ریال",
        "شماره پیگیری 48213",
        "شماره ارجاع 71122334455",
        "تاریخ 1405/07/03", "ساعت 12:40:25",
        "نتیجه تراکنش موفق",
    )
    assert ("sender_card", "5022-29xx-xxxx-4821") in got
    assert ("receiver_card", "6219-86xx-xxxx-3307") in got
    assert ("sender_bank", "057") in got and ("receiver_bank", "056") in got
    assert ("receiver_name", "کاوه صادقی") in got
    assert ("amount", "36500000") in got
    assert ("tracking_number", "48213") in got          # 5-digit tracking number
    assert ("tracking_number", "71122334455") in got
    assert not any(f == "tracking_number" and v.startswith("68577") for f, v in got)  # not from the hex id
    assert not any(f == "iban" for f, _ in got)         # masked cards are never partial IBANs
    assert ("paid_at", "1405/07/03 12:40") in got
    assert hints.status_votes.get("success") and hints.transfer_votes.get("card_to_card")


def test_internal_transfer_parties_mellat_layout():
    got, cands, _ = _parse(
        "برداشت از حساب: قرض الحسنه اشخاص حقیقی",
        "به شماره: 3300112244",
        "به نام: مریم رستمی",
        "واریز به حساب: کوتاه مدت اشخاص حقوقی",
        "به شماره: 4400556677",
        "به نام: شرکت نمونه",
        "مبلغ: 2,450,000 ریال",
        "کارمزد: 360 ریال",
        "شناسه واریز: 5500667788",
        "شماره پیگیری: 2299887766",
    )
    assert ("sender_account", "3300112244") in got and ("receiver_account", "4400556677") in got
    assert ("sender_name", "مریم رستمی") in got and ("receiver_name", "شرکت نمونه") in got
    assert {v for f, v in got if f == "amount"} == {"2450000"}  # fee excluded
    assert {v for f, v in got if f == "tracking_number"} == {"2299887766"}  # deposit id excluded


def test_thermal_slip_parties_melli_layout():
    got, _, _ = _parse(
        "از: بانک ملی ایران",
        "از کارت : **** 6120",
        "از حساب : 0399001122334",
        "مبلغ انتقالی: 12,750,000 ریال",
        "به: بانک ملت",
        "به کارت : **** 8841",
        "به حساب: 7712003344",
        "به نام : سینا کریمی",
    )
    assert ("sender_bank", "017") in got and ("receiver_bank", "012") in got
    assert ("sender_card", "**** 6120") in got and ("receiver_card", "**** 8841") in got
    assert ("sender_account", "0399001122334") in got and ("receiver_account", "7712003344") in got
    assert ("receiver_name", "سینا کریمی") in got
    assert not any(f == "iban" for f, _ in got)


def test_parties_resolved_in_response(settings, now):
    cands = good_printed(IBAN_MELLAT) + [
        C("sender_card", "5022-29xx-xxxx-4821", engine="a"), C("sender_card", "502229xxxxxx4821", engine="b"),
        C("receiver_name", "علی کریمی", engine="a"), C("sender_bank", "057", engine="a")]
    f, *_ = run(cands, settings, now)
    assert f.sender.card.value == "…4821" and f.sender.card.confidence == "high"
    assert f.sender.bank.value == "Pasargad"
    assert f.receiver.name.value == "علی کریمی"
    assert f.receiver.iban.value == IBAN_MELLAT


# -------------------------------------------------------------------- partial IBAN
def test_masked_iban_never_checksum_validated(settings, now):
    full = make_iban("0570000000000123450001")
    masked = f"{full[:4]} {full[4:8]} xxxx xxxx xx{full[-8:-6]} {full[-6:-2]} {full[-2:]}"
    found = find_ibans(f"شبا {masked}")
    assert found and found[0][1].partial and found[0][1].checksum_valid is None
    assert found[0][1].value == "..." + full[-8:]
    cands = [c for c in good_printed(full) if c.field != "iban"] + [
        C("iban", found[0][1].value, role="dest", engine="a"), C("iban", found[0][1].value, role="dest", engine="b")]
    f, matches, best, route, _, warnings = run(cands, settings, now, lines=[line("L-1", full, 450_000_000)])
    assert f.dest_iban.partial and f.dest_iban.checksum_valid is None
    assert "IBAN_FULL_MATCH" not in matches[0].reasons and "IBAN_LAST_DIGITS_MATCH" in matches[0].reasons
    assert route != "match" and "PARTIAL_IBAN" in warnings


# ----------------------------------------------------------- status vs settlement
def test_tracking_number_alone_is_not_success(settings, now):
    cands = [c for c in good_printed(IBAN_MELLAT) if c.field != "document_status"]
    f, _, _, route, _, warnings = run(cands, settings, now, lines=[line("L-17", IBAN_MELLAT, 450_000_000)])
    assert f.document_status == "unknown" and "STATUS_NOT_SUCCESS" in warnings
    assert route == "manager_review"


def test_registered_paya_is_pending():
    _, _, hints = _parse("وضعیت ارسالی - ثبت شده")
    assert hints.status_votes == {"pending": 1}


def _reader(settings):
    cands = good_printed(IBAN_MELLAT)

    def eng(name):
        return ScriptedEngine(name, lambda li, p: EngineOutput(engine=name, candidates=[c for c in cands
                                                                                       if c.engine == name]))
    return ReceiptReader(settings, {"a": eng("a"), "b": eng("b")})


def _jpeg():
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(3).integers(0, 255, (700, 900, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def test_settlement_is_never_confirmed(settings):
    res = _reader(settings).read(_jpeg(), [ExpectedLine(line_id="L-17", iban=IBAN_MELLAT, amount_toman=450_000_000)],
                                 now=NOW)
    assert res.suggested_route == "match"
    assert res.settlement.confirmed_by_this_service is False


# ---------------------------------------------------- extraction independent of lines
def test_extraction_is_identical_with_or_without_lines(settings):
    r = _reader(settings)
    img = _jpeg()
    a = r.read(img, [], now=NOW).fields.model_dump()
    b = r.read(img, [ExpectedLine(line_id="X", iban=IBAN_MELLAT, amount_toman=1)], now=NOW).fields.model_dump()
    assert a == b


def test_extract_endpoint_has_no_matching(settings):
    s = Settings(api_keys="k", engines="a,b", _env_file=None)
    c = TestClient(create_app(s, _reader(s)))
    r = c.post("/v1/receipts/extract", headers={"X-API-Key": "k"}, files={"image": ("r.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert "suggested_route" not in body and "matches" not in body
    assert body["fields"]["amount_rial"]["value"] == 4_500_000_000 and body["settlement"]["confirmed_by_this_service"] is False


# ------------------------------------------------------------------------- Swagger
def test_openapi_documents_every_parameter_and_schema():
    s = Settings(api_keys="k", _env_file=None)
    spec = TestClient(create_app(s, ReceiptReader(s, {}))).get("/openapi.json").json()
    comps = spec["components"]["schemas"]
    for name in ("ReadResponse", "Extraction", "HealthResponse", "ErrorResponse", "ExpectedLine", "Fields",
                 "Party", "AmountField", "IbanField", "LineMatch", "Settlement"):
        assert name in comps, name
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            assert op.get("summary"), (path, method)
            for p in op.get("parameters", []):
                assert p.get("description"), (path, p["name"])
            body = op.get("requestBody")
            if body:
                ref = body["content"]["multipart/form-data"]["schema"]["$ref"].split("/")[-1]
                for prop, sch in comps[ref]["properties"].items():
                    assert sch.get("description"), (path, prop)
            if path.startswith("/v1/receipts"):
                assert {"200", "401", "413", "415", "422"} <= set(op["responses"]), path
    for prop in ("line_id", "amount_toman", "remaining_toman", "iban"):
        assert comps["ExpectedLine"]["properties"][prop].get("description"), prop


# -------------------------------------------------------------- no cloud dependency
def test_local_mode_never_imports_a_cloud_sdk():
    code = ("import sys; from receipt_reader.config import Settings; from receipt_reader.engines import build_engines;"
            "from receipt_reader.api import create_app; s=Settings(_env_file=None, api_keys='k');"
            "e=build_engines(s); [x.available() for x in e.values()]; create_app(s);"
            "print('anthropic' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".")
    assert out.stdout.strip() == "False", out.stderr
