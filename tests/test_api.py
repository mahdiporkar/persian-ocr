import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from receipt_reader.api import create_app
from receipt_reader.config import Settings
from receipt_reader.engines.base import EngineOutput
from receipt_reader.engines.scripted import ScriptedEngine
from receipt_reader.pipeline import ReceiptReader

from .conftest import IBAN_MELLAT
from .helpers import good_printed


def _png(w=800, h=600, color=(255, 255, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    s = Settings(api_keys="k1,k2", engines="a,b", _env_file=None, date_max_age_days=100000)
    cands = good_printed(IBAN_MELLAT)

    def eng(name):
        return ScriptedEngine(name, lambda li, p: EngineOutput(engine=name,
                                                               candidates=[c for c in cands if c.engine == name]))

    broken = ScriptedEngine("broken", lambda li, p: (_ for _ in ()).throw(RuntimeError("boom")))
    reader = ReceiptReader(s, {"a": eng("a"), "b": eng("b"), "broken": broken,
                               "off": ScriptedEngine("off", lambda li, p: None, ok=False, reason="not installed")})
    return TestClient(create_app(s, reader))


H = {"X-API-Key": "k1"}
LINES = json.dumps([{"line_id": "L-17", "iban": IBAN_MELLAT, "holder_name": "x", "amount_toman": 450000000,
                     "remaining_toman": 450000000}])


def test_health(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["engines"]["a"]["loaded"] is True
    assert body["engines"]["off"]["loaded"] is False
    assert r.headers["X-Request-ID"]


def test_requires_api_key(client):
    r = client.post("/v1/receipts/read", files={"image": ("r.png", _png(), "image/png")})
    assert r.status_code == 401
    assert r.json()["error"] == "UNAUTHORIZED"
    assert r.json()["request_id"] == r.headers["X-Request-ID"]


def test_read_and_match(client):
    r = client.post("/v1/receipts/read", headers={**H, "X-Request-ID": "abc-1"},
                    files={"image": ("r.png", _png(), "image/png")}, data={"expected_lines": LINES})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["request_id"] == "abc-1" and r.headers["X-Request-ID"] == "abc-1"
    assert body["suggested_route"] == "match"
    assert body["best_line_id"] == "L-17"
    assert body["fields"]["amount_rial"]["value"] == 4_500_000_000
    assert body["fields"]["amount_toman"] == 450_000_000
    assert isinstance(body["fields"]["amount_rial"]["value"], int)
    assert len(body["image"]["sha256"]) == 64 and body["image"]["phash"]
    assert body["engines"] == {"a": "ok", "b": "ok"}


def test_engine_failure_is_reported_not_fatal(client):
    r = client.post("/v1/receipts/read", headers=H, files={"image": ("r.png", _png(), "image/png")},
                    data={"expected_lines": LINES, "engines": "a,broken,off"})
    assert r.status_code == 200
    e = r.json()["engines"]
    assert e["a"] == "ok" and e["broken"].startswith("error") and e["off"].startswith("skipped")


def test_rejects_non_image(client):
    r = client.post("/v1/receipts/read", headers=H, files={"image": ("x.png", b"%PDF-1.4 hello", "image/png")})
    assert r.status_code == 415 and r.json()["error"] == "UNSUPPORTED_TYPE"


def test_mime_is_checked_against_content(client):
    # claims JPEG, is actually text
    r = client.post("/v1/receipts/read", headers=H, files={"image": ("x.jpg", b"hello world" * 100, "image/jpeg")})
    assert r.status_code == 415


def test_rejects_tiny_image(client):
    r = client.post("/v1/receipts/read", headers=H, files={"image": ("t.png", _png(100, 80), "image/png")})
    assert r.status_code == 422 and r.json()["error"] == "IMAGE_TOO_SMALL"


def test_rejects_too_large():
    s = Settings(api_keys="k1", engines="a", _env_file=None, max_upload_mb=0.001)
    c = TestClient(create_app(s, ReceiptReader(s, {})))
    r = c.post("/v1/receipts/read", headers=H, files={"image": ("r.png", _png(900, 900, (1, 2, 3)) + b"0" * 5000,
                                                                "image/png")})
    assert r.status_code == 413


def test_bad_expected_lines(client):
    r = client.post("/v1/receipts/read", headers=H, files={"image": ("r.png", _png(), "image/png")},
                    data={"expected_lines": "[{\"line_id\": 1}"})
    assert r.status_code == 400 and r.json()["error"] == "BAD_EXPECTED_LINES"


def test_no_api_key_configured():
    s = Settings(api_keys="", _env_file=None)
    c = TestClient(create_app(s, ReceiptReader(s, {})))
    r = c.post("/v1/receipts/read", headers=H, files={"image": ("r.png", _png(), "image/png")})
    assert r.status_code == 503


def test_oversized_body_rejected_before_parsing(client):
    big = b"0" * (11 * 1024 * 1024)
    r = client.post("/v1/receipts/read", headers=H, files={"image": ("r.png", big, "image/png")})
    assert r.status_code == 413 and r.json()["error"] == "FILE_TOO_LARGE"


@pytest.mark.parametrize("bad", [
    [{"line_id": "L-1", "amount_toman": 0}],
    [{"line_id": "L-1", "amount_toman": -5}],
    [{"line_id": "", "amount_toman": 5}],
    [{"line_id": "L-1", "amount_toman": 5, "remaining_toman": -1}],
])
def test_expected_lines_validation(client, bad):
    r = client.post("/v1/receipts/read", headers=H, files={"image": ("r.png", _png(), "image/png")},
                    data={"expected_lines": json.dumps(bad)})
    assert r.status_code == 400
