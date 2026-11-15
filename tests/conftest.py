import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from receipt_reader.config import Settings  # noqa: E402
from receipt_reader.dates import TEHRAN  # noqa: E402
from receipt_reader.iban import make_iban  # noqa: E402

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=TEHRAN)  # = 1405/08/24 12:00 Tehran

# A valid Mellat IBAN (bank code 012) and a valid Melli IBAN (017)
IBAN_MELLAT = make_iban("0120000000001234568550")
IBAN_MELLI = make_iban("0170000000003456781234")


@pytest.fixture
def settings():
    return Settings(api_keys="test-key", engines="a,b", _env_file=None)


@pytest.fixture
def now():
    return NOW
