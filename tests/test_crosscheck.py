from .conftest import IBAN_MELLAT, IBAN_MELLI
from .helpers import C, good_printed, run


def test_two_agreeing_printed_reads_are_high(settings, now):
    f, *_ = run(good_printed(IBAN_MELLAT), settings, now)
    assert f.amount_rial.value == 4_500_000_000
    assert f.amount_rial.confidence == "high"
    assert f.amount_toman == 450_000_000
    assert f.currency == "rial"
    assert f.dest_iban.confidence == "high" and f.dest_iban.bank == "Mellat"
    assert f.paid_at.confidence == "high" and f.paid_at.iso == "2026-11-15T11:42:00+03:30"
    assert f.tracking_number.confidence == "high"


def test_single_printed_amount_is_only_medium(settings, now):
    f, *_ = run([C("amount", 450_000_000, currency="rial")], settings, now)
    assert f.amount_rial.confidence == "medium"


def test_words_corroborate_single_read(settings, now):
    f, *_ = run([C("amount", 450_000_000, currency="rial"),
                 C("amount", 450_000_000, source="words", currency="rial")], settings, now)
    assert f.amount_rial.confidence == "high"


def test_real_failure_handwritten_vs_words_is_conflict(settings, now):
    """Handwritten 1,500,000,000 but words say «چهارصد و پنجاه میلیون»: never passed on."""
    f, _, _, route, _, warnings = run([
        C("amount", 1_500_000_000, source="handwritten", currency="rial"),
        C("amount", 450_000_000, source="words", currency="rial"),
    ], settings, now)
    assert f.amount_rial.confidence == "conflict"
    assert f.amount_rial.value is None
    assert len(f.amount_rial.candidates) == 2  # every candidate kept
    assert "AMOUNT_CONFLICT" in warnings
    assert route == "manager_review"


def test_printed_beats_handwritten(settings, now):
    cands = good_printed(IBAN_MELLAT, amount=4_500_000_000)
    cands.append(C("amount", 15_000_000_000, source="handwritten", engine="b"))
    f, _, _, _, _, warnings = run(cands, settings, now)
    assert f.amount_rial.value == 4_500_000_000
    assert f.amount_rial.confidence == "high"
    assert "HANDWRITTEN_AMOUNT_DISAGREES" in warnings and "AMOUNT_CONFLICT" in warnings


def test_two_printed_amounts_disagree(settings, now):
    f, _, _, route, _, warnings = run([C("amount", 450_000_000, engine="a"),
                                       C("amount", 460_000_000, engine="b")], settings, now)
    assert f.amount_rial.confidence == "conflict"
    assert "MULTIPLE_AMOUNTS" in warnings
    assert route == "manager_review"


def test_handwriting_only_is_low_or_capped_medium(settings, now):
    f, *_ = run([C("amount", 450_000_000, source="handwritten")], settings, now)
    assert f.amount_rial.confidence == "low"
    f, *_ = run([C("amount", 450_000_000, source="handwritten"),
                 C("amount", 450_000_000, source="words")], settings, now)
    assert f.amount_rial.confidence == "medium"


def test_toman_document(settings, now):
    f, *_ = run([C("amount", 45_000_000, currency="toman"), C("amount", 45_000_000, engine="b", currency="toman")],
                settings, now)
    assert f.currency == "toman"
    assert f.amount_rial.value == 450_000_000
    assert f.amount_toman == 45_000_000


def test_unknown_currency(settings, now):
    f, _, _, _, _, warnings = run([C("amount", 450_000_000)], settings, now)
    assert f.currency == "unknown"
    assert "UNKNOWN_CURRENCY" in warnings


def test_amount_toman_only_when_divisible(settings, now):
    f, *_ = run([C("amount", 450_000_005, currency="rial")], settings, now)
    assert f.amount_toman is None


def test_invalid_iban_checksum(settings, now):
    bad = IBAN_MELLAT[:-1] + str((int(IBAN_MELLAT[-1]) + 1) % 10)
    f, _, _, _, _, warnings = run([C("iban", bad, role="dest")], settings, now)
    assert f.dest_iban.checksum_valid is False
    assert f.dest_iban.confidence == "low"
    assert "INVALID_IBAN" in warnings


def test_source_iban_not_taken_as_destination(settings, now):
    f, *_ = run([C("iban", IBAN_MELLI, role="source"), C("iban", IBAN_MELLAT, role="dest")], settings, now)
    assert f.dest_iban.value == IBAN_MELLAT


def test_two_destination_ibans_conflict(settings, now):
    f, _, _, _, _, warnings = run([C("iban", IBAN_MELLI, role="dest"), C("iban", IBAN_MELLAT, role="dest", engine="b")],
                                  settings, now)
    assert f.dest_iban.confidence == "conflict"
    assert "IBAN_CONFLICT" in warnings


def test_partial_iban(settings, now):
    f, _, _, _, _, warnings = run([C("iban", "...568550", role="dest", source="digital")], settings, now)
    assert f.dest_iban.partial and f.dest_iban.value == "...568550"
    assert "PARTIAL_IBAN" in warnings


def test_bank_name_mismatch_caps_confidence(settings, now):
    cands = [C("iban", IBAN_MELLAT, role="dest"), C("dest_bank_name", "017")]
    f, _, _, _, _, warnings = run(cands, settings, now)
    assert "BANK_NAME_MISMATCH" in warnings
    assert f.dest_iban.confidence == "medium"


def test_date_implausible(settings, now):
    f, _, _, _, _, warnings = run([C("paid_at", "1405/07/01 10:00")], settings, now)
    assert "DATE_IMPLAUSIBLE" in warnings and f.paid_at.confidence == "low"
    f, _, _, _, _, warnings = run([C("paid_at", "1405/08/24 23:00")], settings, now)  # future
    assert "DATE_IMPLAUSIBLE" in warnings


def test_tracking_keeps_labels(settings, now):
    f, *_ = run([C("tracking_number", "111111111", label="شماره مرجع"),
                 C("tracking_number", "222222222", label="شماره پیگیری")], settings, now)
    assert f.tracking_number.value == "222222222"
    assert f.tracking_number.label == "شماره پیگیری"
    assert {d["label"] for d in f.tracking_number.all} == {"شماره مرجع", "شماره پیگیری"}


def test_status_failed_wins(settings, now):
    f, _, _, route, _, warnings = run(good_printed(IBAN_MELLAT) + [C("document_status", "failed")], settings, now)
    assert f.document_status == "failed"
    assert "STATUS_NOT_SUCCESS" in warnings
