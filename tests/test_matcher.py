

from .conftest import IBAN_MELLAT, IBAN_MELLI
from .helpers import C, good_printed, line, run


def test_clean_match(settings, now):
    f, matches, best, route, q, warnings = run(
        good_printed(IBAN_MELLAT), settings, now,
        lines=[line("L-17", IBAN_MELLAT, 450_000_000), line("L-18", IBAN_MELLI, 450_000_000)])
    assert route == "match"
    assert best == "L-17"
    assert matches[0].line_id == "L-17"
    assert {"AMOUNT_EXACT", "IBAN_FULL_MATCH"} <= set(matches[0].reasons)
    assert "IBAN_MISMATCH" in matches[1].reasons
    assert q is None


def test_partial_payment(settings, now):
    f, matches, best, route, *_ = run(good_printed(IBAN_MELLAT, amount=2_000_000_000), settings, now,
                                      lines=[line("L-1", IBAN_MELLAT, 450_000_000, remaining=450_000_000)])
    assert "AMOUNT_PARTIAL" in matches[0].reasons
    assert best == "L-1" and route == "match"


def test_amount_above_remaining(settings, now):
    _, matches, best, route, _, warnings = run(good_printed(IBAN_MELLAT), settings, now,
                                               lines=[line("L-1", IBAN_MELLAT, 450_000_000, remaining=100_000_000)])
    assert "AMOUNT_OVER_REMAINING" in matches[0].reasons
    assert best is None and route == "manager_review" and "NO_MATCHING_LINE" in warnings


def test_remaining_exact_after_earlier_partial(settings, now):
    _, matches, best, route, *_ = run(good_printed(IBAN_MELLAT, amount=2_500_000_000), settings, now,
                                      lines=[line("L-1", IBAN_MELLAT, 450_000_000, remaining=250_000_000)])
    assert "AMOUNT_EXACT" in matches[0].reasons and route == "match"


def test_ambiguous_match_not_picked(settings, now):
    _, matches, best, route, _, warnings = run(
        good_printed(IBAN_MELLAT), settings, now,
        lines=[line("L-1", IBAN_MELLAT, 450_000_000), line("L-2", IBAN_MELLAT, 450_000_000)])
    assert best is None
    assert "AMBIGUOUS_MATCH" in warnings
    assert route == "manager_review"


def test_iban_mismatch(settings, now):
    _, matches, best, route, _, warnings = run(good_printed(IBAN_MELLAT), settings, now,
                                               lines=[line("L-1", IBAN_MELLI, 450_000_000)])
    assert "IBAN_MISMATCH" in warnings and route == "manager_review"


def test_wrong_currency_read_both_ways(settings, now):
    """No unit on the document: the matcher tests both readings and says which one matched."""
    cands = [c for c in good_printed(IBAN_MELLAT, amount=450_000_000) if c.field != "amount"]
    cands += [C("amount", 450_000_000, engine="a"), C("amount", 450_000_000, engine="b")]
    f, matches, best, route, _, warnings = run(cands, settings, now,
                                               lines=[line("L-1", IBAN_MELLAT, 450_000_000)])
    assert f.currency == "unknown"
    assert matches[0].currency_assumed == "toman"
    assert "CURRENCY_ASSUMED_TOMAN" in matches[0].reasons
    assert "UNKNOWN_CURRENCY" in warnings
    assert route == "manager_review"


def test_rial_toman_confusion_does_not_match(settings, now):
    """Receipt says 450,000,000 RIAL; line expects 450,000,000 TOMAN -> partial, not exact."""
    _, matches, *_ = run(good_printed(IBAN_MELLAT, amount=450_000_000), settings, now,
                         lines=[line("L-1", IBAN_MELLAT, 450_000_000)])
    assert "AMOUNT_EXACT" not in matches[0].reasons


def test_medium_confidence_goes_to_sender(settings, now):
    cands = [c for c in good_printed(IBAN_MELLAT) if not (c.field == "amount" and c.engine == "b")]
    f, _, best, route, q, _ = run(cands, settings, now, lines=[line("L-17", IBAN_MELLAT, 450_000_000)])
    assert f.amount_rial.confidence == "medium"
    assert route == "confirm_with_sender" and best == "L-17"
    assert q == "این فیش را ۴۵۰٬۰۰۰٬۰۰۰ تومان به حساب …۸۵۵۰ خواندم. درست است؟"


def test_partial_iban_cannot_auto_match(settings, now):
    cands = [c for c in good_printed(IBAN_MELLAT) if c.field != "iban"]
    cands += [C("iban", "...8550", role="dest", source="digital")]
    _, matches, best, route, q, _ = run(cands, settings, now, lines=[line("L-1", IBAN_MELLAT, 450_000_000)])
    assert "IBAN_LAST_DIGITS_MATCH" in matches[0].reasons
    assert route == "confirm_with_sender" and q


def test_invalid_iban_never_exact(settings, now):
    bad = IBAN_MELLAT[:-1] + str((int(IBAN_MELLAT[-1]) + 1) % 10)
    cands = [c for c in good_printed(IBAN_MELLAT) if c.field != "iban"] + [C("iban", bad, role="dest")]
    _, matches, _, route, *_ = run(cands, settings, now, lines=[line("L-1", bad, 450_000_000)])
    assert "IBAN_FULL_MATCH" not in matches[0].reasons
    assert route != "match"


def test_editing_flag_forces_review(settings, now):
    _, _, _, route, _, warnings = run(good_printed(IBAN_MELLAT), settings, now,
                                      lines=[line("L-17", IBAN_MELLAT, 450_000_000)], flags=["POSSIBLE_EDITING"])
    assert route == "manager_review" and "POSSIBLE_EDITING" in warnings


def test_no_lines(settings, now):
    _, _, _, route, _, warnings = run(good_printed(IBAN_MELLAT), settings, now)
    assert route == "manager_review" and "NO_EXPECTED_LINES" in warnings


def test_name_reason(settings, now):
    cands = good_printed(IBAN_MELLAT) + [C("dest_holder_name", "علي كريمي")]
    _, matches, *_ = run(cands, settings, now, lines=[line("L-1", IBAN_MELLAT, 450_000_000, name="علی کریمی")])
    assert "NAME_MATCH" in matches[0].reasons


def test_two_fitting_lines_different_scores_not_auto(settings, now):
    _, matches, best, route, *_ = run(
        good_printed(IBAN_MELLAT, amount=1_000_000_000), settings, now,
        lines=[line("L-1", IBAN_MELLAT, 100_000_000), line("L-2", IBAN_MELLAT, 450_000_000)])
    # L-1 exact, L-2 partial: L-1 is better, but two lines fit so no automatic match
    assert best == "L-1"
    assert route == "confirm_with_sender"
