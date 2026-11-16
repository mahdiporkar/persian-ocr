"""EasyOCR's ambiguous «٥» (Persian zero or five) resolved from glyph shape."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from receipt_reader.glyphs import align, resolve_ambiguous_five, resolved_text
from receipt_reader.normalize import to_ascii_digits

from .synth import _first, FONT_CANDIDATES, fonts_available


def A(token, classes):
    r = align(token, classes)
    return to_ascii_digits(r) if r is not None else None


def test_each_ambiguous_glyph_decided_by_its_class():
    # tracking number 140508230171028390 as EasyOCR reads it, against its glyph classes
    assert A("١٤٥٥٥٨٢٣٥١٧١٥٢٨٣٩٥", "DDZDZDDDZDDDZDDDDZ") == "140508230171028390"


def test_dropped_zero_restored_only_with_valid_grouping():
    # EasyOCR read "3,000,000,000" as "٣,٥٥٥,٥٥,٥٥٥" (one zero dropped by the CTC decoder)
    assert A("٣,٥٥٥,٥٥,٥٥٥", "DSZZZSZZZSZZZ") == "3,000,000,000"
    # without thousands separators a longer run is never invented
    assert A("١٥٥", "DZZZ") is None


def test_full_height_runs_are_never_expanded():
    # handwriting «٥٢» next to other strokes must not become «5552»
    assert A("٥٢", "DDDD") is None


def test_mid_height_glyph_stays_unreadable():
    assert A("١٥", "DM") is None


def test_unaligned_text_is_unreadable():
    assert A("١٥٢", "DZ") is None


def test_unreadable_without_image():
    assert resolved_text(None, [0, 0, 1, 1], "١٤٠٥") == "١٤٠?"
    assert resolved_text(None, [0, 0, 1, 1], "١٤٠٥", enabled=False) == "١٤٠٥"


def test_words_with_five_glyph_are_not_touched():
    gray = np.full((40, 200), 255, np.uint8)
    assert resolve_ambiguous_five(gray, [0, 0, 200, 40], "سلام ٥ دنیا") == "سلام ? دنیا"


@pytest.mark.skipif(not fonts_available(), reason="no Persian font")
def test_rendered_persian_digits():
    from PIL import ImageFont

    font = ImageFont.truetype(_first(FONT_CANDIDATES), 48, layout_engine=ImageFont.Layout.BASIC)
    img = Image.new("L", (900, 90), 255)
    ImageDraw.Draw(img).text((20, 10), "۱۴۰۵۰۸۲۳۰۱۷۱", font=font, fill=0)
    gray = np.array(img)
    ys, xs = np.where(gray < 128)
    box = [xs.min() - 3, ys.min() - 3, xs.max() + 3, ys.max() + 3]
    # what EasyOCR would output: every zero and five as «٥»
    out = resolve_ambiguous_five(gray, box, "١٤٥٥٥٨٢٣٥١٧١")
    assert to_ascii_digits(out) in ("140508230171", "14???823?171")  # resolved correctly, or not at all
