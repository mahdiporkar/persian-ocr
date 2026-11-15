import io

import numpy as np
import pytest
from PIL import Image

from receipt_reader.imaging import ImageError, find_printed_strip, load_image, prepare, sniff_format

from .conftest import IBAN_MELLAT
from .synth import SatnaSpec, fonts_available, photograph, render_satna_form, render_screenshot

KW = dict(max_bytes=10 * 1024 * 1024, min_side=300, max_pixels=60_000_000)


def _encode(img, fmt, **kw):
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def _noise(w=900, h=700):
    return Image.fromarray(np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8))


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_formats(fmt):
    li = load_image(_encode(_noise(), fmt), **KW)
    assert li.format == fmt and len(li.sha256) == 64 and li.phash


def test_exif_orientation_is_applied():
    img = _noise(900, 600)
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    li = load_image(_encode(img, "JPEG", exif=exif), **KW)
    assert li.pil.size == (600, 900)


def test_editing_software_flag():
    exif = Image.Exif()
    exif[0x0131] = "Adobe Photoshop 25.0"
    li = load_image(_encode(_noise(), "JPEG", exif=exif), **KW)
    assert "EDITING_SOFTWARE_IN_EXIF" in li.flags


def test_heic():
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    data = _encode(_noise(), "HEIF")
    assert sniff_format(data) == "HEIC"
    li = load_image(data, **KW)
    assert li.pil.size == (900, 700)


def test_heic_disabled():
    pytest.importorskip("pillow_heif")
    import pillow_heif
    pillow_heif.register_heif_opener()
    with pytest.raises(ImageError) as e:
        load_image(_encode(_noise(), "HEIF"), allow_heic=False, **KW)
    assert e.value.code == "HEIC_NOT_SUPPORTED"


@pytest.mark.parametrize("data,code", [
    (b"", "EMPTY_FILE"),
    (b"GIF89a" + b"0" * 100, "UNSUPPORTED_TYPE"),
    (b"\xff\xd8\xff" + b"garbage" * 50, "CORRUPT_IMAGE"),
])
def test_rejections(data, code):
    with pytest.raises(ImageError) as e:
        load_image(data, **KW)
    assert e.value.code == code


def test_too_large_bytes():
    with pytest.raises(ImageError) as e:
        load_image(_encode(_noise(), "PNG"), max_bytes=1000, min_side=300, max_pixels=60_000_000)
    assert e.value.code == "FILE_TOO_LARGE"


@pytest.mark.skipif(not fonts_available(), reason="no Persian font")
def test_screenshot_vs_photo_and_strip():
    shot = load_image(_encode(render_screenshot(dest_iban=IBAN_MELLAT), "PNG"), **KW)
    assert shot.kind == "screenshot"
    assert prepare(shot).strip_box is None  # screenshots have no teller strip

    photo = load_image(_encode(photograph(render_satna_form(SatnaSpec(dest_iban=IBAN_MELLAT))), "JPEG"), **KW)
    assert photo.kind != "screenshot"
    p = prepare(photo)
    assert p.cropped and p.strip_box is not None
    # the strip is in the lower part of the page
    assert p.strip_box[1] > p.document.shape[0] * 0.45


@pytest.mark.skipif(not fonts_available(), reason="no Persian font")
def test_no_strip_on_form_without_one():
    li = load_image(_encode(photograph(render_satna_form(SatnaSpec(dest_iban=IBAN_MELLAT, strip=False))), "JPEG"),
                    **KW)
    assert prepare(li).strip_box is None
