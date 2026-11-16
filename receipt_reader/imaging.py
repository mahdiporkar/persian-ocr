"""Image loading, validation, preprocessing and printed-strip detection."""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Optional

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps

try:  # optional HEIC support
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
    HEIC_AVAILABLE = True
except Exception:  # pragma: no cover - depends on optional package
    HEIC_AVAILABLE = False


class ImageError(Exception):
    def __init__(self, code: str, detail: str, status: int = 422):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.status = status


_MAGIC = [
    (b"\xff\xd8\xff", "JPEG"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
]

EDITING_SOFTWARE = ("photoshop", "gimp", "snapseed", "picsart", "pixlr", "lightroom", "canva",
                    "affinity", "paint.net", "fotor", "photoroom", "inshot", "remini")


def sniff_format(data: bytes) -> Optional[str]:
    for magic, fmt in _MAGIC:
        if data.startswith(magic):
            return fmt
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    if data[4:8] == b"ftyp" and data[8:12] in (b"heic", b"heix", b"mif1", b"msf1", b"heim", b"heis", b"hevc"):
        return "HEIC"
    return None


@dataclass
class LoadedImage:
    pil: Image.Image            # oriented RGB image (EXIF transpose applied)
    format: str
    sha256: str
    phash: str
    kind: str                   # photo | screenshot | unknown
    flags: list[str] = field(default_factory=list)
    exif_software: Optional[str] = None


def load_image(data: bytes, *, max_bytes: int, min_side: int, max_pixels: int,
               allow_heic: bool = True) -> LoadedImage:
    if len(data) == 0:
        raise ImageError("EMPTY_FILE", "The uploaded file is empty.", 400)
    if len(data) > max_bytes:
        raise ImageError("FILE_TOO_LARGE", f"File is larger than {max_bytes // (1024 * 1024)} MB.", 413)
    fmt = sniff_format(data)
    if fmt is None:
        raise ImageError("UNSUPPORTED_TYPE", "Only JPG, PNG, WEBP (and HEIC when enabled) images are accepted.", 415)
    if fmt == "HEIC" and not (allow_heic and HEIC_AVAILABLE):
        raise ImageError("HEIC_NOT_SUPPORTED", "HEIC is not enabled on this server (install pillow-heif).", 415)

    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except Image.DecompressionBombError as e:
        raise ImageError("IMAGE_TOO_LARGE", str(e), 413) from e
    except Exception as e:
        raise ImageError("CORRUPT_IMAGE", "The file could not be decoded as an image.", 422) from e
    if (img.format or "").upper() not in ("JPEG", "PNG", "WEBP", "HEIF", "HEIC", "MPO"):
        raise ImageError("UNSUPPORTED_TYPE", f"Decoded format {img.format} is not accepted.", 415)

    exif = {}
    try:
        exif = dict(img.getexif())
    except Exception:
        pass
    software = str(exif.get(0x0131, "") or "").strip() or None
    make = exif.get(0x010F)
    model = exif.get(0x0110)

    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB",):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
    w, h = img.size
    if min(w, h) < min_side:
        raise ImageError("IMAGE_TOO_SMALL", f"Image is {w}x{h}; the shorter side must be at least {min_side}px.", 422)

    flags: list[str] = []
    if software and any(s in software.lower() for s in EDITING_SOFTWARE):
        flags.append("EDITING_SOFTWARE_IN_EXIF")
    kind = classify_kind(img, fmt, bool(make or model))
    if kind == "photo" and _looks_like_photo_of_screen(img):
        flags.append("PHOTO_OF_SCREEN")
    if kind == "screenshot" and _has_nested_screenshot_border(img):
        flags.append("SCREENSHOT_OF_SCREENSHOT")

    return LoadedImage(
        pil=img,
        format=fmt,
        sha256=hashlib.sha256(data).hexdigest(),
        phash=str(imagehash.phash(img)),
        kind=kind,
        flags=flags,
        exif_software=software,
    )


def classify_kind(img: Image.Image, fmt: str, has_camera_exif: bool) -> str:
    """Screenshot vs camera photo. Screenshots have large perfectly flat areas."""
    if has_camera_exif:
        return "photo"
    small = np.asarray(img.convert("RGB").resize((256, int(256 * img.height / img.width) or 1)))
    flat = (np.abs(np.diff(small.astype(int), axis=1)).sum(axis=2) == 0).mean()
    if flat > 0.55:
        return "screenshot"
    if flat < 0.15:
        return "photo"
    return "screenshot" if fmt == "PNG" else "unknown"


def _looks_like_photo_of_screen(img: Image.Image) -> bool:
    """Moiré from photographing a display shows up as strong periodic high-frequency energy."""
    g = np.asarray(img.convert("L").resize((512, 512)), dtype=np.float32)
    f = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
    c = 256
    hf = f.copy()
    hf[c - 64:c + 64, c - 64:c + 64] = 0
    peak = hf.max() / (hf.mean() + 1e-6)
    return bool(peak > 400)


def _has_nested_screenshot_border(img: Image.Image) -> bool:
    """A screenshot whose content is itself a smaller screenshot (status bar inside status bar)."""
    g = np.asarray(img.convert("L"))
    h = g.shape[0]
    rows = (np.abs(np.diff(g.astype(int), axis=1)) == 0).mean(axis=1)
    # count long uniform horizontal bands far from the top (a second status bar)
    bands = np.where(rows > 0.995)[0]
    if len(bands) == 0:
        return False
    gaps = np.split(bands, np.where(np.diff(bands) != 1)[0] + 1)
    mid_bands = [b for b in gaps if len(b) > h * 0.03 and 0.08 * h < b[0] < 0.5 * h]
    return len(mid_bands) >= 2


# ---------------------------------------------------------------- preprocessing

def to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def crop_document(bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Find the paper sheet and warp it flat. Returns (image, cropped?)."""
    h, w = bgr.shape[:2]
    scale = 800 / max(h, w)
    small = cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale < 1 else bgr.copy()
    s = scale if scale < 1 else 1.0
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = small.shape[0] * small.shape[1]
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(c)
        if area < 0.30 * area_img or area > 0.98 * area_img:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4:
            continue
        quad = _order_quad(approx) / s
        tl, tr, br, bl = quad
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        if width < 200 or height < 200:
            continue
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(quad, dst)
        return cv2.warpPerspective(bgr, M, (width, height), borderValue=(255, 255, 255)), True
    return bgr, False


def estimate_skew(gray: np.ndarray) -> float:
    """Small-angle skew estimate from text-line orientation (degrees)."""
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    lines = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)))
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for c in contours:
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        if max(rw, rh) < 80 or min(rw, rh) < 5:
            continue
        if rw < rh:  # OpenCV reports a near-horizontal box with swapped sides as ~±90°
            ang = ang - 90
        while ang > 45:
            ang -= 90
        while ang <= -45:
            ang += 90
        if abs(ang) <= 15:
            angles.append(ang)
    if len(angles) < 3:
        return 0.0
    return float(np.median(angles))


def text_direction(gray: np.ndarray) -> str:
    """'horizontal' or 'vertical' text lines (photos taken sideways), from the shape of
    the blobs that text lines form after smearing ink along both axes."""
    h, w = gray.shape[:2]
    s = 1000 / max(h, w)
    small = cv2.resize(gray, (int(w * s), int(h * s))) if s < 1 else gray
    thr = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    score = {}
    for name, k in (("horizontal", (15, 1)), ("vertical", (1, 15))):
        blobs = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, k))
        n, _, st, _ = cv2.connectedComponentsWithStats(blobs, 8)
        long_side = st[1:, 2] if name == "horizontal" else st[1:, 3]
        short_side = st[1:, 3] if name == "horizontal" else st[1:, 2]
        ok = (long_side >= 60) & (short_side >= 4) & (short_side <= 40) & (long_side >= 4 * short_side)
        score[name] = int(long_side[ok].sum())
    return "vertical" if score["vertical"] > 1.8 * max(score["horizontal"], 1) else "horizontal"


def _rot90(bgr: np.ndarray, cw_degrees: int) -> np.ndarray:
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return cv2.rotate(bgr, codes[cw_degrees]) if cw_degrees in codes else bgr


def _ocr_score(gray: np.ndarray) -> float:
    """How readable an orientation is: summed confidence of confidently read words."""
    import pytesseract  # optional; caller handles absence

    h, w = gray.shape[:2]
    s = 1400 / max(h, w)
    g = cv2.resize(gray, (int(w * s), int(h * s))) if s < 1 else gray
    d = pytesseract.image_to_data(g, lang="fas", config="--oem 1 --psm 6",
                                  output_type=pytesseract.Output.DICT)
    return float(sum(float(c) for t, c in zip(d["text"], d["conf"])
                     if t and len(t.strip()) >= 2 and float(c) >= 60))


def detect_orientation(bgr: np.ndarray) -> int:
    """Clockwise rotation (0/90/180/270) that makes the text upright. Lines running
    vertically -> 90 or 270; which one (and 0 vs 180) is decided by which reads better.
    Returns 0 when it cannot be decided (no local OCR installed)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if text_direction(gray) == "horizontal":
        candidates = (0, 180)
    else:
        candidates = (90, 270)
    try:
        scores = {c: _ocr_score(cv2.cvtColor(_rot90(bgr, c), cv2.COLOR_BGR2GRAY)) for c in candidates}
    except Exception:
        return 0
    best = max(scores, key=scores.get)
    other = min(scores, key=scores.get)
    if candidates == (0, 180) and not scores[180] > 2 * scores[0] + 50:
        return 0  # upright unless upside-down clearly reads better
    return best if scores[best] > scores[other] else 0


def rotate(bgr: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.3:
        return bgr
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def upscale(bgr: np.ndarray, min_width: int = 1600) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w >= min_width:
        return bgr
    f = min(3.0, min_width / w)
    return cv2.resize(bgr, (int(w * f), int(h * f)), interpolation=cv2.INTER_CUBIC)


def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide out uneven lighting (shadows across a photographed sheet): estimate the paper
    background with a closing larger than any glyph, then normalise every pixel by it.
    Without this, a shadow over half the page makes global (Otsu) thresholding — used by
    Tesseract and by the skew estimate — mark the whole shadow as ink."""
    h, w = gray.shape[:2]
    k = max(15, (min(h, w) // 40) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (k, k), 0)
    return cv2.divide(gray, np.maximum(bg, 1), scale=255)


def enhance_for_ocr(bgr: np.ndarray, flatten: bool = False) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if flatten:
        gray = flatten_illumination(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=7)
    return gray


def quality_flags(bgr: np.ndarray) -> list[str]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    flags = []
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    if sharp < 40:
        flags.append("BLURRY")
    m = gray.mean()
    if m < 60 or m > 248:
        flags.append("BAD_EXPOSURE")
    return flags


@dataclass
class Prepared:
    oriented: Image.Image          # original orientation-fixed image (for the VLM)
    document: np.ndarray           # cropped + deskewed BGR
    ocr_gray: np.ndarray           # enhanced grayscale for local OCR
    strip_box: Optional[list[int]]  # [x, y, w, h] in `document` coordinates
    strip_gray: Optional[np.ndarray]
    cropped: bool
    skew: float
    quality: list[str]
    strip_origin: Optional[tuple[int, int]] = None  # top-left of the strip crop (before 2x zoom)
    rotation: int = 0  # clockwise degrees applied to make the text upright (0/90/180/270)


def prepare(loaded: LoadedImage) -> Prepared:
    bgr = to_cv(loaded.pil)
    oriented = loaded.pil
    turn = 0
    if loaded.kind != "screenshot":  # photos of paper are often taken sideways
        turn = detect_orientation(bgr)
        if turn:
            bgr = _rot90(bgr, turn)
            oriented = loaded.pil.rotate(-turn, expand=True)  # PIL rotates counter-clockwise
    cropped = False
    if loaded.kind != "screenshot":
        bgr, cropped = crop_document(bgr)
    skew = 0.0
    if loaded.kind != "screenshot":  # screenshots are never skewed
        skew = estimate_skew(flatten_illumination(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
        bgr = rotate(bgr, skew)
    bgr = upscale(bgr)
    gray = enhance_for_ocr(bgr, flatten=loaded.kind != "screenshot")
    box = find_printed_strip(gray) if loaded.kind != "screenshot" else None
    strip = None
    origin = None
    if box:
        x, y, w, h = box
        pad = 10
        y0, y1 = max(0, y - pad), min(gray.shape[0], y + h + pad)
        x0, x1 = max(0, x - pad), min(gray.shape[1], x + w + pad)
        strip = cv2.resize(gray[y0:y1, x0:x1], None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        origin = (x0, y0)
    return Prepared(oriented=oriented, document=bgr, ocr_gray=gray, strip_box=box, strip_gray=strip,
                    strip_origin=origin, rotation=turn,
                    cropped=cropped, skew=skew, quality=quality_flags(bgr))


# ------------------------------------------------------------ printed strip

def remove_rules(thr: np.ndarray) -> np.ndarray:
    """Erase long horizontal/vertical ruling lines and box borders (form lines, stamp frames)."""
    H, W = thr.shape[:2]
    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, W // 8), 1)))
    vert = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, H // 8))))
    rules = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(thr, cv2.bitwise_not(rules))


def find_boxes(gray: np.ndarray) -> list[list[int]]:
    """Rectangular frames (stamps / teller boxes) in the lower half of the page."""
    H, W = gray.shape[:2]
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, W // 8), 1)))
    vert = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, H // 12))))
    frame = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rect_fill = cv2.contourArea(c) / float(w * h + 1)
        if (w > W * 0.25 and h > H * 0.08 and y + h / 2 > H * 0.45 and w * h < 0.6 * W * H
                and rect_fill > 0.85):
            out.append([x, y, w, h])
    # drop duplicates (inner/outer edge of the same frame)
    out.sort(key=lambda b: b[2] * b[3])
    uniq: list[list[int]] = []
    for b in out:
        if not any(abs(b[0] - u[0]) < 15 and abs(b[1] - u[1]) < 15 for u in uniq):
            uniq.append(b)
    return uniq


def text_lines(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Rough text-line boxes via horizontal smearing."""
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    thr = remove_rules(thr)
    k = max(15, gray.shape[1] // 60)
    smear = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 3)))
    contours, _ = cv2.findContours(smear, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > gray.shape[1] * 0.08 and 8 <= h <= gray.shape[0] * 0.08:
            boxes.append((x, y, w, h))
    return sorted(boxes, key=lambda b: b[1])


def _line_regularity(gray: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """0..1, high for machine print: uniform glyph heights and a straight baseline."""
    x, y, w, h = box
    roi = gray[y:y + h, x:x + w]
    thr = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    n, _, stats, _ = cv2.connectedComponentsWithStats(thr, 8)
    hs = [s[3] for s in stats[1:] if s[4] > 6]
    bottoms = [s[1] + s[3] for s in stats[1:] if s[4] > 6]
    if len(hs) < 4:
        return 0.0
    cv_h = np.std(hs) / (np.mean(hs) + 1e-6)
    cv_b = np.std(bottoms) / (h + 1e-6)
    return float(max(0.0, 1.0 - cv_h) * max(0.0, 1.0 - 2 * cv_b))


def find_printed_strip(gray: np.ndarray) -> Optional[list[int]]:
    """Locate the teller's machine-printed strip/stamp, usually in the lower part of a paper form.

    1. A rectangular frame in the lower half that contains at least two text lines
       (teller stamps and strips are very often boxed).
    2. Otherwise: the longest run of consecutive lines in the lower half with
       regular glyph geometry (machine print has uniform digit heights and a
       straight baseline; handwriting does not), widened to the full rows.
    """
    H, W = gray.shape[:2]
    lines = text_lines(gray)
    for x, y, w, h in find_boxes(gray):
        inside = [b for b in lines if b[1] >= y - 5 and b[1] + b[3] <= y + h + 5 and b[0] >= x - 5
                  and b[0] + b[2] <= x + w + 5]
        if len(inside) >= 2:
            return [int(x), int(y), int(w), int(h)]

    lower = [b for b in lines if b[1] > H * 0.45]
    if len(lower) < 2:
        return None
    reg = [_line_regularity(gray, b) for b in lower]
    regular = [b for b, r in zip(lower, reg) if r >= 0.55]
    if len(regular) < 2:
        return None
    regular.sort(key=lambda b: b[1])
    runs: list[list[tuple]] = [[regular[0]]]
    for b in regular[1:]:
        prev = runs[-1][-1]
        if b[1] - (prev[1] + prev[3]) < 3.5 * max(prev[3], b[3]):
            runs[-1].append(b)
        else:
            runs.append([b])
    run = max(runs, key=len)
    if len(run) < 2:
        return None
    y0 = min(b[1] for b in run)
    y1 = max(b[1] + b[3] for b in run)
    lh = int(np.median([b[3] for b in run]))
    y0, y1 = max(0, y0 - 2 * lh), min(H, y1 + 2 * lh)
    rows = [b for b in lines if y0 <= b[1] + b[3] / 2 <= y1]
    x0 = min(b[0] for b in rows)
    x1 = max(b[0] + b[2] for b in rows)
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def encode_jpeg(img: Image.Image, max_side: int, quality: int = 92) -> bytes:
    im = img
    if max(im.size) > max_side:
        f = max_side / max(im.size)
        im = im.resize((int(im.width * f), int(im.height * f)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
