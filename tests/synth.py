"""Synthetic receipt images — for unit/integration tests and the demo ONLY.

Accuracy must never be reported from these images; see eval/README.md.
"""
from __future__ import annotations

import io
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except Exception:  # pragma: no cover
    arabic_reshaper = None

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]
MONO_CANDIDATES = [
    r"C:\Windows\Fonts\cour.ttf",
    r"C:\Windows\Fonts\consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
HAND_CANDIDATES = [
    r"C:\Windows\Fonts\tahomabd.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _first(paths) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def fonts_available() -> bool:
    return arabic_reshaper is not None and _first(FONT_CANDIDATES) is not None


def _font(paths, size):
    p = _first(paths) or _first(FONT_CANDIDATES)
    return ImageFont.truetype(p, size, layout_engine=ImageFont.Layout.BASIC)


def fa(text: str) -> str:
    """Shape + reorder Persian for drawing with a basic layout engine."""
    return get_display(arabic_reshaper.reshape(text))


def draw_rtl(d: ImageDraw.ImageDraw, right_x: int, y: int, text: str, font, fill):
    t = fa(text)
    w = d.textlength(t, font=font)
    d.text((right_x - w, y), t, font=font, fill=fill)


@dataclass
class SatnaSpec:
    printed_amount_rial: int = 4_500_000_000
    handwritten_amount_rial: int = 4_500_000_000
    words: str = "چهار میلیارد و پانصد میلیون ریال"
    dest_iban: str = ""
    dest_name: str = "علی کریمی"
    payer_name: str = "رضا احمدی"
    tracking: str = "140508241234567"
    date: str = "1405/08/24"
    time: str = "11:42"
    bank_header: str = "بانک ملت"
    strip: bool = True
    seed: int = 7
    extras: dict = field(default_factory=dict)


def render_satna_form(spec: SatnaSpec) -> Image.Image:
    """A clean (scanner-like) SATNA form with handwritten-style fields and a printed teller strip."""
    W, H = 1700, 1250
    img = Image.new("RGB", (W, H), (250, 249, 244))
    d = ImageDraw.Draw(img)
    f_head = _font(FONT_CANDIDATES, 46)
    f_lbl = _font(FONT_CANDIDATES, 34)
    f_hand = _font(HAND_CANDIDATES, 40)
    f_mono = _font(MONO_CANDIDATES, 34)
    ink = (25, 40, 140)
    black = (20, 20, 20)
    R = W - 80

    draw_rtl(d, R, 50, spec.bank_header, f_head, black)
    draw_rtl(d, R - 600, 55, "فرم درخواست انتقال وجه ساتنا", f_lbl, black)
    d.line((80, 130, W - 80, 130), fill=black, width=3)

    rows = [
        ("مبلغ به ریال:", f"{spec.handwritten_amount_rial:,}"),
        ("مبلغ به حروف:", spec.words),
        ("به حساب شبا:", spec.dest_iban),
        ("نام ذینفع:", spec.dest_name),
        ("نام واریز کننده:", spec.payer_name),
    ]
    y = 170
    for label, value in rows:
        draw_rtl(d, R, y, label, f_lbl, black)
        lw = d.textlength(fa(label), font=f_lbl)
        vx = R - lw - 30
        if value and any("\u0600" <= ch <= "\u06ff" for ch in value):
            draw_rtl(d, int(vx), y - 2, value, f_hand, ink)
        else:
            vw = d.textlength(value, font=f_hand)
            d.text((vx - vw, y - 2), value, font=f_hand, fill=ink)
        d.line((120, y + 52, vx, y + 52), fill=(160, 160, 160), width=1)
        y += 95

    if spec.strip:
        top = 720
        d.rectangle((110, top, W - 110, top + 420), outline=(90, 30, 110), width=3)
        purple = (70, 20, 90)
        sy = top + 25
        strip_rows = [
            f"SATNA  {fa('ساتنا')}",
            f"{spec.tracking}  :{fa('شماره پیگیری')}",
            f"{spec.printed_amount_rial:,} {fa('ریال')}  :{fa('مبلغ')}",
            f"{spec.date} {spec.time}  :{fa('تاریخ')}",
            f"{spec.dest_iban}  :{fa('شبا مقصد')}",
        ]
        for row in strip_rows:
            w = d.textlength(row, font=f_mono)
            d.text((W - 150 - w, sy), row, font=f_mono, fill=purple)
            sy += 76
    return img


def photograph(img: Image.Image, seed: int = 7, angle: float = 2.0) -> Image.Image:
    """Make a clean render look like a phone photo: background, rotation, blur, noise, JPEG."""
    rnd = random.Random(seed)
    bg = Image.new("RGB", (img.width + 300, img.height + 300), (70, 62, 55))
    bg.paste(img, (150, 150))
    bg = bg.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(70, 62, 55))
    arr = np.asarray(bg).astype(np.float32)
    yy = np.linspace(0.85, 1.05, arr.shape[0])[:, None, None]
    arr = arr * yy
    rng = np.random.default_rng(seed)
    arr += rng.normal(0, 4, arr.shape)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=88)
    return Image.open(io.BytesIO(buf.getvalue()))


def render_screenshot(amount_rial=450_000_000, dest_iban="", tracking="987654321012",
                      date="1405/08/24", time="09:15", status="موفق", kind="پایا") -> Image.Image:
    W, H = 1080, 1920
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 90), fill=(0, 92, 175))
    d.rectangle((0, 90, W, 260), fill=(240, 244, 250))
    f_big = _font(FONT_CANDIDATES, 52)
    f = _font(FONT_CANDIDATES, 40)
    color = (0, 110, 50) if status == "موفق" else (170, 0, 0)
    draw_rtl(d, W - 60, 110, f"انتقال {kind}", f_big, (20, 20, 20))
    draw_rtl(d, W - 60, 180, f"وضعیت: {status}", f, color)
    rows = [
        ("مبلغ:", f"{amount_rial:,} ریال"),
        ("شبای مقصد:", dest_iban),
        ("نام ذینفع:", "مریم حسینی"),
        ("شماره پیگیری:", tracking),
        ("تاریخ:", f"{date} {time}"),
        ("کارمزد:", "5,000 ریال"),
    ]
    y = 340
    for label, value in rows:
        draw_rtl(d, W - 60, y, label, f, (90, 90, 90))
        if any("\u0600" <= ch <= "\u06ff" for ch in value):
            t = fa(value)
        else:
            t = value
        d.text((60, y), t, font=f, fill=(10, 10, 10))
        d.line((60, y + 80, W - 60, y + 80), fill=(225, 225, 225), width=2)
        y += 130
    return img


def to_bytes(img: Image.Image, fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=90) if fmt == "JPEG" else img.save(buf, format=fmt)
    return buf.getvalue()
