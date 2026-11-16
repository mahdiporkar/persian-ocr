"""Resolve EasyOCR's ambiguous «٥» from the glyph's own shape.

EasyOCR's 'fa' model outputs Arabic «٥» both for a Persian five «۵» and for the round
Persian zero «۰» (measured on real receipts). The two are easy to tell apart by size:
in Persian digit fonts the zero is a small dot or circle, the five is a full-height digit.

For each number token that contains «٥» we find the glyphs in the detection box, give
each glyph a class from its size and shape —

    D  full-height glyph (a digit other than zero, or «/»)
    Z  small and round (zero; also a dot)
    S  small and narrow (comma, dot)
    M  in between (never matched by a digit)

— and align the token's characters with that glyph sequence. A «٥» aligned with a Z
glyph is a zero, with a D glyph a five. CTC decoders (EasyOCR's) also tend to drop one
of several identical glyphs in a row ("3,000,000,000" -> "٣,٥٥٥,٥٥,٥٥٥"), so a run of
«٥» may stand for MORE identical glyphs of one class; the run is then expanded.

Strictly conservative: the token is resolved only if exactly one alignment exists with
exactly one glyph group of the detection. Anything else becomes «?» (unreadable), which
the parser drops whole.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

AMBIGUOUS = "٥"
UNREADABLE = "?"
SMALL_MAX_RATIO = 0.66   # Z/S: at most this fraction of the full digit height
FULL_MIN_RATIO = 0.85    # D: at least this fraction
ROUND_MIN_WH = 0.7       # Z: width/height at least this (a comma is narrow)
MAX_RUN_EXTRA = 2        # a run of «٥» may hide up to this many dropped identical glyphs

_DIGITS = set("0123456789٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹")
_ZEROS = set("0٠۰")
_COMMAS = set(",،٬")
_TOKEN_CHARS = _DIGITS | _COMMAS | set("./:-") | set("IRirXx*")


# --------------------------------------------------------------------------- glyphs
def _columns(binary: np.ndarray) -> list[list[int]]:
    """Connected components merged into glyph columns (a piece mostly inside another's
    x-range belongs to it, e.g. the two dots of «:»). [x0, y0, x1, y1], left to right."""
    n, _, st, _ = cv2.connectedComponentsWithStats(binary, 8)
    H, W = binary.shape
    boxes = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if a < 4 or (h < 0.06 * H and w < 0.06 * H):
            continue  # specks
        if h > 0.98 * H or w > 0.9 * W:
            continue  # frame / underline
        boxes.append([int(x), int(y), int(x + w), int(y + h)])
    boxes.sort()
    cols: list[list[int]] = []
    for b in boxes:
        if cols:
            c = cols[-1]
            overlap = min(c[2], b[2]) - max(c[0], b[0])
            if overlap > 0.8 * min(c[2] - c[0], b[2] - b[0]):
                cols[-1] = [min(c[0], b[0]), min(c[1], b[1]), max(c[2], b[2]), max(c[3], b[3])]
                continue
        cols.append(b)
    return cols


def _group_by_gaps(cols: list[list[int]]) -> list[list[list[int]]]:
    """Split glyph columns into words at gaps wider than ~half a small glyph."""
    if not cols:
        return []
    hs = sorted(c[3] - c[1] for c in cols)
    gap = 0.6 * hs[len(hs) // 2]
    groups = [[cols[0]]]
    for c in cols[1:]:
        if c[0] - groups[-1][-1][2] > gap:
            groups.append([c])
        else:
            groups[-1].append(c)
    return groups


def classify(cols: list[list[int]]) -> str:
    """One class letter (D/Z/S/M) per glyph column, relative to the group's digit height."""
    hs = [c[3] - c[1] for c in cols]
    if not hs:
        return ""
    tall = [h for h in hs if h >= 0.75 * max(hs)]
    full = float(np.median(tall))
    out = []
    for c, h in zip(cols, hs):
        w = c[2] - c[0]
        r = h / full
        if r >= FULL_MIN_RATIO:
            out.append("D")
        elif r <= SMALL_MAX_RATIO:
            out.append("Z" if w >= ROUND_MIN_WH * h else "S")
        else:
            out.append("M")
    return "".join(out)


# --------------------------------------------------------------------------- alignment
def _allowed(ch: str) -> set[str]:
    if ch in _ZEROS:
        return {"Z"}
    if ch in _DIGITS:
        return {"D"}
    if ch in _COMMAS:
        return {"S"}
    if ch == ".":
        return {"S", "Z"}
    if ch == "/":
        return {"D", "M"}
    return {"D", "Z", "S", "M"}  # ':', '-', IBAN/mask letters


def align(token: str, classes: str) -> Optional[str]:
    """Resolve every «٥» of `token` against glyph classes. Returns the resolved token
    (possibly longer, when a run of «٥» hid dropped identical glyphs) or None when there is
    not exactly one alignment."""
    # split the token into units: a run of «٥» or a single other character
    units: list[tuple[str, int]] = []
    for ch in token:
        if ch == AMBIGUOUS and units and units[-1][0] == AMBIGUOUS:
            units[-1] = (AMBIGUOUS, units[-1][1] + 1)
        else:
            units.append((ch, 1))

    @lru_cache(maxsize=None)
    def solve(u: int, g: int) -> frozenset[tuple[str, bool]]:
        """All (resolved suffix, expanded?) for units[u:] against classes[g:] (capped)."""
        if u == len(units):
            return frozenset({("", False)}) if g == len(classes) else frozenset()
        ch, k = units[u]
        if ch != AMBIGUOUS:
            if g < len(classes) and classes[g] in _allowed(ch):
                return frozenset((ch + rest, ex) for rest, ex in solve(u + 1, g + 1))
            return frozenset()
        options: list[tuple[int, str, bool]] = []
        # exact length: each glyph decides its own digit
        seg = classes[g:g + k]
        if len(seg) == k and set(seg) <= {"Z", "D"}:
            options.append((k, "".join("۰" if c == "Z" else "۵" for c in seg), False))
        # longer: the decoder dropped identical glyphs. Only for zeros (by far the most
        # repeated digit), and only kept if the thousands grouping then checks out.
        for extra in range(1, MAX_RUN_EXTRA + 1):
            seg = classes[g:g + k + extra]
            if len(seg) == k + extra and set(seg) == {"Z"}:
                options.append((k + extra, "۰" * (k + extra), True))
        out: set[tuple[str, bool]] = set()
        for used, digits, expanded in options:
            for rest, ex in solve(u + 1, g + used):
                out.add((digits + rest, expanded or ex))
                if len(out) > 8:
                    return frozenset(out)
        return frozenset(out)

    valid = {r for r, expanded in solve(0, 0) if not expanded or _grouping_ok(r)}
    return valid.pop() if len(valid) == 1 else None


def _grouping_ok(token: str) -> bool:
    """Thousands grouping is consistent: 1-3 digits, then groups of exactly 3."""
    parts = [p for p in "".join("," if c in _COMMAS else c for c in token).split(",")]
    if len(parts) < 2 or not all(p and all(c in _DIGITS for c in p) for p in parts):
        return False
    return 1 <= len(parts[0]) <= 3 and all(len(p) == 3 for p in parts[1:])


# --------------------------------------------------------------------------- entry points
def resolve_ambiguous_five(gray: np.ndarray, box: list[int], text: str) -> str:
    """Return `text` with every «٥» resolved to «۰»/«۵», or «?» where that is not certain.

    gray: the image EasyOCR read; box: the detection's [x0, y0, x1, y1]."""
    if AMBIGUOUS not in text:
        return text
    tokens = text.split()
    numeric = [i for i, t in enumerate(tokens) if AMBIGUOUS in t and all(ch in _TOKEN_CHARS for ch in t)]
    if not numeric:
        return text.replace(AMBIGUOUS, UNREADABLE)
    try:
        x0, y0, x1, y1 = (int(v) for v in box)
        crop = gray[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2]
        if crop.size == 0 or crop.shape[0] < 8:
            return text.replace(AMBIGUOUS, UNREADABLE)
        _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if binary.mean() > 127:  # light text on a dark background
            binary = 255 - binary
        group_classes = [classify(g) for g in _group_by_gaps(_columns(binary))]
    except Exception:
        return text.replace(AMBIGUOUS, UNREADABLE)

    resolved = list(tokens)
    for i in numeric:
        # the visual order of tokens in a mixed RTL/LTR detection is not known, so the token
        # must align with exactly one glyph group of the detection
        hits = {r for r in (align(tokens[i], cls) for cls in group_classes) if r is not None}
        resolved[i] = hits.pop() if len(hits) == 1 else tokens[i]
    return " ".join(t.replace(AMBIGUOUS, UNREADABLE) for t in resolved)


def resolved_text(gray: Optional[np.ndarray], box, text: str, enabled: bool = True) -> str:
    """Engine helper: resolve when the image is available, else mark «٥» unreadable."""
    if not enabled:
        return text
    if gray is None:
        return text.replace(AMBIGUOUS, UNREADABLE)
    return resolve_ambiguous_five(gray, box, text)
