"""Optional second local engine: EasyOCR with its Persian ('fa') model.

Installed separately (`pip install easyocr`, pulls PyTorch CPU). Kept behind the same
interface so the eval harness can compare it against Tesseract on real receipts.
"""
from __future__ import annotations

import re
import threading

from ..config import Settings
from ..glyphs import resolved_text
from ..imaging import LoadedImage, Prepared
from ..normalize import to_ascii_digits
from ..textparse import TextLine, parse_lines
from .base import Engine, EngineOutput


_NUMERIC = re.compile(r"^[\d,.\s/:\-]+$")


def _join_rtl(tokens: list[str]) -> str:
    """Join right-to-left ordered detections, but keep runs of numeric boxes left-to-right
    (one number split over several boxes, e.g. "4,500," + "000,000")."""
    out: list[str] = []
    run: list[str] = []
    for t in tokens:
        if _NUMERIC.match(to_ascii_digits(t)):
            run.append(t)
            continue
        if run:
            out.append("".join(reversed(run)) if all("," in r or len(r) <= 3 for r in run) else " ".join(reversed(run)))
            run = []
        out.append(t)
    if run:
        out.append("".join(reversed(run)) if all("," in r or len(r) <= 3 for r in run) else " ".join(reversed(run)))
    return " ".join(out)


class EasyOCREngine(Engine):
    name = "easyocr"

    def __init__(self, settings: Settings):
        self.s = settings
        self._reader = None
        self._lock = threading.Lock()

    def available(self) -> tuple[bool, str]:
        try:
            import easyocr  # noqa: F401
        except Exception:
            return False, "easyocr not installed (optional)"
        return True, "ok"

    def _get(self):
        with self._lock:  # model load (and first-run download) happens once
            if self._reader is None:
                import easyocr

                self._reader = easyocr.Reader(["fa", "en"], gpu=False, verbose=False)
        return self._reader

    def _lines(self, img, pass_id, src_for, offset=(0, 0), scale=1.0) -> list[TextLine]:
        res = self._get().readtext(img, detail=1, paragraph=False)
        self._last_raw = [(t, round(c, 2)) for _, t, c in res]
        # group detections into lines by vertical centre
        items = []
        for box, text, conf in res:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            if self.s.easyocr_ambiguous_five:
                # Measured on real receipts: EasyOCR's 'fa' model outputs Arabic «٥» both for a
                # real 5 and for the round Persian zero «۰». Decide from the glyph's height
                # (glyphs.py); whatever cannot be decided becomes «?» and is never parsed.
                text = resolved_text(img, [min(xs), min(ys), max(xs), max(ys)], text)
            items.append((min(xs), min(ys), max(xs), max(ys), text, conf * 100))
        items.sort(key=lambda t: (t[1] + t[3]) / 2)
        lines: list[list] = []
        for it in items:
            cy = (it[1] + it[3]) / 2
            if lines and abs(cy - lines[-1][0]) < (it[3] - it[1]) * 0.6:
                lines[-1][1].append(it)
            else:
                lines.append([cy, [it]])
        out = []
        for _, group in lines:
            group.sort(key=lambda t: -t[0])  # right-to-left reading order
            x0 = min(g[0] for g in group); y0 = min(g[1] for g in group)
            x1 = max(g[2] for g in group); y1 = max(g[3] for g in group)
            conf = sum(g[5] for g in group) / len(group)
            bbox = [int(x0 / scale) + offset[0], int(y0 / scale) + offset[1],
                    int((x1 - x0) / scale), int((y1 - y0) / scale)]
            text = _join_rtl([g[4] for g in group])
            out.append(TextLine(text=text, bbox=bbox, conf=conf,
                                source=src_for(conf, bbox), pass_id=pass_id))
        return out

    def read(self, loaded: LoadedImage, prep: Prepared) -> EngineOutput:
        out = EngineOutput(engine=self.name)
        from .tesseract import _inside
        strip = prep.strip_box

        def src(conf, bbox):
            if loaded.kind == "screenshot":
                return "digital"
            if strip is not None:
                return "printed" if _inside(bbox, strip) else "handwritten"
            return "printed" if conf >= 70 else "handwritten"

        cands, hints = parse_lines(self._lines(prep.ocr_gray, "full", src), self.name)
        out.candidates += cands
        out.debug["full_text"] = hints.full_text
        out.transfer_votes = hints.transfer_votes
        out.status_votes = hints.status_votes
        out.issuer_banks = set(hints.issuer_banks)
        if prep.strip_gray is not None and prep.strip_box is not None:
            sl = self._lines(prep.strip_gray, "strip", lambda c, b: "printed",
                             offset=prep.strip_origin or tuple(prep.strip_box[:2]), scale=2.0)
            scands, shints = parse_lines(sl, self.name)
            out.candidates += scands
            out.debug["strip_text"] = shints.full_text
        return out
