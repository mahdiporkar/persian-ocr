"""Engine B: local OCR with Tesseract (fas+eng).

Two passes: the whole (cropped, deskewed, contrast-enhanced) document, and a 2x
zoomed crop of the detected printed strip. The passes count as independent reads
in cross-checking because they see different pixels at different scales.
"""
from __future__ import annotations

import os
import shutil
from collections import defaultdict
from typing import Optional

import numpy as np

from ..config import Settings
from ..imaging import LoadedImage, Prepared
from ..textparse import TextLine, parse_lines
from .base import Engine, EngineOutput

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

_WIN_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class TesseractEngine(Engine):
    name = "tesseract"

    def __init__(self, settings: Settings):
        self.s = settings
        if pytesseract is not None:
            cmd = settings.tesseract_cmd or shutil.which("tesseract") or (_WIN_DEFAULT if os.path.exists(_WIN_DEFAULT) else None)
            if cmd:
                pytesseract.pytesseract.tesseract_cmd = cmd
        if settings.tessdata_prefix:
            os.environ["TESSDATA_PREFIX"] = settings.tessdata_prefix

    def available(self) -> tuple[bool, str]:
        if pytesseract is None:
            return False, "pytesseract not installed"
        try:
            langs = set(pytesseract.get_languages(config=""))
        except Exception as e:
            return False, f"tesseract binary not found ({e.__class__.__name__})"
        need = set(self.s.tesseract_langs.split("+"))
        missing = need - langs
        if missing:
            return False, f"missing tessdata: {', '.join(sorted(missing))}"
        return True, "ok"

    def _lines(self, img: np.ndarray, pass_id: str, source_for, offset=(0, 0), psm: int = 4,
               scale: float = 1.0) -> list[TextLine]:
        data = pytesseract.image_to_data(img, lang=self.s.tesseract_langs, config=f"--oem 1 --psm {psm}",
                                         output_type=pytesseract.Output.DICT)
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, word in enumerate(data["text"]):
            if word and word.strip() and float(data["conf"][i]) >= 0:
                groups[(data["block_num"][i], data["par_num"][i], data["line_num"][i])].append(i)
        lines = []
        for key in sorted(groups, key=lambda k: (min(data["top"][i] for i in groups[k]), k)):
            idx = groups[key]
            words = [data["text"][i] for i in idx]
            confs = [float(data["conf"][i]) for i in idx]
            x0 = min(data["left"][i] for i in idx)
            y0 = min(data["top"][i] for i in idx)
            x1 = max(data["left"][i] + data["width"][i] for i in idx)
            y1 = max(data["top"][i] + data["height"][i] for i in idx)
            conf = float(np.mean(confs))
            bbox = [int(x0 / scale) + offset[0], int(y0 / scale) + offset[1], int((x1 - x0) / scale), int((y1 - y0) / scale)]
            lines.append(TextLine(text=" ".join(words), bbox=bbox, conf=conf, source=source_for(conf, bbox),
                                  pass_id=pass_id))
        return lines

    def read(self, loaded: LoadedImage, prep: Prepared) -> EngineOutput:
        out = EngineOutput(engine=self.name)
        thr = self.s.tesseract_printed_min_conf
        strip = prep.strip_box

        def src_full(conf: float, bbox: list[int]) -> str:
            if loaded.kind == "screenshot":
                return "digital"
            if strip is not None:
                # Paper form with a detected teller strip: values outside it are the
                # customer's handwriting, values inside it are machine print.
                return "printed" if _inside(bbox, strip) else "handwritten"
            # No strip found: Tesseract cannot tell ink from print, so low-confidence
            # lines are treated as handwriting (the conservative choice).
            return "printed" if conf >= thr else "handwritten"
        lines = self._lines(prep.ocr_gray, "full", src_full)
        cands, hints = parse_lines(lines, self.name)
        out.candidates.extend(cands)
        _merge(out.transfer_votes, hints.transfer_votes)
        _merge(out.status_votes, hints.status_votes)
        out.debug["full_text"] = hints.full_text

        if prep.strip_gray is not None and prep.strip_box is not None:
            bx, by = prep.strip_origin or (prep.strip_box[0], prep.strip_box[1])
            src_strip = lambda conf, bbox: "printed"  # noqa: E731
            slines = self._lines(prep.strip_gray, "strip", src_strip, offset=(bx, by), psm=6, scale=2.0)
            scands, shints = parse_lines(slines, self.name)
            out.candidates.extend(scands)
            _merge(out.transfer_votes, shints.transfer_votes)
            _merge(out.status_votes, shints.status_votes)
            out.debug["strip_text"] = shints.full_text
        return out


def _inside(b: list[int], box: list[int]) -> bool:
    cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
    return box[0] <= cx <= box[0] + box[2] and box[1] <= cy <= box[1] + box[3]


def _merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v
