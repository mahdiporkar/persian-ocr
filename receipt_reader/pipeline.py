"""Two independent steps:

1. extract(): image bytes -> blind read by every engine -> cross-check -> Extraction.
   It takes NO expected values; nothing about the debtor's lines can influence it.
2. match(): Extraction + expected lines -> line scores + suggested route.

read() = extract() then match().
"""
from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from datetime import datetime
from typing import Optional

from .config import Settings
from .crosscheck import Hints, resolve
from .dates import TEHRAN
from .engines.base import Engine, EngineOutput
from .imaging import load_image, prepare
from .matcher import decide
from .schema import Candidate, ExpectedLine, Extraction, ImageInfo, ReadResponse

log = logging.getLogger("receipt_reader.pipeline")

EDIT_FLAGS = {"EDITING_SOFTWARE_IN_EXIF", "VLM_SUSPECTED_EDITING"}


class ReceiptReader:
    def __init__(self, settings: Settings, engines: dict[str, Engine]):
        self.s = settings
        self.engines = engines
        self._pool = ThreadPoolExecutor(max_workers=max(2, len(engines) * 2))
        self.status = {name: e.available() for name, e in engines.items()}

    def refresh_status(self) -> dict[str, tuple[bool, str]]:
        self.status = {name: e.available() for name, e in self.engines.items()}
        return self.status

    # ------------------------------------------------------------- step 1
    def extract(self, data: bytes, engines: Optional[list[str]] = None, include_debug: bool = False,
                request_id: Optional[str] = None, now: Optional[datetime] = None) -> Extraction:
        t0 = time.perf_counter()
        now = now or datetime.now(TEHRAN)
        request_id = request_id or str(uuid.uuid4())
        loaded = load_image(data, max_bytes=int(self.s.max_upload_mb * 1024 * 1024),
                            min_side=self.s.min_image_side, max_pixels=self.s.max_image_pixels,
                            allow_heic=self.s.allow_heic)
        prep = prepare(loaded)

        wanted = engines or self.s.engine_list
        engine_report: dict[str, str] = {}
        futures = {}
        for name in wanted:
            eng = self.engines.get(name)
            if eng is None:
                engine_report[name] = "skipped: unknown engine"
                continue
            ok, reason = self.status.get(name) or eng.available()
            if not ok:
                engine_report[name] = f"skipped: {reason}"
                continue
            futures[name] = self._pool.submit(eng.read, loaded, prep)

        outputs: list[EngineOutput] = []
        deadline = time.perf_counter() + self.s.request_timeout_s
        for name, fut in futures.items():
            try:
                outputs.append(fut.result(timeout=max(1.0, deadline - time.perf_counter())))
                engine_report[name] = "ok"
            except FutTimeout:
                engine_report[name] = "error: timeout"
                log.warning("engine %s timed out", name)
            except Exception as e:  # an engine failure never fails the request
                engine_report[name] = f"error: {e.__class__.__name__}"
                log.warning("engine %s failed: %s", name, e.__class__.__name__)

        cands: list[Candidate] = [c for o in outputs for c in o.candidates]
        hints = Hints(transfer_votes={}, status_votes={})
        for o in outputs:
            for k, v in o.transfer_votes.items():
                hints.transfer_votes[k] = hints.transfer_votes.get(k, 0) + v
            for k, v in o.status_votes.items():
                hints.status_votes[k] = hints.status_votes.get(k, 0) + v

        fields, warnings = resolve(cands, hints, self.s, now)

        flags = list(loaded.flags)
        if any(o.suspected_editing for o in outputs):
            flags.append("VLM_SUSPECTED_EDITING")
        if EDIT_FLAGS & set(flags):
            warnings.append("POSSIBLE_EDITING")
        if prep.quality:
            warnings.append("LOW_IMAGE_QUALITY")
            flags += prep.quality
        if not outputs:
            warnings.append("NO_ENGINE_AVAILABLE")

        debug = None
        if include_debug:
            debug = {
                "image_kind": loaded.kind,
                "document_cropped": prep.cropped,
                "skew_deg": round(prep.skew, 2),
                "printed_strip_box": prep.strip_box,
                "quality": prep.quality,
                "engines": {o.engine: o.debug for o in outputs},
            }
        return Extraction(
            receipt_id=str(uuid.uuid4()),
            request_id=request_id,
            image=ImageInfo(sha256=loaded.sha256, phash=loaded.phash, width=loaded.pil.width,
                            height=loaded.pil.height, format=loaded.format, kind=loaded.kind, flags=flags),
            fields=fields,
            tracking_number_normalized=fields.tracking_number.value,
            warnings=warnings,
            engines=engine_report,
            processing_ms=int((time.perf_counter() - t0) * 1000),
            debug=debug,
        )

    # ------------------------------------------------------------- step 2
    def match(self, ext: Extraction, expected_lines: list[ExpectedLine]) -> ReadResponse:
        t0 = time.perf_counter()
        image_flags = ["POSSIBLE_EDITING"] if "POSSIBLE_EDITING" in ext.warnings else []
        matches, best, route, question, warnings = decide(
            ext.fields, list(ext.warnings), expected_lines, self.s.name_match_threshold, image_flags)
        if "NO_ENGINE_AVAILABLE" in warnings:
            route, question = "manager_review", None
        return ReadResponse(
            **ext.model_dump(exclude={"warnings", "processing_ms"}),
            warnings=warnings,
            processing_ms=ext.processing_ms + int((time.perf_counter() - t0) * 1000),
            matches=matches,
            best_line_id=best,
            suggested_route=route,
            sender_question_fa=question,
        )

    def read(self, data: bytes, expected_lines: list[ExpectedLine], engines: Optional[list[str]] = None,
             include_debug: bool = False, request_id: Optional[str] = None,
             now: Optional[datetime] = None) -> ReadResponse:
        return self.match(self.extract(data, engines, include_debug, request_id, now), expected_lines)
