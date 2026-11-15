"""FastAPI service. Internal; called only by the settlement bot."""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import uuid
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter, ValidationError
from starlette.concurrency import run_in_threadpool

from . import __version__
from .config import Settings, get_settings
from .engines import build_engines
from .imaging import ImageError
from .logging_utils import setup_logging
from .pipeline import ReceiptReader
from .schema import ErrorResponse, ExpectedLine, ReadResponse

log = logging.getLogger("receipt_reader.api")


class ApiError(Exception):
    def __init__(self, status: int, code: str, detail: str):
        self.status, self.code, self.detail = status, code, detail


def create_app(settings: Optional[Settings] = None, reader: Optional[ReceiptReader] = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    app = FastAPI(title="Receipt Reader", version=__version__,
                  description="Reads Iranian bank transfer receipts and suggests which open line they pay. "
                              "It never approves anything; a receipt image is not proof that money arrived.")
    app.state.settings = settings
    app.state.reader = reader or ReceiptReader(settings, build_engines(settings))

    @app.middleware("http")
    async def request_id_mw(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        rid = getattr(request.state, "request_id", "")
        return JSONResponse(status_code=exc.status,
                            content=ErrorResponse(error=exc.code, detail=exc.detail, request_id=rid).model_dump(),
                            headers={"X-Request-ID": rid})

    @app.exception_handler(ImageError)
    async def image_error(request: Request, exc: ImageError):
        rid = getattr(request.state, "request_id", "")
        return JSONResponse(status_code=exc.status,
                            content=ErrorResponse(error=exc.code, detail=exc.detail, request_id=rid).model_dump(),
                            headers={"X-Request-ID": rid})

    def require_key(x_api_key: Optional[str] = Header(default=None)):
        keys = app.state.settings.api_key_set
        if not keys:
            raise ApiError(503, "NO_API_KEY_CONFIGURED", "Set RR_API_KEYS on the server.")
        if not x_api_key or not any(hmac.compare_digest(x_api_key, k) for k in keys):
            raise ApiError(401, "UNAUTHORIZED", "Missing or invalid X-API-Key header.")

    @app.get("/v1/health")
    async def health(request: Request):
        rdr: ReceiptReader = app.state.reader
        status = rdr.refresh_status()
        engines = {n: {"loaded": ok, "detail": reason} for n, (ok, reason) in status.items()}
        default = [e for e in app.state.settings.engine_list if status.get(e, (False,))[0]]
        return {"status": "ok" if default else "degraded", "version": __version__,
                "engines": engines, "default_engines": app.state.settings.engine_list,
                "cloud_vlm_allowed": app.state.settings.allow_cloud_vlm,
                "request_id": request.state.request_id}

    @app.post("/v1/receipts/read", response_model=ReadResponse,
              responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse},
                         413: {"model": ErrorResponse}, 415: {"model": ErrorResponse},
                         422: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
              dependencies=[Depends(require_key)])
    async def read_receipt(
        request: Request,
        image: UploadFile = File(..., description="JPG, PNG, WEBP (HEIC optional)"),
        expected_lines: Optional[str] = Form(None, description="JSON array of the debtor's open lines"),
        engines: Optional[str] = Form(None, description="Comma-separated engine names; default from config"),
        include_debug: bool = Form(False),
    ):
        s: Settings = app.state.settings
        max_bytes = int(s.max_upload_mb * 1024 * 1024)
        data = await image.read(max_bytes + 1)
        await image.close()
        if len(data) > max_bytes:
            raise ApiError(413, "FILE_TOO_LARGE", f"File is larger than {s.max_upload_mb} MB.")
        lines: list[ExpectedLine] = []
        if expected_lines:
            try:
                lines = TypeAdapter(list[ExpectedLine]).validate_python(json.loads(expected_lines))
            except (json.JSONDecodeError, ValidationError) as e:
                raise ApiError(400, "BAD_EXPECTED_LINES", f"expected_lines is not valid: {str(e)[:300]}")
        eng = [e.strip() for e in engines.split(",") if e.strip()] if engines else None
        try:
            result = await asyncio.wait_for(
                run_in_threadpool(app.state.reader.read, data, lines, eng, include_debug,
                                  request.state.request_id),
                timeout=s.request_timeout_s + 5)
        except asyncio.TimeoutError:
            raise ApiError(504, "TIMEOUT", "Reading the receipt took too long.")
        finally:
            del data  # nothing is stored; the bytes only live for this request
        log.info("receipt read route=%s warnings=%s ms=%s tracking=%s", result.suggested_route,
                 ",".join(result.warnings), result.processing_ms, result.tracking_number_normalized or "-")
        return result

    return app


def app_factory() -> FastAPI:  # for `uvicorn receipt_reader.api:app_factory --factory`
    return create_app()
