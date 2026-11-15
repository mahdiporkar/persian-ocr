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
from .schema import EngineStatus, ErrorResponse, ExpectedLine, Extraction, HealthResponse, ReadResponse

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
        # refuse oversized bodies before the multipart parser reads them
        limit = int(app.state.settings.max_upload_mb * 1024 * 1024) + 256 * 1024  # + form overhead
        cl = request.headers.get("content-length")
        if request.method == "POST" and cl and cl.isdigit() and int(cl) > limit:
            return JSONResponse(status_code=413, headers={"X-Request-ID": rid},
                                content=ErrorResponse(error="FILE_TOO_LARGE", request_id=rid,
                                                      detail=f"Request is larger than {app.state.settings.max_upload_mb} MB.").model_dump())
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

    def require_key(x_api_key: Optional[str] = Header(default=None, description="API key (RR_API_KEYS).")):
        keys = app.state.settings.api_key_set
        if not keys:
            raise ApiError(503, "NO_API_KEY_CONFIGURED", "Set RR_API_KEYS on the server.")
        if not x_api_key or not any(hmac.compare_digest(x_api_key, k) for k in keys):
            raise ApiError(401, "UNAUTHORIZED", "Missing or invalid X-API-Key header.")

    ERRORS = {400: {"model": ErrorResponse, "description": "Bad expected_lines / empty file"},
              401: {"model": ErrorResponse, "description": "Missing or invalid X-API-Key"},
              413: {"model": ErrorResponse, "description": "File or request too large"},
              415: {"model": ErrorResponse, "description": "Not a JPG/PNG/WEBP/HEIC image (checked from bytes)"},
              422: {"model": ErrorResponse, "description": "Corrupt image / too small"},
              503: {"model": ErrorResponse, "description": "No API key configured on the server"},
              504: {"model": ErrorResponse, "description": "Reading took longer than RR_REQUEST_TIMEOUT_S"}}
    IMAGE_DOC = "Receipt image: JPG, PNG, WEBP (HEIC when pillow-heif is installed). Max RR_MAX_UPLOAD_MB."
    ENGINES_DOC = "Comma-separated engines to run, e.g. `tesseract,easyocr` or `vlm,tesseract`. Default: RR_ENGINES."
    DEBUG_DOC = "Include OCR text, strip box, skew etc. in `debug`."
    LINES_DOC = ("JSON array (as a string) of the debtor's open lines. Each item: "
                 "`{line_id: str, iban: str|null, holder_name: str|null, amount_toman: int>0, "
                 "remaining_toman: int>=0|null}` (schema: ExpectedLine). Used only AFTER the blind read. "
                 'Example: `[{"line_id":"L-17","iban":"IR360570000000000123450001","holder_name":"علی کریمی",'
                 '"amount_toman":50000000,"remaining_toman":50000000}]`')

    @app.get("/v1/health", response_model=HealthResponse, summary="Service status and loaded engines")
    async def health(request: Request) -> HealthResponse:
        rdr: ReceiptReader = app.state.reader
        status = rdr.refresh_status()
        engines = {n: EngineStatus(loaded=ok, detail=reason) for n, (ok, reason) in status.items()}
        default = [e for e in app.state.settings.engine_list if status.get(e, (False,))[0]]
        return HealthResponse(status="ok" if default else "degraded", version=__version__, engines=engines,
                              default_engines=app.state.settings.engine_list,
                              cloud_vlm_allowed=app.state.settings.allow_cloud_vlm,
                              request_id=request.state.request_id)

    async def _image_bytes(image: UploadFile) -> bytes:
        s: Settings = app.state.settings
        max_bytes = int(s.max_upload_mb * 1024 * 1024)
        data = await image.read(max_bytes + 1)
        await image.close()
        if len(data) > max_bytes:
            raise ApiError(413, "FILE_TOO_LARGE", f"File is larger than {s.max_upload_mb} MB.")
        return data

    async def _run(fn, *args):
        try:
            return await asyncio.wait_for(run_in_threadpool(fn, *args),
                                          timeout=app.state.settings.request_timeout_s + 5)
        except asyncio.TimeoutError:
            raise ApiError(504, "TIMEOUT", "Reading the receipt took too long.")

    def _engines(engines: Optional[str]):
        return [e.strip() for e in engines.split(",") if e.strip()] if engines else None

    @app.post("/v1/receipts/read", response_model=ReadResponse, responses=ERRORS,
              dependencies=[Depends(require_key)],
              summary="Read a receipt blind, then match it against the debtor's open lines",
              description="Step 1 reads the image without any expected values. Step 2 scores each "
                          "expected line and suggests a route (match / confirm_with_sender / "
                          "manager_review). Nothing is ever approved and settlement is never confirmed.")
    async def read_receipt(
        request: Request,
        image: UploadFile = File(..., description=IMAGE_DOC),
        expected_lines: Optional[str] = Form(None, description=LINES_DOC),
        engines: Optional[str] = Form(None, description=ENGINES_DOC),
        include_debug: bool = Form(False, description=DEBUG_DOC),
    ) -> ReadResponse:
        data = await _image_bytes(image)
        lines: list[ExpectedLine] = []
        if expected_lines:
            try:
                lines = TypeAdapter(list[ExpectedLine]).validate_python(json.loads(expected_lines))
            except (json.JSONDecodeError, ValidationError) as e:
                raise ApiError(400, "BAD_EXPECTED_LINES", f"expected_lines is not valid: {str(e)[:300]}")
        try:
            result = await _run(app.state.reader.read, data, lines, _engines(engines), include_debug,
                                request.state.request_id)
        finally:
            del data  # nothing is stored; the bytes only live for this request
        log.info("receipt read route=%s warnings=%s ms=%s tracking=%s", result.suggested_route,
                 ",".join(result.warnings), result.processing_ms, result.tracking_number_normalized or "-")
        return result

    @app.post("/v1/receipts/extract", response_model=Extraction, responses=ERRORS,
              dependencies=[Depends(require_key)],
              summary="Read a receipt blind (no matching)",
              description="Extraction only: fields with all candidates and evidence, sender/receiver, "
                          "warnings. No expected lines are accepted, so nothing can bias the read.")
    async def extract_receipt(
        request: Request,
        image: UploadFile = File(..., description=IMAGE_DOC),
        engines: Optional[str] = Form(None, description=ENGINES_DOC),
        include_debug: bool = Form(False, description=DEBUG_DOC),
    ) -> Extraction:
        data = await _image_bytes(image)
        try:
            return await _run(app.state.reader.extract, data, _engines(engines), include_debug,
                              request.state.request_id)
        finally:
            del data

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        from fastapi.openapi.utils import get_openapi
        schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
        comps = schema.setdefault("components", {}).setdefault("schemas", {})
        # expected_lines travels as a JSON string inside the multipart form: publish its schema too
        comps["ExpectedLine"] = ExpectedLine.model_json_schema(ref_template="#/components/schemas/{model}")
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi
    return app


def app_factory() -> FastAPI:  # for `uvicorn receipt_reader.api:app_factory --factory`
    return create_app()
