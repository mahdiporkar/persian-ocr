"""Request/response models. Money is always an integer (rial or toman); never float."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Source = Literal["printed", "handwritten", "digital", "words"]
Confidence = Literal["high", "medium", "low", "conflict"]
TransferType = Literal["satna", "paya", "internal", "card_to_card", "unknown"]
DocStatus = Literal["success", "failed", "pending", "unknown"]
Currency = Literal["rial", "toman", "unknown"]
Route = Literal["match", "confirm_with_sender", "manager_review"]

REQUIRED_FIELDS = ("amount_rial", "dest_iban", "paid_at", "tracking_number")


class Candidate(BaseModel):
    """One reading of one field, from one engine pass. Never merged silently."""
    field: str
    value: Any
    raw_text: str
    source: Source
    engine: str
    pass_id: str = "full"            # "full" | "strip" | ... — different crops count as independent
    region: Optional[list[int]] = None  # [x, y, w, h] in the oriented image, when known
    label: Optional[str] = None      # e.g. «شماره پیگیری» for tracking numbers
    role: Optional[Literal["dest", "source", "unknown"]] = None  # for IBAN / names
    currency: Optional[Currency] = None
    ocr_conf: Optional[float] = None  # engine-native confidence, informational only

    def public(self) -> dict:
        d = {"value": self.value, "source": self.source, "engine": self.engine,
             "raw_text": self.raw_text}
        if self.pass_id != "full":
            d["pass"] = self.pass_id
        for k in ("label", "role", "currency", "region"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d


class AmountField(BaseModel):
    value: Optional[int] = None
    confidence: Confidence = "low"
    candidates: list[dict] = Field(default_factory=list)


class IbanField(BaseModel):
    value: Optional[str] = None
    partial: bool = False
    checksum_valid: Optional[bool] = None
    bank: Optional[str] = None
    bank_fa: Optional[str] = None
    confidence: Confidence = "low"
    candidates: list[dict] = Field(default_factory=list)


class DateField(BaseModel):
    jalali: Optional[str] = None
    iso: Optional[str] = None
    plausible: Optional[bool] = None
    confidence: Confidence = "low"
    candidates: list[dict] = Field(default_factory=list)


class TrackingField(BaseModel):
    value: Optional[str] = None
    label: Optional[str] = None
    all: list[dict] = Field(default_factory=list)   # every distinct number with its own label
    confidence: Confidence = "low"
    candidates: list[dict] = Field(default_factory=list)


class TextField(BaseModel):
    value: Optional[str] = None
    confidence: Confidence = "low"
    candidates: list[dict] = Field(default_factory=list)


class Fields(BaseModel):
    amount_rial: AmountField = Field(default_factory=AmountField)
    currency: Currency = "unknown"
    amount_toman: Optional[int] = None
    dest_iban: IbanField = Field(default_factory=IbanField)
    paid_at: DateField = Field(default_factory=DateField)
    tracking_number: TrackingField = Field(default_factory=TrackingField)
    transfer_type: TransferType = "unknown"
    document_status: DocStatus = "unknown"
    dest_holder_name: TextField = Field(default_factory=TextField)
    dest_bank: Optional[str] = None
    source_name: TextField = Field(default_factory=TextField)
    source_account: TextField = Field(default_factory=TextField)
    amount_words: TextField = Field(default_factory=TextField)


class ExpectedLine(BaseModel):
    line_id: str
    iban: Optional[str] = None
    holder_name: Optional[str] = None
    amount_toman: int
    remaining_toman: Optional[int] = None


class LineMatch(BaseModel):
    line_id: str
    score: float
    fits: bool
    reasons: list[str]
    currency_assumed: Optional[Currency] = None


class ImageInfo(BaseModel):
    sha256: str
    phash: str
    width: int
    height: int
    format: str
    kind: Literal["photo", "screenshot", "unknown"] = "unknown"
    flags: list[str] = Field(default_factory=list)


class ReadResponse(BaseModel):
    receipt_id: str
    request_id: str
    image: ImageInfo
    fields: Fields
    tracking_number_normalized: Optional[str] = None
    matches: list[LineMatch] = Field(default_factory=list)
    best_line_id: Optional[str] = None
    suggested_route: Route
    sender_question_fa: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    engines: dict[str, str] = Field(default_factory=dict)  # engine -> "ok" | "error: ..." | "skipped: ..."
    processing_ms: int
    debug: Optional[dict] = None


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str
