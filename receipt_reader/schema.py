"""Request/response models. Money is always an integer (rial or toman); never float.

Every field carries a description so the generated OpenAPI (Swagger, /docs) documents it.
"""
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

CONFIDENCE_DOC = ("Confidence level (not a probability): high = two different engines agree, or a "
                  "printed/digital reading passes every validator of its field; medium = one clean "
                  "reading, no conflict; low = handwriting only, failed validator or unclear; "
                  "conflict = readings disagree, no value is picked.")


class Candidate(BaseModel):
    """One reading of one field, from one engine pass. Never merged silently."""
    field: str
    value: Any
    raw_text: str
    source: Source
    engine: str
    pass_id: str = "full"            # "full" | "strip" — the same engine on another crop is NOT independent
    region: Optional[list[int]] = None  # [x, y, w, h] in the image the engine saw
    label: Optional[str] = None      # the label printed next to the value, e.g. «شماره پیگیری»
    role: Optional[Literal["dest", "source", "unknown"]] = None
    currency: Optional[Currency] = None
    ocr_conf: Optional[float] = None  # engine-native confidence, informational only

    def public(self) -> dict:
        d = {"value": self.value, "source": self.source, "engine": self.engine, "raw_text": self.raw_text}
        if self.pass_id != "full":
            d["pass"] = self.pass_id
        for k in ("label", "role", "currency", "region"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d


_CANDS = Field(default_factory=list, description="Every reading of this field with its evidence: value, source "
                                                 "(printed/handwritten/digital/words), engine, pass, raw_text, "
                                                 "label, region. Kept even when they disagree.")


class AmountField(BaseModel):
    value: Optional[int] = Field(None, description="Amount in RIAL (integer). null when unreadable or in conflict.")
    confidence: Confidence = Field("low", description=CONFIDENCE_DOC)
    suggested_value: Optional[int] = Field(
        None, description="Only when confidence is 'conflict': the reading a human should look at first "
                          "(the single printed/digital value), in rial. Never used for matching.")
    suggestion_basis: Optional[str] = Field(None, description="Why suggested_value was proposed.")
    candidates: list[dict] = _CANDS


class IbanField(BaseModel):
    value: Optional[str] = Field(None, description="Destination IBAN 'IR'+24 digits, or '...NNNN' when only the "
                                                   "end is visible (masked/partial).")
    partial: bool = Field(False, description="True for a masked/partial IBAN. A partial IBAN is never "
                                             "checksum-validated and never gives an exact match.")
    checksum_valid: Optional[bool] = Field(None, description="ISO 13616 mod-97 result; null for partial IBANs.")
    bank: Optional[str] = Field(None, description="Bank from the IBAN bank code (data/banks.json).")
    bank_fa: Optional[str] = None
    confidence: Confidence = Field("low", description=CONFIDENCE_DOC)
    candidates: list[dict] = _CANDS


class DateField(BaseModel):
    jalali: Optional[str] = Field(None, description="As printed, normalized: 'YYYY/MM/DD HH:MM' (Jalali).")
    iso: Optional[str] = Field(None, description="ISO-8601 (Tehran, +03:30); date only when no time was read.")
    plausible: Optional[bool] = Field(None, description="Not in the future and not older than RR_DATE_MAX_AGE_DAYS.")
    confidence: Confidence = Field("low", description=CONFIDENCE_DOC)
    candidates: list[dict] = _CANDS


class TrackingField(BaseModel):
    value: Optional[str] = Field(None, description="Primary tracking/reference number (digits).")
    label: Optional[str] = Field(None, description="The label it had on the document, e.g. «شماره پیگیری».")
    all: list[dict] = Field(default_factory=list,
                            description="Every distinct tracking/reference number with its own label.")
    confidence: Confidence = Field("low", description=CONFIDENCE_DOC)
    candidates: list[dict] = _CANDS


class TextField(BaseModel):
    value: Optional[str] = Field(None, description="The value as read (names), digits for accounts, "
                                                   "'…NNNN' (last 4 digits) for card numbers, bank name for banks.")
    confidence: Confidence = Field("low", description=CONFIDENCE_DOC)
    candidates: list[dict] = _CANDS


class Party(BaseModel):
    """What the receipt shows about one side of the transfer. Only filled when actually visible."""
    name: TextField = Field(default_factory=TextField, description="Account/card holder name.")
    account: TextField = Field(default_factory=TextField, description="Account number (not IBAN).")
    card: TextField = Field(default_factory=TextField, description="Card number, last 4 digits ('…4821').")
    iban: TextField = Field(default_factory=TextField, description="IBAN of this party. For the receiver this "
                                                                    "mirrors fields.dest_iban.")
    bank: TextField = Field(default_factory=TextField, description="Bank named on the document for this party.")


class Fields(BaseModel):
    amount_rial: AmountField = Field(default_factory=AmountField)
    currency: Currency = Field("unknown", description="Unit printed on the document; 'unknown' when none is "
                                                      "visible (never guessed).")
    amount_toman: Optional[int] = Field(None, description="amount_rial // 10, only when it divides exactly.")
    dest_iban: IbanField = Field(default_factory=IbanField)
    paid_at: DateField = Field(default_factory=DateField)
    tracking_number: TrackingField = Field(default_factory=TrackingField)
    transfer_type: TransferType = Field("unknown", description="From the document's own words / layout.")
    document_status: DocStatus = Field(
        "unknown", description="What the DOCUMENT says (موفق / ناموفق / در انتظار / ثبت شده …). It is not a "
                               "settlement confirmation; see `settlement`.")
    dest_bank: Optional[str] = Field(None, description="Bank of dest_iban (from its bank code).")
    sender: Party = Field(default_factory=Party, description="Payer side (از حساب / مبدا / برداشت از …).")
    receiver: Party = Field(default_factory=Party, description="Payee side (به حساب / مقصد / ذینفع / دارنده کارت …).")
    amount_words: TextField = Field(default_factory=TextField, description="Amount written in words, as read.")


class ExpectedLine(BaseModel):
    """One open line of the debtor, sent by the bot. Used only AFTER the blind read, for matching."""
    line_id: str = Field(min_length=1, max_length=100, description="The bot's line id.")
    iban: Optional[str] = Field(None, description="Destination IBAN of the line.")
    holder_name: Optional[str] = Field(None, description="Account holder name of the line.")
    amount_toman: int = Field(gt=0, description="Line amount in TOMAN (integer).")
    remaining_toman: Optional[int] = Field(default=None, ge=0,
                                           description="Still owed on this line, in toman. Defaults to amount_toman.")

    model_config = {"json_schema_extra": {"examples": [{
        "line_id": "L-17", "iban": "IR360570000000000123450001", "holder_name": "علی کریمی",
        "amount_toman": 50000000, "remaining_toman": 50000000}]}}


class LineMatch(BaseModel):
    line_id: str
    score: float = Field(description="Ranking aid only, not a probability.")
    fits: bool = Field(description="Amount exact/partial AND IBAN not a mismatch.")
    reasons: list[str] = Field(description="AMOUNT_EXACT, AMOUNT_PARTIAL, AMOUNT_OVER_REMAINING, IBAN_FULL_MATCH, "
                                           "IBAN_LAST_DIGITS_MATCH, IBAN_MISMATCH, NAME_MATCH, …")
    currency_assumed: Optional[Currency] = Field(None, description="Set when the document had no unit and the "
                                                                   "matcher tried both readings.")


class ImageInfo(BaseModel):
    sha256: str = Field(description="Of the uploaded bytes — for the bot's duplicate check.")
    phash: str = Field(description="Perceptual hash — near-duplicate check (re-screenshots, crops).")
    width: int
    height: int
    format: str
    kind: Literal["photo", "screenshot", "unknown"] = "unknown"
    flags: list[str] = Field(default_factory=list, description="Signals only, never a rejection.")


class Settlement(BaseModel):
    confirmed_by_this_service: Literal[False] = Field(
        False, description="Always false. A receipt image is not proof that money arrived.")
    note: str = Field("Settlement must be confirmed outside this service (by the receiver / bank statement). "
                      "document_status only reports what the document says.")


class Extraction(BaseModel):
    """Result of the blind read, independent of any expected lines."""
    receipt_id: str
    request_id: str
    image: ImageInfo
    fields: Fields
    tracking_number_normalized: Optional[str] = Field(None, description="For the bot's duplicate check.")
    settlement: Settlement = Field(default_factory=Settlement)
    warnings: list[str] = Field(default_factory=list)
    engines: dict[str, str] = Field(default_factory=dict, description="engine -> ok | error: … | skipped: …")
    processing_ms: int
    debug: Optional[dict] = None


class ReadResponse(Extraction):
    matches: list[LineMatch] = Field(default_factory=list)
    best_line_id: Optional[str] = None
    suggested_route: Route = Field(description="A hint; the bot applies its own policy on top. "
                                               "match | confirm_with_sender | manager_review")
    sender_question_fa: Optional[str] = Field(None, description="Persian question for the debtor when the route "
                                                               "is confirm_with_sender.")


class EngineStatus(BaseModel):
    loaded: bool
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    engines: dict[str, EngineStatus]
    default_engines: list[str]
    cloud_vlm_allowed: bool = Field(description="Images are sent to a cloud provider only when true.")
    request_id: str


class ErrorResponse(BaseModel):
    error: str = Field(description="Machine-readable code, e.g. UNAUTHORIZED, FILE_TOO_LARGE, UNSUPPORTED_TYPE.")
    detail: str
    request_id: str
