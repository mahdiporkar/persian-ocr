"""Service configuration, read from environment variables (prefix RR_)."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RR_", env_file=".env", extra="ignore")

    # --- API ---
    api_keys: str = Field("", description="Comma-separated API keys accepted in X-API-Key")
    max_upload_mb: float = 10.0
    min_image_side: int = 300
    max_image_pixels: int = 60_000_000
    request_timeout_s: float = 90.0
    allow_heic: bool = True

    # --- engines ---
    engines: str = "vlm,tesseract"  # default set run on every receipt
    allow_cloud_vlm: bool = False   # images leave the server only when true
    vlm_provider: str = "none"      # "anthropic" | "openai_compatible" | "none"
    vlm_model: str = "claude-opus-5-5"
    vlm_base_url: Optional[str] = None   # e.g. http://localhost:8000/v1 for vLLM/Ollama
    vlm_api_key: Optional[str] = None
    vlm_self_hosted: Optional[bool] = None  # override auto-detection for openai_compatible
    vlm_effort: str = "high"
    vlm_timeout_s: float = 60.0
    vlm_max_side: int = 2000         # downscale only when larger (keeps original otherwise)
    vlm_strip_reread: bool = True    # second VLM pass on the printed-strip crop

    tesseract_cmd: Optional[str] = None
    tessdata_prefix: Optional[str] = None
    tesseract_langs: str = "fas+eng"
    tesseract_printed_min_conf: float = 75.0
    # EasyOCR outputs «٥» for both 5 and the circle-shaped Persian zero; treat it as unreadable
    easyocr_ambiguous_five: bool = True

    # --- rules ---
    date_max_age_days: int = 3
    max_amount_rial: int = 10**14
    min_amount_rial: int = 10_000
    name_match_threshold: int = 85

    log_level: str = "INFO"

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def engine_list(self) -> list[str]:
        return [e.strip() for e in self.engines.split(",") if e.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
