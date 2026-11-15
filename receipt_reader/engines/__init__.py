"""Engine registry."""
from __future__ import annotations

from ..config import Settings
from .base import Engine, EngineOutput


def build_engines(settings: Settings) -> dict[str, Engine]:
    from .easyocr_engine import EasyOCREngine
    from .tesseract import TesseractEngine
    from .vlm import VLMEngine

    return {
        "vlm": VLMEngine(settings),
        "tesseract": TesseractEngine(settings),
        "easyocr": EasyOCREngine(settings),
    }


__all__ = ["Engine", "EngineOutput", "build_engines"]
