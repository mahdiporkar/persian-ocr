"""A deterministic engine that returns pre-set candidates. For tests and the eval harness's dry runs."""
from __future__ import annotations

from typing import Callable, Optional

from ..imaging import LoadedImage, Prepared
from .base import Engine, EngineOutput


class ScriptedEngine(Engine):
    def __init__(self, name: str, fn: Callable[[LoadedImage, Prepared], EngineOutput],
                 ok: bool = True, reason: str = "ok"):
        self.name = name
        self._fn = fn
        self._ok = ok
        self._reason = reason

    def available(self) -> tuple[bool, str]:
        return self._ok, self._reason

    def read(self, loaded: LoadedImage, prep: Prepared) -> EngineOutput:
        return self._fn(loaded, prep)
