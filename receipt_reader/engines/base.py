"""Engine interface. Every reader returns candidates; none of them sees expected values."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..imaging import LoadedImage, Prepared
from ..schema import Candidate


@dataclass
class EngineOutput:
    engine: str
    candidates: list[Candidate] = field(default_factory=list)
    transfer_votes: dict[str, int] = field(default_factory=dict)
    status_votes: dict[str, int] = field(default_factory=dict)
    issuer_banks: set[str] = field(default_factory=set)       # bank codes named in header/footer
    printed_strip_box: Optional[list[int]] = None   # in Prepared.document coordinates
    suspected_editing: bool = False
    debug: dict = field(default_factory=dict)


class Engine(ABC):
    name: str = "engine"

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """(usable?, reason). Must be cheap; called at startup and on /health."""

    @abstractmethod
    def read(self, loaded: LoadedImage, prep: Prepared) -> EngineOutput:
        """Read the image blind. Must not receive or use any expected values."""
