"""Shared request/response models and detection types."""
from dataclasses import dataclass
from typing import Any, Literal, Optional

from pydantic import BaseModel

TeamId = Literal["home", "away"]
TrackState = dict[str, Any]


class RawFrame(BaseModel):
    base64: str
    timestamp: float


class AnalyzeFramesRequest(BaseModel):
    frames: list[RawFrame]
    homeKitColor: Optional[str] = None
    awayKitColor: Optional[str] = None


@dataclass
class Detection:
    cls_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    tracker_id: Optional[int] = None
