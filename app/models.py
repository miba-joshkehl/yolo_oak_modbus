from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    center_x: float
    center_y: float
    angle_deg: float


class InferenceRun(BaseModel):
    trigger_source: Literal["modbus", "api"]
    captured_at_utc: datetime
    image_path: str
    annotated_image_path: str
    inference_ms: float
    detections: list[Detection] = Field(default_factory=list)
    error: str | None = None


class SystemStatus(BaseModel):
    is_busy: bool
    ready: bool
    has_error: bool
    heartbeat: int
    last_run: InferenceRun | None = None
