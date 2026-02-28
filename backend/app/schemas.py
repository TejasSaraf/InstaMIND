from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class IncidentType(str, Enum):
    fainting = "fainting"
    choking = "choking"
    violent_activity = "violent_activity"
    suspicious_activity = "suspicious_activity"
    intrusion = "intrusion"
    none = "none"


class Incident(BaseModel):
    incident_type: IncidentType
    confidence: float = Field(ge=0, le=1)
    timestamp_seconds: float = Field(ge=0)
    evidence: str
    recommended_action: str


class IncidentReport(BaseModel):
    report_id: str
    source_filename: str
    created_at: datetime
    processing_time_ms: float
    emergency_latency_target_ms: int
    met_latency_target: bool
    offline_mode: bool
    video_never_leaves_device: bool
    summary: str
    incidents: list[Incident]
    timeline: list[dict[str, Any]]
    raw_signals: dict[str, Any]


class AnalyzeResponse(BaseModel):
    message: str
    report: IncidentReport
