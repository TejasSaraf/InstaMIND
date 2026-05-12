from datetime import datetime

from enum import Enum

from typing import Any



from pydantic import BaseModel, Field





class IncidentType(str, Enum):

    fighting = "fighting"

    robbery = "robbery"

    shoplifting = "shoplifting"

    shooting = "shooting"

    fainting = "fainting"

    normal = "normal"





class Incident(BaseModel):

    incident_type: IncidentType

    confidence: float = Field(ge=0, le=1)

    timestamp_seconds: float = Field(ge=0)

    evidence: str

    recommended_action: str





class AgentGoal(BaseModel):

    name: str

    priority: int = Field(ge=1, le=5)

    rationale: str





class AgentDecision(BaseModel):

    action: str

    reason: str

    status: str





class AgentToolCall(BaseModel):

    tool_name: str

    arguments: dict[str, Any] = Field(default_factory=dict)

    reason: str = ""





class AgentExecutionStep(BaseModel):

    step: int = Field(ge=1)

    tool_call: AgentToolCall

    status: str

    outcome: str

    requires_human: bool = False





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

    primary_incident: Incident

    agent_goals: list[AgentGoal] = Field(default_factory=list)

    agent_decisions: list[AgentDecision] = Field(default_factory=list)

    agent_execution_steps: list[AgentExecutionStep] = Field(

        default_factory=list)

    timeline: list[dict[str, Any]]

    raw_signals: dict[str, Any]





class AnalyzeResponse(BaseModel):

    message: str

    report: IncidentReport





class GoogleAuthRequest(BaseModel):

    id_token: str





class AuthUser(BaseModel):

    id: str

    google_sub: str

    email: str

    name: str

    picture: str | None = None

    email_verified: bool = False

    created_at: datetime

    updated_at: datetime

    last_login_at: datetime





class GoogleAuthResponse(BaseModel):

    message: str

    user: AuthUser





class AuthStatusResponse(BaseModel):

    logged_in: bool

    user: AuthUser | None = None
