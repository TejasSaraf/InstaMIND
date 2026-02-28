import json
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.schemas import Incident, IncidentReport, IncidentType
from app.services.local_gemma_client import LocalGemmaClient
from app.services.pose_event_detector import PoseEventDetector


class IncidentAnalysisAgent:
    """
    Agentic analyzer:
      1) Observe multi-modal signals (video/audio proxies + TensorFlow posture)
      2) Reason about threats
      3) Emit incidents + timeline + actions
    """

    def __init__(self) -> None:
        self.pose_event_detector = PoseEventDetector(
            model_path=settings.pose_event_model_path,
            label_path=settings.pose_event_label_path,
        )
        self.local_gemma_client = LocalGemmaClient()

    def analyze(self, source_filename: str, signals: dict, processing_time_ms: float) -> IncidentReport:
        signals["fast_path"] = self.pose_event_detector.predict(signals)
        incidents = self._local_reasoning(signals)
        if settings.model_mode == "local_gemma":
            incidents = self.local_gemma_client.refine_incidents(signals, incidents)
        summary = self._build_summary(incidents)
        timeline = [
            {
                "t": incident.timestamp_seconds,
                "type": incident.incident_type.value,
                "confidence": incident.confidence,
                "note": incident.evidence,
            }
            for incident in incidents
        ]

        return IncidentReport(
            report_id=str(uuid4()),
            source_filename=source_filename,
            created_at=datetime.now(tz=timezone.utc),
            processing_time_ms=processing_time_ms,
            emergency_latency_target_ms=settings.emergency_latency_target_ms,
            met_latency_target=processing_time_ms <= settings.emergency_latency_target_ms,
            offline_mode=settings.offline_mode,
            video_never_leaves_device=settings.video_never_leaves_device,
            summary=summary,
            incidents=incidents,
            timeline=timeline,
            raw_signals=signals,
        )

    def _local_reasoning(self, signals: dict) -> list[Incident]:
        video = signals.get("video", {})
        pose = signals.get("pose", {})
        audio = signals.get("audio", {})

        motion_mean = float(video.get("motion_mean", 0.0))
        motion_std = float(video.get("motion_std", 0.0))
        horizontal_score = float(pose.get("horizontal_posture_score", 0.0))
        area_change_mean = float(pose.get("area_change_mean", 0.0))
        audio_distress_score = float(audio.get("distress_score", 0.0))
        fast_path = signals.get("fast_path", {})
        fast_probs = fast_path.get("event_probs", {})

        incidents: list[Incident] = []

        if fast_path.get("available"):
            faint_prob = float(fast_probs.get("fainting", 0.0))
            choke_prob = float(fast_probs.get("choking", 0.0))
            violent_prob = float(fast_probs.get("violent_activity", 0.0))
            if faint_prob >= 0.65:
                incidents.append(
                    Incident(
                        incident_type=IncidentType.fainting,
                        confidence=min(0.99, faint_prob),
                        timestamp_seconds=1.0,
                        evidence="Fast-path pose detector triggered fainting pattern.",
                        recommended_action="Dispatch nearby security/medical responder immediately.",
                    )
                )
            if choke_prob >= 0.65:
                incidents.append(
                    Incident(
                        incident_type=IncidentType.choking,
                        confidence=min(0.99, choke_prob),
                        timestamp_seconds=1.0,
                        evidence="Fast-path pose detector triggered choking/distress pattern.",
                        recommended_action="Issue immediate emergency alert and request human confirmation.",
                    )
                )
            if violent_prob >= 0.7:
                incidents.append(
                    Incident(
                        incident_type=IncidentType.violent_activity,
                        confidence=min(0.99, violent_prob),
                        timestamp_seconds=1.0,
                        evidence="Fast-path pose detector triggered violent motion pattern.",
                        recommended_action="Trigger high-priority security escalation.",
                    )
                )

        if horizontal_score > 0.65 and motion_mean < 12:
            incidents.append(
                Incident(
                    incident_type=IncidentType.fainting,
                    confidence=min(0.95, 0.45 + horizontal_score / 2),
                    timestamp_seconds=2.0,
                    evidence="Body posture became horizontal while movement dropped.",
                    recommended_action="Dispatch nearby security/medical responder immediately.",
                )
            )

        if motion_mean > 18 and motion_std > 14:
            incidents.append(
                Incident(
                    incident_type=IncidentType.violent_activity,
                    confidence=min(0.92, 0.4 + motion_std / 40),
                    timestamp_seconds=3.5,
                    evidence="Abrupt high-variance motion pattern observed.",
                    recommended_action="Trigger high-priority security escalation.",
                )
            )
        elif motion_mean > 10:
            incidents.append(
                Incident(
                    incident_type=IncidentType.suspicious_activity,
                    confidence=min(0.88, 0.35 + motion_mean / 35),
                    timestamp_seconds=4.0,
                    evidence="Sustained movement anomaly relative to baseline.",
                    recommended_action="Monitor in real-time and notify floor guard.",
                )
            )

        if audio_distress_score > 0.6 or area_change_mean > 12000:
            incidents.append(
                Incident(
                    incident_type=IncidentType.choking,
                    confidence=min(0.9, 0.4 + max(audio_distress_score, 0.3)),
                    timestamp_seconds=5.0,
                    evidence="Distress-like audio/respiration proxy spike detected.",
                    recommended_action="Issue immediate emergency alert and request human confirmation.",
                )
            )

        if not incidents:
            incidents.append(
                Incident(
                    incident_type=IncidentType.none,
                    confidence=0.8,
                    timestamp_seconds=0.0,
                    evidence="No severe event detected from current multimodal signals.",
                    recommended_action="Continue monitoring.",
                )
            )

        if settings.model_mode == "gemini" and (not settings.offline_mode):
            incidents = self._augment_with_gemini_reasoning(signals, incidents)

        return incidents

    def _augment_with_gemini_reasoning(self, signals: dict, baseline: list[Incident]) -> list[Incident]:
        if not settings.gemini_api_key:
            return baseline

        prompt = ChatPromptTemplate.from_template(
            "You are a security incident reasoning agent.\n"
            "Input multimodal signals:\n{signals}\n\n"
            "Given baseline incidents:\n{baseline}\n\n"
            "Return ONLY a strict JSON list with fields: incident_type, confidence, timestamp_seconds, evidence, recommended_action.\n"
            "Allowed incident_type: fainting, choking, violent_activity, suspicious_activity, intrusion, none.\n"
        )
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model_name,
            google_api_key=settings.gemini_api_key,
            temperature=0.1,
        )
        chain = prompt | llm

        try:
            result = chain.invoke(
                {
                    "signals": json.dumps(signals, default=str),
                    "baseline": json.dumps([x.model_dump() for x in baseline], default=str),
                }
            )
            content = getattr(result, "content", "")
            parsed = self._parse_llm_incidents(content)
            return parsed or baseline
        except Exception:
            return baseline

    @staticmethod
    def _parse_llm_incidents(content: str) -> list[Incident]:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        payload = json.loads(text)
        if not isinstance(payload, list):
            return []

        incidents: list[Incident] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            raw_type = str(item.get("incident_type", "none"))
            try:
                incident_type = IncidentType(raw_type)
            except ValueError:
                incident_type = IncidentType.none

            incidents.append(
                Incident(
                    incident_type=incident_type,
                    confidence=float(item.get("confidence", 0.0)),
                    timestamp_seconds=float(item.get("timestamp_seconds", 0.0)),
                    evidence=str(item.get("evidence", "Model reasoning")),
                    recommended_action=str(item.get("recommended_action", "Continue monitoring.")),
                )
            )
        return incidents

    @staticmethod
    def _build_summary(incidents: list[Incident]) -> str:
        meaningful = [x for x in incidents if x.incident_type != IncidentType.none]
        if not meaningful:
            return "No critical event detected. Monitoring remains active."
        top = sorted(meaningful, key=lambda x: x.confidence, reverse=True)[:2]
        labels = ", ".join(f"{x.incident_type.value} ({x.confidence:.2f})" for x in top)
        return f"Detected potential incident(s): {labels}. Local alert workflow engaged."
