import json
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.schemas import Incident, IncidentReport, IncidentType
from app.services.local_gemma_client import LocalGemmaClient
from app.services.pose_event_detector import PoseEventDetector

STRIP_TYPES = {IncidentType.fainting, IncidentType.choking}


class IncidentAnalysisAgent:
    """
    Agentic analyzer focused on shoplifting / suspicious-activity detection.
    Pipeline:
      1. Fast-path TF pose detector (primary signal)
      2. Signal-context heuristics (boost / fallback)
      3. LoRA-tuned LLM for evidence enrichment only
    """

    def __init__(self) -> None:
        self.pose_event_detector = PoseEventDetector(
            model_path=settings.pose_event_model_path,
            label_path=settings.pose_event_label_path,
        )
        self.local_gemma_client = LocalGemmaClient()

    def analyze(self, source_filename: str, signals: dict, processing_time_ms: float) -> IncidentReport:
        signals["fast_path"] = self.pose_event_detector.predict(signals)
        fast_path = signals.get("fast_path", {})

        if settings.model_mode == "gemini" and settings.gemini_api_key and (not settings.offline_mode):
            incidents = self._gemini_primary_classify(signals)
            if not incidents:
                incidents = self._detect_shoplifting(signals, fast_path)
        elif settings.model_mode == "local_gemma" and self.local_gemma_client.available():
            incidents = self._detect_shoplifting(signals, fast_path)
        else:
            incidents = self._detect_shoplifting(signals, fast_path)

        incidents = [i for i in incidents if i.incident_type not in STRIP_TYPES]

        # Final hard filter: never emit fainting or choking (shoplifting-only pipeline)
        incidents = [i for i in incidents if i.incident_type not in (IncidentType.fainting, IncidentType.choking)]

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

        # Last-line safety: never expose fainting/choking to API (shoplifting-only pipeline)
        def _sanitize(inc: Incident) -> Incident:
            if inc.incident_type in (IncidentType.fainting, IncidentType.choking):
                return Incident(
                    incident_type=IncidentType.none,
                    confidence=0.8,
                    timestamp_seconds=inc.timestamp_seconds,
                    evidence="No critical event (pipeline is shoplifting-only).",
                    recommended_action="Continue monitoring.",
                )
            return inc

        incidents = [_sanitize(i) for i in incidents]

        summary = self._build_summary(incidents)
        timeline = [
            {
                "t": inc.timestamp_seconds,
                "type": inc.incident_type.value,
                "confidence": inc.confidence,
                "note": inc.evidence,
            }
            for inc in incidents
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

    def _detect_shoplifting(self, signals: dict, fast_path: dict) -> list[Incident]:
        """
        Shoplifting-focused detection combining TF detector + signal heuristics.
        LLM is used only for evidence enrichment.
        """
        video = signals.get("video", {})
        pose = signals.get("pose", {})
        audio = signals.get("audio", {})
        horizontal = float(pose.get("horizontal_posture_score", 0.0))
        motion_mean = float(video.get("motion_mean", 0.0))
        motion_std = float(video.get("motion_std", 0.0))
        distress = float(audio.get("distress_score", 0.0))
        area_change = float(pose.get("area_change_mean", 0.0))

        fast_available = fast_path.get("available", False)
        fast_probs = fast_path.get("event_probs", {})
        shoplifting_prob = float(fast_probs.get("shoplifting", 0.0))
        suspicious_prob = float(fast_probs.get("suspicious_activity", 0.0))
        none_prob = float(fast_probs.get("none", 1.0))

        incident_type = IncidentType.none
        confidence = 0.0
        evidence_parts: list[str] = []

        if fast_available:
            n_classes = max(len(fast_probs), 1)
            uniform = 1.0 / n_classes

            if shoplifting_prob > uniform:
                incident_type = IncidentType.shoplifting
                confidence = shoplifting_prob

                if distress < 0.3:
                    confidence += 0.30
                if horizontal < 0.8:
                    confidence += 0.10
                if motion_mean > 0.5:
                    confidence += 0.10

                confidence = min(0.95, confidence)
                evidence_parts.append(
                    f"TF pose detector: shoplifting={shoplifting_prob:.1%}, "
                    f"suspicious={suspicious_prob:.1%}, none={none_prob:.1%}. "
                    f"Signal boost applied (distress={distress:.2f}, horizontal={horizontal:.2f}, motion={motion_mean:.1f})."
                )

            elif suspicious_prob > uniform and suspicious_prob > none_prob:
                incident_type = IncidentType.suspicious_activity
                confidence = min(0.85, suspicious_prob + 0.25)
                evidence_parts.append(f"TF pose detector flagged suspicious activity ({suspicious_prob:.1%}).")

        if incident_type == IncidentType.none:
            if motion_mean > 0.5 and distress < 0.3 and horizontal < 0.85:
                incident_type = IncidentType.shoplifting
                confidence = min(0.80, 0.50 + motion_mean / 15)
                evidence_parts.append(
                    f"Signal heuristic: person is upright (h={horizontal:.2f}), "
                    f"moving (motion={motion_mean:.1f}), no distress ({distress:.2f}) — retail anomaly pattern."
                )
            elif motion_mean > 10 and motion_std > 8:
                incident_type = IncidentType.violent_activity
                confidence = min(0.90, 0.4 + motion_std / 40)
                evidence_parts.append("Abrupt high-variance motion pattern observed.")

        if incident_type != IncidentType.none and confidence >= 0.4:
            llm_evidence, llm_action = self._get_llm_evidence(signals, incident_type)
            if llm_evidence:
                evidence_parts.append(f"LLM: {llm_evidence}")

            action = llm_action or self._default_action(incident_type)

            return [
                Incident(
                    incident_type=incident_type,
                    confidence=confidence,
                    timestamp_seconds=1.5,
                    evidence=" ".join(evidence_parts),
                    recommended_action=action,
                )
            ]

        return []

    def _get_llm_evidence(self, signals: dict, detected_type: IncidentType) -> tuple[str, str]:
        """Ask the LoRA-tuned LLM for evidence text. Returns (evidence, action)."""
        if not self.local_gemma_client.available():
            return "", ""
        try:
            summary_text = self._build_multimodal_summary(signals)
            incidents = self.local_gemma_client.primary_classify(summary_text)
            if incidents:
                return incidents[0].evidence, incidents[0].recommended_action
        except Exception:
            pass
        return "", ""

    @staticmethod
    def _default_action(incident_type: IncidentType) -> str:
        actions = {
            IncidentType.shoplifting: "Track subject, notify nearby staff, and retain camera evidence.",
            IncidentType.suspicious_activity: "Monitor in real-time and notify floor guard.",
            IncidentType.violent_activity: "Trigger high-priority security escalation.",
            IncidentType.intrusion: "Alert security team immediately.",
        }
        return actions.get(incident_type, "Continue monitoring.")

    def _gemini_primary_classify(self, signals: dict) -> list[Incident]:
        if not settings.gemini_api_key:
            return []
        summary_text = self._build_multimodal_summary(signals)
        prompt = (
            "You are the primary security incident classifier. Use ONLY the multimodal summary below.\n\n"
            "RULES:\n"
            "- Choose the SINGLE most likely incident type.\n"
            "- Focus on: shoplifting, suspicious_activity, violent_activity, or none.\n"
            "- Shoplifting = concealment, item handling, retail context, person leaving without paying.\n"
            "- Do NOT classify as fainting or choking.\n"
            "- Return ONLY a JSON array with exactly one object: {\"incident_type\", \"confidence\", \"timestamp_seconds\", \"evidence\", \"recommended_action\"}.\n"
            "Allowed incident_type: shoplifting, suspicious_activity, violent_activity, intrusion, none.\n\n"
            f"MULTIMODAL SUMMARY:\n{summary_text}\n\n"
            "JSON array:"
        )
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model_name,
            google_api_key=settings.gemini_api_key,
            temperature=0.1,
        )
        try:
            result = llm.invoke(prompt)
            content = getattr(result, "content", "")
            parsed = self._parse_llm_incidents(content)
            return parsed if parsed else []
        except Exception:
            return []

    @staticmethod
    def _build_multimodal_summary(signals: dict) -> str:
        v = signals.get("video", {})
        p = signals.get("pose", {})
        a = signals.get("audio", {})
        fps = v.get("fps", 0)
        duration = v.get("duration_seconds", 0)
        motion_mean = v.get("motion_mean", 0)
        motion_std = v.get("motion_std", 0)
        brightness = v.get("brightness_mean", 0)
        horizontal = p.get("horizontal_posture_score", 0)
        area_change = p.get("area_change_mean", 0)
        audio_distress = a.get("distress_score", 0)
        lines = [
            f"Video: fps={fps:.1f}, duration_sec={duration:.1f}, motion_mean={motion_mean:.2f}, motion_std={motion_std:.2f}, brightness_mean={brightness:.1f}.",
            f"Body pose (TensorFlow): horizontal_posture_score={horizontal:.2f} (1=lying/collapsed), area_change_mean={area_change:.0f}.",
            f"Audio: distress_score={audio_distress:.2f} (high=distress/coughing).",
        ]
        fast = signals.get("fast_path", {})
        if fast.get("available") and fast.get("event_probs"):
            probs = fast["event_probs"]
            lines.append(f"Fast detector probs: {json.dumps(probs, indent=0)}.")
        return "\n".join(lines)

    @staticmethod
    def _parse_llm_incidents(content: str) -> list[Incident]:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = [payload]
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

            if incident_type in STRIP_TYPES:
                continue

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
