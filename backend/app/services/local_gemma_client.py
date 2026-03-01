import json
import urllib.error
import urllib.request

from app.config import settings
from app.schemas import Incident, IncidentType

# Must match build_gemma_sft_dataset.CLASSIFIER_RULES and gemma_agent prompt for LoRA-tuned model.
PRIMARY_CLASSIFIER_RULES = (
    "You are the primary security incident classifier. Use ONLY the multimodal summary below.\n\n"
    "RULES:\n"
    "- Choose the SINGLE most likely incident type.\n"
    "- Focus on: shoplifting, suspicious_activity, violent_activity, or none.\n"
    "- Shoplifting = concealment, item handling, retail context, person moving with items.\n"
    "- Do NOT classify as fainting or choking. Those are not valid.\n"
    "- Return ONLY a JSON array with exactly one object: {\"incident_type\", \"confidence\", \"timestamp_seconds\", \"evidence\", \"recommended_action\"}.\n"
    "Allowed incident_type: shoplifting, suspicious_activity, violent_activity, intrusion, none.\n\n"
)


class LocalGemmaClient:
    """
    Local Gemma runtime adapter.
    Expected endpoint format is Ollama-compatible /api/generate.
    """

    def available(self) -> bool:
        return bool(settings.local_gemma_endpoint and settings.local_gemma_model_name)

    def primary_classify(self, multimodal_summary: str) -> list[Incident]:
        """
        Use LoRA-tuned model as primary classifier (same prompt as SFT).
        Call with summary from gemma_agent._build_multimodal_summary(signals).
        """
        if not self.available():
            return []
        prompt = PRIMARY_CLASSIFIER_RULES + f"MULTIMODAL SUMMARY:\n{multimodal_summary}\n\nJSON array:"
        body = {
            "model": settings.local_gemma_model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        req = urllib.request.Request(
            settings.local_gemma_endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as res:
                payload = json.loads(res.read().decode("utf-8"))
            content = str(payload.get("response", "")).strip()
            return self._parse(content)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
            return []

    def refine_incidents(self, signals: dict, baseline: list[Incident]) -> list[Incident]:
        if not self.available():
            return baseline

        prompt = (
            "You are InstaMIND local emergency reasoning model.\n"
            "Given multimodal input and baseline detections, return ONLY JSON array.\n"
            "Fields: incident_type, confidence, timestamp_seconds, evidence, recommended_action.\n"
            "Allowed incident_type: fainting, choking, violent_activity, shoplifting, suspicious_activity, intrusion, none.\n\n"
            f"signals={json.dumps(signals, default=str)}\n"
            f"baseline={json.dumps([x.model_dump() for x in baseline], default=str)}\n"
        )
        body = {
            "model": settings.local_gemma_model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        req = urllib.request.Request(
            settings.local_gemma_endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as res:
                payload = json.loads(res.read().decode("utf-8"))
            content = str(payload.get("response", "")).strip()
            parsed = self._parse(content)
            return parsed or baseline
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
            return baseline

    @staticmethod
    def _parse(content: str) -> list[Incident]:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            return []

        incidents: list[Incident] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                incident_type = IncidentType(str(item.get("incident_type", "none")))
            except ValueError:
                incident_type = IncidentType.none
            incidents.append(
                Incident(
                    incident_type=incident_type,
                    confidence=float(item.get("confidence", 0.0)),
                    timestamp_seconds=float(item.get("timestamp_seconds", 0.0)),
                    evidence=str(item.get("evidence", "Local model reasoning")),
                    recommended_action=str(item.get("recommended_action", "Continue monitoring.")),
                )
            )
        return incidents
