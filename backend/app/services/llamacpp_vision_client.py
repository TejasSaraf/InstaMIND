"""
Lightweight client for llama.cpp server with vision support.
Talks to the OpenAI-compatible /v1/chat/completions endpoint
exposed by `llama-server --mmproj`.
"""



import json

import logging

import urllib.request

from typing import Any



from app.config import settings



logger = logging.getLogger(__name__)



SYSTEM_PROMPT = (

    "You are a surveillance security analyst. "

    "Classify incidents as: fighting, robbery, shoplifting, shooting, fainting, normal. "

    "Return ONLY valid JSON with: incident_type, confidence, timestamp_seconds, evidence, recommended_action."

)





class LlamaCppVisionClient:

    """Send frames to a running llama-server with --mmproj for vision inference."""



    def __init__(self) -> None:

        self.endpoint = settings.llamacpp_vision_endpoint



    def available(self) -> bool:

        """Quick liveness check against llama-server /health."""

        try:

            base = self.endpoint.rsplit("/v1/", 1)[0]

            req = urllib.request.Request(f"{base}/health", method="GET")

            with urllib.request.urlopen(req, timeout=3) as res:

                return res.status == 200

        except Exception:

            return False



    def classify_frame(

        self,

        frame_b64: str,

        timestamp: float = 0.0,

        timeout: int | None = None,

    ) -> dict[str, Any]:

        """Classify a single base64-encoded JPEG frame."""

        return self.classify_frames([frame_b64], [timestamp], timeout)



    def classify_frames(

        self,

        frames_b64: list[str],

        timestamps: list[float],

        timeout: int | None = None,

    ) -> dict[str, Any]:

        """Classify one or more frames via llama-server vision endpoint."""

        if timeout is None:

            timeout = settings.llamacpp_timeout_seconds



        ts_str = ", ".join(f"{t:.1f}s" for t in timestamps)

        user_text = (

            f"Analyze these {len(frames_b64)} surveillance frame(s) "

            f"at timestamps [{ts_str}]. "

            "Classify the security incident. Return JSON: "

            "{incident_type, confidence, timestamp_seconds, evidence, recommended_action}."

        )



        content_parts: list[dict[str, Any]] = [

            {"type": "text", "text": user_text}]

        for b64 in frames_b64:

            content_parts.append({

                "type": "image_url",

                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},

            })



        body = {

            "messages": [

                {"role": "system", "content": SYSTEM_PROMPT},

                {"role": "user", "content": content_parts},

            ],

            "temperature": 0.1,

            "max_tokens": 1024,

            "stop": ["<end_of_turn>", "<|im_end|>", "<|eot_id|>"],

        }



        data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(

            self.endpoint,

            data=data,

            headers={"Content-Type": "application/json"},

            method="POST",

        )

        try:

            with urllib.request.urlopen(req, timeout=timeout) as res:

                payload = json.loads(res.read().decode("utf-8"))

            raw = payload["choices"][0]["message"]["content"].strip()

            return self._parse_response(raw, timestamps)

        except Exception as e:

            logger.warning("[llamacpp] inference failed: %s", e)

            return {

                "incident_type": "normal",

                "confidence": 0.0,

                "timestamp_seconds": timestamps[0] if timestamps else 0.0,

                "evidence": f"Model inference failed: {e}",

                "recommended_action": "Retry analysis.",

            }



    @staticmethod

    def _parse_response(raw: str, timestamps: list[float]) -> dict[str, Any]:

        """Parse the model's JSON response, with fallback."""

        text = raw.strip()



        for marker in ("<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|im_start|>"):

            idx = text.find(marker)

            if idx != -1:

                text = text[:idx].strip()



        if text.startswith("```"):

            text = text.strip("`")

            if text.lower().startswith("json"):

                text = text[4:].strip()



        try:

            result = json.loads(text)

            if isinstance(result, list):

                result = result[0] if result else {}

            if isinstance(result, dict):

                return result

        except json.JSONDecodeError:

            pass



        extracted = LlamaCppVisionClient._extract_first_json_object(text)

        if extracted:

            try:

                result = json.loads(extracted)

                if isinstance(result, dict):

                    return result

            except json.JSONDecodeError:

                pass



        start = text.find("{")

        if start != -1:

            salvaged = text[start:]



            if salvaged.count('"') % 2 != 0:

                salvaged += '"'



            salvaged += "}"

            try:

                return json.loads(salvaged)

            except json.JSONDecodeError:

                pass



        logger.warning(

            "[llamacpp] Could not parse model response: %s", raw[:300])

        return {

            "incident_type": "normal",

            "confidence": 0.0,

            "timestamp_seconds": timestamps[0] if timestamps else 0.0,

            "evidence": f"Could not parse model response: {raw[:200]}",

            "recommended_action": "Retry analysis.",

        }



    @staticmethod

    def _extract_first_json_object(text: str) -> str | None:

        """Extract the first complete {...} JSON object using brace depth tracking."""

        start = text.find("{")

        if start == -1:

            return None

        depth = 0

        in_string = False

        escape_next = False

        for i in range(start, len(text)):

            c = text[i]

            if escape_next:

                escape_next = False

                continue

            if c == "\\" and in_string:

                escape_next = True

                continue

            if c == '"' and not escape_next:

                in_string = not in_string

                continue

            if in_string:

                continue

            if c == "{":

                depth += 1

            elif c == "}":

                depth -= 1

                if depth == 0:

                    return text[start: i + 1]

        return None
