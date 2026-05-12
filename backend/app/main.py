from app.services.human_detector import HumanDetector

from app.services.realtime_stream import RealtimeStreamHandler

from app.services.user_store import UserStore

from app.services.storage import StorageService

from app.services.google_auth_service import GoogleAuthService

from app.services.local_gemma_client import LocalGemmaClient

from app.services.video_normalize import ensure_decodable_video

from app.services.llamacpp_vision_client import LlamaCppVisionClient

from app.schemas import AnalyzeResponse, AuthStatusResponse, GoogleAuthRequest, GoogleAuthResponse

from app.config import settings

from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, HTTPException, WebSocket

import os

import time

from pathlib import Path



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"





try:

    from app.services.autonomous_response_agent import AutonomousResponseAgent

except ModuleNotFoundError:

    class AutonomousResponseAgent:

        TOOL_SPECS: list = []



app = FastAPI(title=settings.app_name, version="0.1.0")



app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

    expose_headers=["*"],

)



storage = StorageService()

llama_vision = LlamaCppVisionClient()

local_gemma_client = LocalGemmaClient()

google_auth_service = GoogleAuthService()

user_store = UserStore()

human_detector = HumanDetector()





@app.get("/health")

def health() -> dict:

    local_available = local_gemma_client.available()

    return {

        "status": "ok",

        "app": settings.app_name,

        "classifier": settings.model_mode,

        "local_gemma": {

            "available": local_available,

            "runtime": settings.local_gemma_runtime,

            "model_dir": settings.local_gemma_model_dir,

            "endpoint": settings.local_gemma_endpoint,

            "model_name": settings.local_gemma_model_name,

            "required": settings.require_local_gemma,

        },

        "positioning": {

            "emergency_detection_target": f"<{settings.emergency_latency_target_ms}ms",

            "video_never_leaves_device": settings.video_never_leaves_device,

            "offline_capable": settings.offline_mode,

        },

    }





@app.get("/api/v1/positioning")

def positioning() -> dict:

    return {

        "statement": (

            "InstaMIND is not just a surveillance system. "

            "It is an on-device emergency response agent designed for scenarios where: "

            "latency must be near-zero, data cannot leave the device, and systems must work offline."

        ),

        "critical_requirements": [

            "Emergency detection must happen in <100ms",

            "Video never leaves the device",

            "Works without internet, still detects and alerts locally",

        ],

        "primary_use_cases": ["Stores", "Schools", "Public Spaces", "Office"],

    }





@app.get("/api/v1/agentic/tools")

def get_agentic_tools() -> dict:

    from app.services.function_gemma import ANALYSIS_TOOLS

    return {

        "enabled": settings.function_gemma_enabled,

        "function_model": settings.function_gemma_model_name or settings.local_gemma_model_name,

        "max_steps": settings.agentic_max_steps,

        "analysis_tools": ANALYSIS_TOOLS,

        "response_tools": AutonomousResponseAgent.TOOL_SPECS,

    }





@app.post("/api/v1/detect/frame")

async def detect_humans_frame(request: dict) -> dict:

    """Detect humans in a single base64-encoded JPEG frame.

    Request body: {"frame": "<base64 JPEG>"}
    Returns: {"detections": [{"x","y","w","h","confidence"}], "person_count", "inference_ms"}
    """

    frame_b64 = request.get("frame", "")

    if not frame_b64:

        raise HTTPException(

            status_code=400, detail="Missing 'frame' (base64 JPEG).")

    if not human_detector.available():

        raise HTTPException(

            status_code=503, detail="YOLOv4-tiny model files not found. See server logs.")

    return human_detector.detect_frame_b64(frame_b64)





@app.post("/api/v1/analyze/frame")

async def analyze_single_frame(request: dict) -> dict:

    """Classify a single base64-encoded JPEG frame via on-device Gemma 3.

    Request body: {"frame": "<base64 JPEG>", "timestamp": 3.0}
    Returns: {"incident_type", "confidence", "timestamp_seconds", "evidence", "recommended_action"}
    """

    frame_b64 = request.get("frame", "")

    timestamp = float(request.get("timestamp", 0.0))

    if not frame_b64:

        raise HTTPException(

            status_code=400, detail="Missing 'frame' (base64 JPEG).")

    result = llama_vision.classify_frame(frame_b64, timestamp)

    return result





@app.get("/api/v1/reports")

def list_reports() -> dict:

    try:

        return {"reports": storage.list_reports()}

    except Exception:

        return {"reports": []}





@app.get("/api/v1/reports/{report_id}")

def get_report(report_id: str) -> dict:

    try:

        return storage.load_report(report_id)

    except FileNotFoundError as exc:

        raise HTTPException(

            status_code=404, detail=f"Report not found: {report_id}") from exc





@app.post("/api/v1/auth/google", response_model=GoogleAuthResponse)

def google_authenticate(payload: GoogleAuthRequest) -> GoogleAuthResponse:

    try:

        claims = google_auth_service.verify_id_token(payload.id_token)

        user = user_store.upsert_google_user(claims)

        return GoogleAuthResponse(message="Google authentication successful.", user=user)

    except (ValueError, RuntimeError) as exc:

        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:

        err_msg = str(exc).lower()

        if "google" in err_msg or "token" in err_msg or "auth" in err_msg:

            raise HTTPException(

                status_code=400, detail=f"Authentication error: {str(exc)}") from exc



        import traceback

        traceback.print_exc()

        raise HTTPException(

            status_code=500, detail=f"Google authentication failed: {str(exc)}") from exc





@app.get("/api/v1/auth/users/{user_id}", response_model=AuthStatusResponse)

def get_authenticated_user(user_id: str) -> AuthStatusResponse:

    try:

        user = user_store.get_user_by_id(user_id)

        if not user:

            return AuthStatusResponse(logged_in=False, user=None)

        return AuthStatusResponse(logged_in=True, user=user)

    except ValueError as exc:

        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:

        raise HTTPException(

            status_code=500, detail=f"Auth lookup failed: {str(exc)}") from exc





@app.websocket("/api/v1/stream")

async def websocket_stream_endpoint(websocket: WebSocket):

    handler = RealtimeStreamHandler(

        fps_target=2,

        max_dimension=640,

        scene_change_threshold=30.0,

        keyframes_only_save=True,

        emit_interval_s=3.0,

    )

    await handler.handle_websocket(websocket)
