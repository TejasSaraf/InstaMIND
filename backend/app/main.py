import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import AnalyzeResponse
from app.services.frame_stream_analyzer import FrameStreamAnalyzer
from app.services.gemma_agent import IncidentAnalysisAgent
from app.services.notifier import AlertNotifier
from app.services.storage import StorageService

app = FastAPI(title=settings.app_name, version="0.1.0")

# CORS: allow frontend dev servers (localhost/127.0.0.1 any port) and all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

storage = StorageService()
frame_stream_analyzer = FrameStreamAnalyzer()
agent = IncidentAnalysisAgent()
notifier = AlertNotifier(storage=storage)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "classifier": "shoplifting_only",
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


@app.post("/api/v1/analyze/upload", response_model=AnalyzeResponse)
async def analyze_uploaded_video(file: UploadFile = File(...)) -> AnalyzeResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    if Path(file.filename).suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload.")

    try:
        saved_path = storage.save_upload(file.filename, content)

        start = time.perf_counter()
        signal_bundle = frame_stream_analyzer.analyze(saved_path)
        total_elapsed_ms = (time.perf_counter() - start) * 1000.0

        frame_latency = signal_bundle["latency"]
        if not frame_latency.get("met_target", False):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Frame-by-frame latency target not met (<100ms). "
                    "Reduce input resolution/fps or increase hardware capacity."
                ),
            )

        effective_latency_ms = float(frame_latency["p95_ms"])
        signals = {
            "video": signal_bundle["video"],
            "pose": signal_bundle["pose"],
            "audio": signal_bundle["audio"],
            "latency": {
                **frame_latency,
                "total_analysis_ms": total_elapsed_ms,
            },
        }

        report = agent.analyze(source_filename=file.filename, signals=signals, processing_time_ms=effective_latency_ms)
        storage.save_report(report.report_id, report.model_dump(mode="json"))
        notifier.notify_if_needed(report)

        return AnalyzeResponse(
            message="Video analyzed successfully with local-first incident agent.",
            report=report,
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


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
        raise HTTPException(status_code=404, detail=f"Report not found: {report_id}") from exc
