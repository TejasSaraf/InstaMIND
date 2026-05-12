# instaMIND

<p align="center">
  <a href="https://youtu.be/zAtyOGOH0Uo">
    <img
      src="https://img.youtube.com/vi/zAtyOGOH0Uo/maxresdefault.jpg"
      alt="Watch the instaMIND demo"
      width="100%"
    />
  </a>
</p>

**AI security operators for every camera feed.** instaMIND watches CCTV feeds on-device, detects critical incidents, and routes alerts in under two seconds without uploading surveillance video to the cloud.

---

## Why instaMIND

Security teams have more cameras than attention. A store, school, or public venue can have dozens of feeds, but one human operator still has to notice fights, robberies, medical collapses, shoplifting, or weapons in real time.

instaMIND adds a local AI operator on top of existing camera infrastructure. It watches every feed, detects people and incidents, explains what happened, and helps route the next action from a single dashboard.

---

## What It Does

- **Live camera dashboard**: renders multiple CCTV/demo feeds with camera health, online/offline state, recent incidents, and lightweight human bounding boxes.
- **Realtime detection**: runs human detection on live webcam/WebRTC streams and overlays green bounding boxes continuously.
- **Video upload analysis**: analyzes uploaded videos frame-by-frame and surfaces model-generated scene summaries and key moments.
- **On-device incident classification**: uses local vision inference to classify scenes into security-relevant incident types.
- **Agentic response flow**: supports escalation actions such as warnings, alerts, and continued monitoring.
- **Privacy-first architecture**: video is processed locally; no cloud video upload is required for inference.

---

## Core Capabilities

| Capability | Description |
| --- | --- |
| Incident analysis | Local Gemma 3 / llama.cpp vision pipeline for frame-level incident classification |
| Dashboard monitoring | Multi-feed dashboard with live camera status and incident controls |
| Realtime stream | Browser camera/WebRTC feed with continuous frame detection |
| Video upload | Upload a video and run on-device frame analysis |
| Authentication | Google sign-in with backend token verification and user storage |

---

## Tech Stack

| Layer | Stack |
| --- | --- |
| Frontend | React 19, Vite, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python, Uvicorn |
| Vision detection | OpenCV DNN, YOLOv4-tiny |
| Local vision LLM | llama.cpp / Gemma 3 style local vision client |
| Training | QLoRA / FunctionGemma routing experiments, dataset generation scripts |
| Native acceleration | C++ / pybind11 OpenCV extensions for video frame extraction and analysis |
| Storage | Local filesystem for reports, uploads, model outputs, and generated artifacts |

---

## Project Structure

```text
instaMIND/
├── backend/
│   ├── app/
│   │   ├── main.py                         # FastAPI app and API routes
│   │   ├── config.py                       # Runtime configuration
│   │   ├── orchestrator.py                 # Analysis orchestration
│   │   ├── schemas.py                      # API response/request schemas
│   │   ├── services/
│   │   │   ├── human_detector.py           # YOLO human detection service
│   │   │   ├── llamacpp_vision_client.py   # Local vision inference client
│   │   │   ├── realtime_stream.py          # WebSocket stream handling
│   │   │   ├── user_store.py               # User persistence
│   │   │   └── video_normalize.py          # Video normalization helpers
│   │   └── training/                       # Dataset + fine-tuning utilities
│   ├── cpp_extensions/                     # Native OpenCV / pybind11 extensions
│   ├── scripts/                            # Training and data pipeline scripts
│   └── requirements.txt
├── frontend/
│   ├── public/
│   │   ├── instaMIND_Demo.mov              # README demo video
│   │   └── data-videos -> ../../data/videos
│   ├── src/
│   │   ├── components/                     # Landing page, navbar, UI components
│   │   ├── pages/                          # Dashboard, Realtime, UploadVideo
│   │   └── types.ts
│   └── package.json
├── data/                                   # Local videos and annotations
└── README.md
```

---

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.10+
- OpenCV-compatible environment
- YOLOv4-tiny model files for human detection:
  - `backend/models/yolov4-tiny.cfg`
  - `backend/models/yolov4-tiny.weights`

Download YOLOv4-tiny files:

```bash
mkdir -p backend/models
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg -P backend/models
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights -P backend/models
```

### 1. Start the Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Backend runs at:

```text
http://localhost:8000
```

Health check:

```bash
curl http://localhost:8000/health
```

### 2. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:

```text
http://localhost:5173
```

If the backend runs somewhere else, create `frontend/.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

---

## API Overview

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Backend health and local model status |
| `GET` | `/api/v1/positioning` | Product positioning metadata |
| `POST` | `/api/v1/detect/frame` | Detect humans in one base64 JPEG frame |
| `POST` | `/api/v1/analyze/frame` | Analyze one frame using local vision inference |
| `GET` | `/api/v1/reports` | List locally stored reports |
| `GET` | `/api/v1/reports/{report_id}` | Load one report |
| `POST` | `/api/v1/auth/google` | Verify Google ID token and upsert user |
| `GET` | `/api/v1/auth/users/{user_id}` | Check stored authenticated user |
| `WS` | `/api/v1/stream` | Realtime stream WebSocket endpoint |

---

## Environment Variables

### Backend

Create `backend/.env` as needed for local model and auth settings. Common values:

```env
GOOGLE_OAUTH_CLIENT_ID=...
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
MODEL_MODE=local_gemma
LOCAL_GEMMA_ENDPOINT=http://127.0.0.1:8080
LOCAL_GEMMA_MODEL_NAME=...
```

### Frontend

Create `frontend/.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_GOOGLE_CLIENT_ID=...
```

---

## Training and Model Utilities

The repository includes training and data preparation scripts under:

```text
backend/app/training/
backend/scripts/
backend/scripts/data_pipeline/
```

Current utilities include:

- Function-calling dataset generation for routing response tools
- Gemma 3 / FunctionGemma fine-tuning experiments
- Data parsing, augmentation, balancing, splitting, and verification
- Vision path verification and deployment helpers
- Native C++ video frame extraction and analysis utilities

---

## Build and Verification

Frontend production build:

```bash
npm --prefix frontend run build
```

Backend import/health smoke test:

```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
```

---

## Notes

- The dashboard demo videos are served from `frontend/public/data-videos`, which symlinks to `data/videos`.
- Human detection requires the YOLOv4-tiny model files in `backend/models`.
- Autoplaying demo videos in browsers must be muted, which is why the landing page demo runs muted by default.
- The project is designed around local inference and privacy-preserving security workflows.

---
