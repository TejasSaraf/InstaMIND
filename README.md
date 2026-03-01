# InstaMIND

**On-device video intelligence for retail and security.** InstaMIND analyzes video (upload or stream) and detects incidents such as shoplifting in real time—with sub-100ms latency targets and no video leaving the device.

---

## Features

- **On-device analysis** — Video stays on your machine; no cloud upload required
- **Fast-path detection** — TensorFlow pose/event model for low-latency classification (shoplifting, suspicious activity, none)
- **Optional LLM layer** — LoRA-tuned small model for human-readable evidence and recommendations (served locally via PEFT or Ollama)
- **Privacy-first** — Works offline; suitable for sensitive environments (retail, schools, offices)

---

## Tech stack

| Layer      | Stack |
|-----------|--------|
| Frontend  | React 19, Vite, TypeScript, Tailwind CSS |
| Backend   | FastAPI (Python), Uvicorn |
| ML        | TensorFlow (pose/event classifier), optional PEFT/LLM (e.g. Qwen2) for evidence text |
| Storage   | Local filesystem (uploads, reports, alerts) |

---

## Quick start

### Prerequisites

- **Node.js** 18+ (for frontend)
- **Python** 3.10+ (for backend)
- **Optional:** Ollama or the project’s PEFT server for local LLM evidence

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # edit .env if needed (see backend/.env.example)
mkdir -p data/uploads data/reports data/alerts
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Backend runs at **http://localhost:8000**. Health check: `curl http://localhost:8000/health`

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at **http://localhost:5173** (or the port Vite prints). Set `VITE_API_BASE_URL=http://localhost:8000` in `frontend/.env` if the API is on another host.

### 3. Optional: local LLM (evidence text)

To use the fine-tuned model for evidence and recommendations:

**Option A — PEFT server (no Ollama):**

```bash
cd backend
pip install torch peft transformers accelerate
python scripts/serve_peft_model.py --base-model Qwen/Qwen2-0.5B-Instruct --adapter-dir ./adapter --port 11434
```

**Option B — Ollama:** install Ollama, pull a base model, then use `app/training/export_to_ollama.py` if you have an adapter.

In `backend/.env` set:

- `MODEL_MODE=local_gemma`
- `LOCAL_GEMMA_ENDPOINT=http://127.0.0.1:11434/api/generate`
- `LOCAL_GEMMA_MODEL_NAME=instamind-shoplifting` (or your model name)

---

## Project structure

```
instaMIND/
├── backend/                 # FastAPI app
│   ├── app/
│   │   ├── main.py          # Routes, upload, reports
│   │   ├── config.py        # Settings
│   │   ├── schemas.py       # API models
│   │   ├── services/        # Analysis, pose detector, LLM client, storage
│   │   ├── training/        # Fine-tuning and dataset scripts (optional)
│   │   └── models/          # Pose event detector weights + labels
│   ├── scripts/
│   │   ├── serve_peft_model.py   # Local PEFT server (Ollama-compatible API)
│   │   └── GPU_FINETUNE_RUNBOOK.md
│   └── requirements.txt
├── frontend/                # React app
│   ├── src/
│   │   ├── pages/           # UploadVideo, etc.
│   │   └── components/      # Navbar, Hero, InputBox, ResultCard, …
│   └── package.json
├── presentation/            # 2-minute pitch slide deck (HTML)
├── PRESENTATION_2MIN.md     # Slide content + timing for PowerPoint/Google Slides
└── README.md                # This file
```

---

## API overview

| Method | Path | Description |
|--------|------|-------------|
| GET    | `/health` | Health check |
| POST   | `/api/v1/analyze/upload` | Upload video, run analysis, return report |
| GET    | `/api/v1/reports` | List reports |
| GET    | `/api/v1/reports/{id}` | Get report by ID |

---

## Training and fine-tuning

- **Pose/event model:** See `backend/app/training/` (e.g. `train_pose_event_model.py`) and dataset under `backend/app/dataset/`.
- **LLM fine-tuning (QLoRA):** See `backend/scripts/GPU_FINETUNE_RUNBOOK.md` and `backend/app/training/finetune_gemma_qlora.py` for running on a GPU VM and exporting the adapter for local use.

---

## License

Private / internal use unless otherwise specified.
