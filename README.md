# ICU Guardian

AI-assisted ICU monitoring platform with a real-time web dashboard, FastAPI backend, computer vision safety pillars, alerting pipeline, and session-based historical analytics.

---

## 1) What This Project Is

ICU Guardian is a full-stack monitoring system designed to support ICU operations by combining:

- Real-time vitals simulation and risk scoring
- Computer vision-based bedside safety monitoring
- Alert persistence and operator-facing alert feed
- Historical session storage and export
- Clinical PDF report generation from session telemetry

This repository includes both:

- Modern stack: FastAPI backend + React (Vite + TypeScript) frontend
- Legacy fallback: Streamlit dashboard

---

## 2) Core Capabilities

- Real-time dashboard with WebSocket updates
- Four vision safety pillars:
  - Self-harm / tube interference
  - Agitation detection
  - Distress proxy detection
  - Fall / bed-exit detection
- Smart alert throttling to reduce alert spam
- Optional Telegram alert notifications
- Session lifecycle management and historical analytics
- CSV export of complete session data
- Clinical PDF report generation from live session data

---

## 3) Architecture Overview

### Runtime Components

1. Frontend (React)
	- Displays live vitals, alert feed, pillar status, and controls.
2. Backend (FastAPI)
	- Serves REST endpoints and WebSocket events.
	- Runs background loops for vitals/alert broadcasting.
3. Vision Process (Python + OpenCV + MediaPipe)
	- Runs as a separate process started/stopped by backend endpoints.
4. Data Layer (SQLite)
	- Stores sessions, vitals, alerts, and system events.
5. Alert Integration
	- Shared alert state file and optional Telegram push.

### Event/Data Flow

1. Vision engine evaluates camera frames and updates alert state.
2. Backend reads alert state and broadcasts alerts_update events.
3. Backend generates vitals, computes ML risk, stores to DB, and broadcasts vitals_update events.
4. Frontend receives updates via WebSocket and renders dashboard in real time.

---

## 4) Tech Stack

### Backend

- FastAPI
- Uvicorn
- WebSockets
- Pydantic
- SQLite
- Scikit-learn
- Psutil

### Vision / AI

- OpenCV
- MediaPipe
- NumPy
- DeepFace (optional path currently disabled in main vision loop)

### Frontend

- React 18
- TypeScript
- Vite
- Tailwind CSS
- TanStack Query
- Recharts
- Framer Motion
- jsPDF

---

## 5) Repository Layout

```text
ICU-Guardian/
├─ backend/                 # FastAPI API server and schemas
├─ engine/                  # Core services: alerts, DB, simulation, model inference
├─ pillars/                 # Vision detection pipelines and CV helpers
├─ frontend/                # React dashboard and landing page
├─ config/                  # Vision configuration (YAML)
├─ data/                    # Runtime DB and sample data
├─ models/                  # Trained ML model artifacts
├─ logs/                    # Runtime logs
├─ dashboard.py             # Legacy Streamlit dashboard
├─ train_model.py           # ML model training utility
├─ check_db.py              # DB inspection utility
├─ setup.ps1 / setup.sh     # Dependency setup scripts
└─ start-backend.ps1 / start-frontend.ps1
```

---

## 6) Prerequisites

- Python 3.10+ (recommended)
- Node.js 18+ and npm
- Webcam for vision pipeline testing
- Windows PowerShell or Bash shell

---

## 7) Quick Start

### Option A: Scripted Setup

Windows:

```powershell
.\setup.ps1
```

Linux/macOS:

```bash
chmod +x setup.sh
./setup.sh
```

### Option B: Manual Setup

Backend dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Frontend dependencies:

```bash
cd ../frontend
npm install
```

---

## 8) Running The System

### Start Backend

```powershell
python backend/main.py
```

Backend default URL: http://localhost:8000

### Start Frontend

```powershell
cd frontend
npm run dev
```

Frontend default URL: http://localhost:5173

### Legacy Streamlit Dashboard (Optional)

```bash
streamlit run dashboard.py
```

---

## 9) Environment Variables

Create a .env file at repository root (or set variables in your shell).

### Core

- LOG_LEVEL=INFO
- ENVIRONMENT=production

### API Protection (optional)

- ICU_API_KEY=your_api_key

When ICU_API_KEY is set, control endpoints require header:

- X-API-Key: your_api_key

### Telegram Alerts (optional)

- TELEGRAM_BOT_TOKEN=your_bot_token
- TELEGRAM_CHAT_ID=your_chat_id

### Camera Selection (optional)

- ICU_CAMERA_INDEX=0

---

## 10) API Reference

### Health and Status

- GET /api/health
- GET /api/system/status

### Vision Control

- POST /api/vision/start
- POST /api/vision/stop
- GET /api/vision/status

### Monitoring Control

- POST /api/monitoring/start
- POST /api/monitoring/stop

### Live Data

- GET /api/vitals/current
- GET /api/alerts/status

### History and Sessions

- GET /api/history/vitals?limit=100
- GET /api/history/alerts?pillar=name&limit=100
- GET /api/history/stats
- POST /api/alerts/{alert_id}/acknowledge
- POST /api/session/new?patient_id=default
- GET /api/sessions/{session_id}/export

### WebSocket

- WS /ws

Incoming event types to client:

- connection_established
- vitals_update
- alerts_update
- system_event
- ping

Client keepalive message:

- { "type": "ping" }

---

## 11) Vision Pillars

Primary runtime script:

- pillars/self_harm.py

Pillars implemented:

1. Self-harm / tube interference
	- Hand landmark enters critical zone near face.
2. Agitation
	- Sustained elevated motion from upper-body landmark tracking.
3. Distress proxy
	- Mouth aspect ratio persistence threshold.
4. Fall / bed exit
	- Hip landmark exits configured safe zone.

Safety features:

- Privacy blur of face when no active danger
- Audible local beep on critical states
- Smart threshold + cooldown alert handling

Configuration source:

- config/vision.yaml

---

## 12) Data Persistence

SQLite database location:

- data/icu_guardian.db

Tables:

- sessions
- vitals
- alerts
- system_events

Utility:

```bash
python check_db.py
```

---

## 13) ML Model Workflow

Model file expected by backend:

- models/psychosis_model.pkl

Train or regenerate model:

```bash
python train_model.py
```

Training script can generate synthetic sample data when dataset is missing.

---

## 14) Frontend Features

- Landing page with product narrative and architecture visuals
- Dashboard command center with:
  - Connection status
  - Vision and monitoring controls
  - Live metric cards
  - Trend charts
  - Pillar status grid
  - Alert feed
  - Clinical PDF report download

---

## 15) Telegram Setup

Detailed setup guide:

- TELEGRAM_SETUP.md

Optional helper scripts:

- utils/setup_telegram.py
- utils/diagnose_network.py

---

## 16) Common Troubleshooting

### Backend fails to start

- Ensure backend dependencies are installed.
- Confirm Python version compatibility.
- Check whether model file exists under models/.

### Frontend cannot connect to backend

- Verify backend is running on port 8000.
- Verify frontend is running on port 5173.
- Check browser console for WebSocket errors.

### Vision process does not detect camera

- Run camera diagnostics:

```bash
python utils/check_cameras.py
python test_camera.py
```

- Ensure camera permission is enabled in OS settings.
- Set ICU_CAMERA_INDEX appropriately.

### No Telegram alerts

- Verify TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID values.
- Run network diagnostics:

```bash
python utils/diagnose_network.py
```

---

## 17) Security Notes

- Do not commit real secrets in source files.
- Prefer environment variables for credentials and API keys.
- Use HTTPS/WSS and restricted CORS in production.
- Add authentication and role-based access before clinical deployment.

---

## 18) Development Notes

- Backend structured logging includes request IDs.
- Frontend uses React Query polling for status and WebSocket for high-frequency updates.
- Streamlit app is kept as a fallback/legacy interface and should be maintained only if required.

---

## 19) Suggested Next Enhancements

1. Add full auth (JWT/OAuth2) and role-based controls.
2. Add comprehensive automated test coverage.
3. Externalize all endpoint URLs to environment config on frontend.
4. Containerize backend + frontend with a production-ready compose stack.
5. Add CI checks for lint, type-check, tests, and dependency scanning.

---

## 20) License and Clinical Disclaimer

This project is a technical prototype and decision-support system.
It is not a replacement for clinical judgment, hospital policy, or regulated medical devices.
