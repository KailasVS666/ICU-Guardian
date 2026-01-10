"""
ICU Guardian FastAPI Backend
Production-grade API server with WebSocket support for real-time monitoring
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import asyncio
import json
import os
import psutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import logging
import uuid

# Add parent directory to path for imports
BASE_DIR = Path(__file__).parent.parent
STATUS_FILE = BASE_DIR / "alert_status.json"
sys.path.insert(0, str(BASE_DIR))

# Import Pydantic models
try:
    from backend.models import (
        VitalsResponse, VitalsHistoryResponse, AlertStatusResponse, AlertHistoryResponse,
        SystemStatusResponse, HealthCheckResponse, MessageResponse, SessionResponse,
        CreateSessionRequest, AcknowledgeAlertRequest, ErrorResponse, ErrorDetail
    )
    from backend.logging_config import setup_logging, set_request_id, get_request_id
except ModuleNotFoundError:
    from models import (
        VitalsResponse, VitalsHistoryResponse, AlertStatusResponse, AlertHistoryResponse,
        SystemStatusResponse, HealthCheckResponse, MessageResponse, SessionResponse,
        CreateSessionRequest, AcknowledgeAlertRequest, ErrorResponse, ErrorDetail
    )
    from logging_config import setup_logging, set_request_id, get_request_id

# Setup structured logging
setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"), structured=True)
logger = logging.getLogger(__name__)

# Import existing modules
from engine.simulation import generate_patient_data
from engine.predict_model import load_trained_model, predict_psychosis
from engine.database import get_db
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Global state
class ApplicationState:
    def __init__(self):
        self.vision_process = None
        self.ml_model = None
        self.scaler = MinMaxScaler()
        self.scaler.fit([[50, 80, 0, -5], [150, 100, 5, 4]])
        self.connected_clients: List[WebSocket] = []
        self.monitoring_active = False
        self.system_start_time = datetime.now()
        self.current_session_id = None
        self.db = get_db()
        
    def load_model(self):
        """Load ML model on startup"""
        try:
            self.ml_model = load_trained_model()
            logger.info("✅ ML Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load ML model: {e}")
    
    def ensure_session(self):
        """Ensure there's an active session"""
        if self.current_session_id is None:
            self.current_session_id = self.db.get_active_session()
            if self.current_session_id is None:
                self.current_session_id = self.db.create_session()
        return self.current_session_id

app_state = ApplicationState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting ICU Guardian API Server...")
    app_state.load_model()
    
    # Initialize database and create session
    app_state.ensure_session()
    logger.info(f"📊 Database initialized with session {app_state.current_session_id}")
    
    # Log startup event
    app_state.db.log_event("system_startup", "ICU Guardian API Server started")
    
    # Start background tasks
    vitals_task = asyncio.create_task(broadcast_vitals_loop())
    alerts_task = asyncio.create_task(broadcast_alerts_loop())
    logger.info("📊 Background broadcast tasks started")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down ICU Guardian API Server...")
    vitals_task.cancel()
    alerts_task.cancel()
    
    # Log shutdown event
    app_state.db.log_event("system_shutdown", "ICU Guardian API Server stopped")
    
    if app_state.vision_process:
        try:
            parent = psutil.Process(app_state.vision_process.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
        except:
            pass

# Initialize FastAPI app
app = FastAPI(
    title="ICU Guardian API",
    description="Real-time patient monitoring and AI analysis",
    version="2.0.0",
    lifespan=lifespan
)

# ============================================
# EXCEPTION HANDLERS
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Standardized error response for HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail
            },
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Standardized error response for validation errors"""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": errors
            },
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# REQUEST ID MIDDLEWARE
# ============================================
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """
    Generate and propagate request ID for tracing.
    Adds X-Request-ID header to response.
    """
    # Check if request already has ID (from load balancer/proxy)
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    set_request_id(request_id)
    
    # Log incoming request
    logger.info(
        f"Incoming request",
        extra={"extra_fields": {
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "unknown"
        }}
    )
    
    # Process request
    response = await call_next(request)
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    # Log response
    logger.info(
        f"Request completed",
        extra={"extra_fields": {
            "status_code": response.status_code
        }}
    )
    
    return response

# ============================================
# AUTH CONFIG
# ============================================
API_KEY = os.getenv("ICU_API_KEY") or os.getenv("API_KEY")
API_KEY_HEADER = "X-API-Key"


async def verify_api_key(request: Request):
    """Simple header-based API key guard for control endpoints."""
    if not API_KEY:
        return  # No key configured; allow for dev/hackathon
    provided = request.headers.get(API_KEY_HEADER)
    if provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ============================================
# WEBSOCKET CONNECTION MANAGER
# ============================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"✅ Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"❌ Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)

manager = ConnectionManager()

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "ICU Guardian API",
        "version": "2.0.0",
        "uptime": str(datetime.now() - app_state.system_start_time),
        "model_loaded": app_state.ml_model is not None
    }

@app.get("/api/health", response_model=HealthCheckResponse)
async def health_check():
    """Detailed health check"""
    # Check database connectivity
    db_connected = False
    try:
        app_state.db.get_connection().close()
        db_connected = True
    except Exception:
        pass
    
    uptime_secs = (datetime.now() - app_state.system_start_time).total_seconds()
    status = "healthy"
    
    if not db_connected or not app_state.ml_model:
        status = "degraded"
    
    return HealthCheckResponse(
        status=status,
        vision_active=is_vision_running(),
        ml_model_loaded=app_state.ml_model is not None,
        connected_clients=len(manager.active_connections),
        monitoring_active=app_state.monitoring_active,
        uptime_seconds=uptime_secs,
        database_connected=db_connected
    )

@app.get("/api/system/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get current system status"""
    uptime = datetime.now() - app_state.system_start_time
    hours, remainder = divmod(int(uptime.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return SystemStatusResponse(
        uptime=f"{hours:02d}:{minutes:02d}:{seconds:02d}",
        uptime_seconds=uptime.total_seconds(),
        vision_running=is_vision_running(),
        monitoring_active=app_state.monitoring_active,
        connected_clients=len(manager.active_connections),
        system_start_time=app_state.system_start_time.isoformat()
    )

@app.get("/api/alerts/status")
async def get_alert_status():
    """Get current alert status from vision system"""
    try:
        if STATUS_FILE.exists():
            try:
                with open(STATUS_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Partial alert status detected; returning empty state")
        return {"pillars": {}}
    except Exception as e:
        logger.error(f"Error reading alert status: {e}")
        return {"pillars": {}}

@app.post("/api/vision/start")
async def start_vision_system(_=Depends(verify_api_key)):
    """Start the vision monitoring system"""
    if app_state.vision_process is None or not is_vision_running():
        try:
            # Use virtual environment Python if available
            venv_python = BASE_DIR / ".venv" / "Scripts" / "python.exe"
            python_path = venv_python if venv_python.exists() else Path(sys.executable)
            
            # Run comprehensive self_harm detection (includes all features)
            script_path = BASE_DIR / "pillars" / "self_harm.py"
            
            # Create logs directory
            log_dir = BASE_DIR / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "vision_system.log"
            
            import os
            env = os.environ.copy()
            env['PYTHONPATH'] = str(BASE_DIR)
            env['VISION_LOG_FILE'] = str(log_file)
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # Open log file for output
            log_handle = open(log_file, 'a', encoding='utf-8')
            log_handle.write(f"\n{'='*50}\n")
            log_handle.write(f"Vision System Start: {datetime.now().isoformat()}\n")
            log_handle.write(f"Python: {python_path}\n")
            log_handle.write(f"{'='*50}\n\n")
            log_handle.flush()
            
            app_state.vision_process = subprocess.Popen(
                [str(python_path), str(script_path)],
                cwd=str(BASE_DIR),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )
            
            logger.info(f"Vision system started with PID {app_state.vision_process.pid}. Logs: {log_file}")
            
            # Broadcast to all connected clients
            await manager.broadcast({
                "type": "system_event",
                "event": "vision_started",
                "timestamp": datetime.now().isoformat()
            })
            
            return {"status": "success", "message": "Vision system started"}
        except Exception as e:
            logger.error(f"Failed to start vision system: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return {"status": "already_running", "message": "Vision system is already active"}

@app.post("/api/vision/stop")
async def stop_vision_system(_=Depends(verify_api_key)):
    """Stop the vision monitoring system"""
    if app_state.vision_process and is_vision_running():
        try:
            parent = psutil.Process(app_state.vision_process.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            parent.wait(timeout=5)
        except Exception as e:
            logger.error(f"Error stopping vision system: {e}")
        
        app_state.vision_process = None
        
        # Broadcast to all connected clients
        await manager.broadcast({
            "type": "system_event",
            "event": "vision_stopped",
            "timestamp": datetime.now().isoformat()
        })
        
        return {"status": "success", "message": "Vision system stopped"}
    
    return {"status": "not_running", "message": "Vision system is not active"}

@app.get("/api/vision/status")
async def get_vision_status():
    """Get vision system status"""
    return {
        "running": is_vision_running(),
        "pid": app_state.vision_process.pid if app_state.vision_process else None
    }

@app.post("/api/monitoring/start")
async def start_monitoring(_=Depends(verify_api_key)):
    """Start AI monitoring loop"""
    app_state.monitoring_active = True
    await manager.broadcast({
        "type": "system_event",
        "event": "monitoring_started",
        "timestamp": datetime.now().isoformat()
    })
    return {"status": "success", "message": "Monitoring started"}

@app.post("/api/monitoring/stop")
async def stop_monitoring(_=Depends(verify_api_key)):
    """Stop AI monitoring loop"""
    app_state.monitoring_active = False
    await manager.broadcast({
        "type": "system_event",
        "event": "monitoring_stopped",
        "timestamp": datetime.now().isoformat()
    })
    return {"status": "success", "message": "Monitoring stopped"}

@app.get("/api/vitals/current")
async def get_current_vitals():
    """Get current patient vitals and AI prediction"""
    try:
        vitals = generate_patient_data()
        
        if app_state.ml_model:
            input_scaled = app_state.scaler.transform(np.array([[
                vitals['HR_Avg'], vitals['SpO2_Min'],
                vitals['Sleep_Score'], vitals['RASS_Score']
            ]]))
            prediction, probability = predict_psychosis(app_state.ml_model, input_scaled)
        else:
            prediction, probability = 0, 0.0
        
        # Save to database
        session_id = app_state.ensure_session()
        vitals_data = {
            **vitals,
            'risk_probability': float(probability)
        }
        app_state.db.save_vitals(session_id, vitals_data)
        
        return {
            "vitals": vitals,
            "prediction": {
                "risk_level": int(prediction),
                "probability": float(probability),
                "risk_category": "critical" if probability > 0.6 else "warning" if probability > 0.4 else "stable"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating vitals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# DATABASE / HISTORY ENDPOINTS
# ============================================

@app.get("/api/history/vitals")
async def get_vitals_history(limit: int = 100):
    """Get historical vitals data"""
    try:
        session_id = app_state.ensure_session()
        history = app_state.db.get_vitals_history(session_id, limit)
        return {"vitals": history, "count": len(history)}
    except Exception as e:
        logger.error(f"Error fetching vitals history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/alerts")
async def get_alerts_history(pillar: str = None, limit: int = 100):
    """Get historical alert data"""
    try:
        session_id = app_state.ensure_session()
        alerts = app_state.db.get_alerts(session_id, pillar, limit=limit)
        return {"alerts": alerts, "count": len(alerts)}
    except Exception as e:
        logger.error(f"Error fetching alerts history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/stats")
async def get_session_stats():
    """Get statistical summary for current session"""
    try:
        session_id = app_state.ensure_session()
        vitals_stats = app_state.db.get_vitals_stats(session_id)
        alert_summary = app_state.db.get_alert_summary(session_id)
        
        return {
            "session_id": session_id,
            "vitals_stats": vitals_stats,
            "alert_summary": alert_summary
        }
    except Exception as e:
        logger.error(f"Error fetching session stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int, acknowledged_by: str = "staff"):
    """Acknowledge an alert"""
    try:
        success = app_state.db.acknowledge_alert(alert_id, acknowledged_by)
        if success:
            return {"status": "success", "message": f"Alert {alert_id} acknowledged"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/session/new")
async def create_new_session(patient_id: str = "default"):
    """Create a new monitoring session"""
    try:
        # End current session if exists
        if app_state.current_session_id:
            app_state.db.end_session(app_state.current_session_id)
        
        # Create new session
        new_session_id = app_state.db.create_session(patient_id)
        app_state.current_session_id = new_session_id
        
        return {
            "status": "success",
            "session_id": new_session_id,
            "patient_id": patient_id
        }
    except Exception as e:
        logger.error(f"Error creating new session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}/export")
async def export_session_data(session_id: int):
    """
    Export complete session data (vitals + alerts) as CSV.
    Returns downloadable file with combined session history.
    """
    try:
        import csv
        from io import StringIO
        from fastapi.responses import StreamingResponse
        
        # Fetch vitals history
        vitals_rows = app_state.db.get_vitals_history(session_id, limit=10000)
        
        # Fetch alerts history
        alerts_rows = app_state.db.get_alerts(session_id, limit=10000)
        
        if not vitals_rows and not alerts_rows:
            raise HTTPException(status_code=404, detail="No data found for this session")
        
        # Create CSV in memory
        output = StringIO()
        
        # Write vitals section
        if vitals_rows:
            output.write("# VITALS DATA\n")
            writer = csv.DictWriter(output, fieldnames=vitals_rows[0].keys())
            writer.writeheader()
            writer.writerows(vitals_rows)
            output.write("\n\n")
        
        # Write alerts section
        if alerts_rows:
            output.write("# ALERTS DATA\n")
            writer = csv.DictWriter(output, fieldnames=alerts_rows[0].keys())
            writer.writeheader()
            writer.writerows(alerts_rows)
        
        # Prepare response
        output.seek(0)
        filename = f"icu_guardian_session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting session data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# WEBSOCKET ENDPOINT
# ============================================

PING_INTERVAL_SECONDS = 30


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "connection_established",
            "timestamp": datetime.now().isoformat(),
            "message": "Connected to ICU Guardian"
        })
        
        # Keep connection alive and listen for messages with keepalive ping
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=PING_INTERVAL_SECONDS)
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"type": "ping"})
                    continue
                except Exception:
                    logger.warning("WebSocket keepalive failed; closing connection")
                    break
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
                
    finally:
        manager.disconnect(websocket)

# ============================================
# BACKGROUND TASKS
# ============================================

async def broadcast_vitals_loop():
    """Background task to broadcast vitals to all connected clients"""
    logger.info("📊 Vitals broadcast loop started")
    while True:
        try:
            # If monitoring is paused, hold off on generating/saving vitals
            if not app_state.monitoring_active:
                await asyncio.sleep(2.5)
                continue

            if len(manager.active_connections) > 0:
                # Generate vitals
                vitals = generate_patient_data()
                
                # Get AI prediction
                if app_state.ml_model:
                    input_scaled = app_state.scaler.transform(np.array([[
                        vitals['HR_Avg'], vitals['SpO2_Min'],
                        vitals['Sleep_Score'], vitals['RASS_Score']
                    ]]))
                    prediction, probability = predict_psychosis(app_state.ml_model, input_scaled)
                else:
                    prediction, probability = 0, 0.0
                
                # Save vitals to database
                try:
                    session_id = app_state.ensure_session()
                    vitals_with_risk = vitals.copy()
                    vitals_with_risk['risk_probability'] = probability
                    app_state.db.save_vitals(
                        session_id=session_id,
                        vitals=vitals_with_risk
                    )
                except Exception as e:
                    logger.error(f"Failed to save vitals to database: {e}")
                
                # Broadcast to all clients
                await manager.broadcast({
                    "type": "vitals_update",
                    "data": {
                        "vitals": vitals,
                        "prediction": {
                            "risk_level": int(prediction),
                            "probability": float(probability),
                            "risk_category": "critical" if probability > 0.6 else "warning" if probability > 0.4 else "stable"
                        }
                    },
                    "timestamp": datetime.now().isoformat()
                })
                logger.debug(f"📡 Broadcast vitals to {len(manager.active_connections)} clients")
                
        except Exception as e:
            logger.error(f"Error in vitals broadcast: {e}")
        
        await asyncio.sleep(2.5)

async def broadcast_alerts_loop():
    """Background task to broadcast alert status to all connected clients"""
    while True:
        if len(manager.active_connections) > 0:
            try:
                if STATUS_FILE.exists():
                    try:
                        with open(STATUS_FILE, 'r') as f:
                            alert_data = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning("Partial alert status detected; sending empty payload")
                        alert_data = {"pillars": {}}
                    
                    await manager.broadcast({
                        "type": "alerts_update",
                        "data": alert_data,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    # Send default empty alerts if file doesn't exist
                    await manager.broadcast({
                        "type": "alerts_update",
                        "data": {"pillars": {}},
                        "timestamp": datetime.now().isoformat()
                    })
            except Exception as e:
                logger.error(f"Error in alerts broadcast: {e}")
        
        await asyncio.sleep(4.0)

# ============================================
# UTILITY FUNCTIONS
# ============================================

def is_vision_running():
    """Check if vision system process is active"""
    if app_state.vision_process:
        try:
            return psutil.pid_exists(app_state.vision_process.pid)
        except:
            return False
    return False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
