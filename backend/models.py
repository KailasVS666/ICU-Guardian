"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from datetime import datetime


# ============================================
# VITALS MODELS
# ============================================

class VitalsData(BaseModel):
    """Patient vitals data"""
    HR_Avg: float = Field(..., ge=0, le=300, description="Average heart rate (BPM)")
    SpO2_Min: float = Field(..., ge=0, le=100, description="Minimum SpO2 (%)")
    Sleep_Score: float = Field(..., ge=0, le=5, description="Sleep quality score")
    RASS_Score: int = Field(..., ge=-5, le=4, description="Richmond Agitation-Sedation Scale")


class PredictionData(BaseModel):
    """AI prediction results"""
    risk_level: int = Field(..., ge=0, le=1, description="Binary risk classification")
    probability: float = Field(..., ge=0, le=1, description="Risk probability")
    risk_category: Literal["stable", "warning", "critical"] = Field(..., description="Risk category")


class VitalsResponse(BaseModel):
    """Response for vitals endpoint"""
    vitals: VitalsData
    prediction: PredictionData
    timestamp: str


class VitalsHistoryResponse(BaseModel):
    """Response for vitals history endpoint"""
    vitals: list[dict]
    count: int


# ============================================
# ALERT MODELS
# ============================================

class AlertStatus(BaseModel):
    """Single pillar alert status"""
    active: bool
    elapsed_seconds: float = 0
    count: int = 0
    last_alert: Optional[str] = None


class AlertStatusResponse(BaseModel):
    """Response for alert status endpoint"""
    pillars: dict[str, AlertStatus]
    last_updated: Optional[str] = None


class AlertHistoryResponse(BaseModel):
    """Response for alert history endpoint"""
    alerts: list[dict]
    count: int


class AcknowledgeAlertRequest(BaseModel):
    """Request to acknowledge an alert"""
    acknowledged_by: str = Field(default="staff", description="Person acknowledging the alert")


# ============================================
# SESSION MODELS
# ============================================

class CreateSessionRequest(BaseModel):
    """Request to create a new session"""
    patient_id: str = Field(default="default", description="Patient identifier")


class SessionResponse(BaseModel):
    """Response for session creation"""
    status: str
    session_id: int
    patient_id: str


# ============================================
# SYSTEM MODELS
# ============================================

class SystemStatusResponse(BaseModel):
    """System status response"""
    uptime: str
    uptime_seconds: float
    vision_running: bool
    monitoring_active: bool
    connected_clients: int
    system_start_time: str


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: Literal["healthy", "degraded", "unhealthy"]
    vision_active: bool
    ml_model_loaded: bool
    connected_clients: int
    monitoring_active: bool
    uptime_seconds: float
    database_connected: Optional[bool] = None


class MessageResponse(BaseModel):
    """Generic message response"""
    status: str
    message: str
    details: Optional[dict] = None


# ============================================
# ERROR MODELS
# ============================================

class ErrorDetail(BaseModel):
    """Error detail structure"""
    code: str
    message: str
    field: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standardized error response"""
    error: ErrorDetail
    timestamp: str
    path: Optional[str] = None
