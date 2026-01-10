"""
Database module for ICU Guardian
SQLite-based storage for vitals, alerts, and session history
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "icu_guardian.db"

class Database:
    def __init__(self, db_path: str = None):
        """Initialize database connection"""
        self.db_path = db_path or str(DB_PATH)
        self._ensure_db_exists()
        self._create_tables()
    
    def _ensure_db_exists(self):
        """Ensure data directory exists"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Table 1: Patient Sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT DEFAULT 'active',
                notes TEXT
            )
        """)
        
        # Table 2: Vitals History
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vitals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                hr_avg REAL,
                spo2_min REAL,
                sleep_score REAL,
                rass_score REAL,
                psychosis_risk REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        # Table 3: Alert History
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                pillar TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT,
                acknowledged BOOLEAN DEFAULT 0,
                acknowledged_by TEXT,
                acknowledged_at TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        # Table 4: System Events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                description TEXT,
                metadata TEXT
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_vitals_session 
            ON vitals(session_id, timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_session 
            ON alerts(session_id, timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_pillar 
            ON alerts(pillar, timestamp DESC)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Database initialized at {self.db_path}")
    
    # ============================================
    # SESSION MANAGEMENT
    # ============================================
    
    def create_session(self, patient_id: str = "default", notes: str = None) -> int:
        """Create new patient monitoring session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sessions (patient_id, notes)
            VALUES (?, ?)
        """, (patient_id, notes))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"📝 Created session {session_id} for patient {patient_id}")
        return session_id
    
    def get_active_session(self) -> Optional[int]:
        """Get the current active session ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id FROM sessions 
            WHERE status = 'active' 
            ORDER BY start_time DESC 
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        return result['id'] if result else None
    
    def end_session(self, session_id: int):
        """End a monitoring session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE sessions 
            SET end_time = CURRENT_TIMESTAMP, status = 'completed'
            WHERE id = ?
        """, (session_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"🔚 Ended session {session_id}")
    
    # ============================================
    # VITALS STORAGE
    # ============================================
    
    def save_vitals(self, session_id: int, vitals: Dict) -> int:
        """Save vitals record"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO vitals (
                session_id, hr_avg, spo2_min, sleep_score, 
                rass_score, psychosis_risk
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            vitals.get('HR_Avg'),
            vitals.get('SpO2_Min'),
            vitals.get('Sleep_Score'),
            vitals.get('RASS_Score'),
            vitals.get('risk_probability')
        ))
        
        vitals_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return vitals_id
    
    def get_vitals_history(self, session_id: int = None, limit: int = 100) -> List[Dict]:
        """Get vitals history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute("""
                SELECT * FROM vitals 
                WHERE session_id = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (session_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM vitals 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_vitals_stats(self, session_id: int) -> Dict:
        """Get statistical summary of vitals for a session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as record_count,
                AVG(hr_avg) as avg_hr,
                MIN(hr_avg) as min_hr,
                MAX(hr_avg) as max_hr,
                AVG(spo2_min) as avg_spo2,
                MIN(spo2_min) as min_spo2,
                AVG(psychosis_risk) as avg_risk,
                MAX(psychosis_risk) as max_risk
            FROM vitals
            WHERE session_id = ?
        """, (session_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        return dict(result) if result else {}
    
    # ============================================
    # ALERT MANAGEMENT
    # ============================================
    
    def save_alert(self, session_id: int, pillar: str, severity: str, message: str) -> int:
        """Save alert record"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alerts (session_id, pillar, severity, message)
            VALUES (?, ?, ?, ?)
        """, (session_id, pillar, severity, message))
        
        alert_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"🚨 Saved alert {alert_id}: {pillar} - {severity}")
        return alert_id
    
    def get_alerts(self, session_id: int = None, pillar: str = None, 
                   unacknowledged_only: bool = False, limit: int = 100) -> List[Dict]:
        """Get alert history with optional filters"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM alerts WHERE 1=1"
        params = []
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        if pillar:
            query += " AND pillar = ?"
            params.append(pillar)
        
        if unacknowledged_only:
            query += " AND acknowledged = 0"
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def acknowledge_alert(self, alert_id: int, acknowledged_by: str = "staff") -> bool:
        """Mark alert as acknowledged"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE alerts 
            SET acknowledged = 1, 
                acknowledged_by = ?,
                acknowledged_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (acknowledged_by, alert_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if success:
            logger.info(f"✅ Alert {alert_id} acknowledged by {acknowledged_by}")
        
        return success
    
    def get_alert_summary(self, session_id: int) -> Dict:
        """Get alert statistics for a session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                pillar,
                COUNT(*) as total_count,
                SUM(CASE WHEN acknowledged = 1 THEN 1 ELSE 0 END) as acknowledged_count,
                MAX(timestamp) as last_alert
            FROM alerts
            WHERE session_id = ?
            GROUP BY pillar
        """, (session_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return {row['pillar']: dict(row) for row in rows}
    
    # ============================================
    # SYSTEM EVENTS
    # ============================================
    
    def log_event(self, event_type: str, description: str, metadata: Dict = None):
        """Log system event"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO system_events (event_type, description, metadata)
            VALUES (?, ?, ?)
        """, (event_type, description, json.dumps(metadata) if metadata else None))
        
        conn.commit()
        conn.close()
    
    def get_events(self, event_type: str = None, limit: int = 100) -> List[Dict]:
        """Get system events"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if event_type:
            cursor.execute("""
                SELECT * FROM system_events 
                WHERE event_type = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (event_type, limit))
        else:
            cursor.execute("""
                SELECT * FROM system_events 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    # ============================================
    # CLEANUP
    # ============================================
    
    def cleanup_old_data(self, days: int = 30):
        """Delete data older than specified days"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM vitals 
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        """, (days,))
        vitals_deleted = cursor.rowcount
        
        cursor.execute("""
            DELETE FROM alerts 
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        """, (days,))
        alerts_deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        logger.info(f"🧹 Cleaned up {vitals_deleted} vitals and {alerts_deleted} alerts older than {days} days")
        return vitals_deleted, alerts_deleted

# Global database instance
db = Database()

# Convenience functions
def get_db() -> Database:
    """Get database instance"""
    return db
