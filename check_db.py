import sqlite3
from pathlib import Path

db_path = Path("data/icu_guardian.db")

if not db_path.exists():
    print("❌ Database not found!")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print(f"📊 Tables: {tables}\n")

# Check sessions
cursor.execute("SELECT * FROM sessions")
sessions = cursor.fetchall()
print(f"🏥 Sessions ({len(sessions)}):")
for session in sessions:
    print(f"  ID: {session[0]}, Patient: {session[1]}, Start: {session[2]}, End: {session[3]}")

# Check vitals count
cursor.execute("SELECT COUNT(*) FROM vitals")
vitals_count = cursor.fetchone()[0]
print(f"\n❤️ Vitals records: {vitals_count}")

# Show latest vitals if any
if vitals_count > 0:
    cursor.execute("SELECT session_id, timestamp, hr_avg, spo2_min, sleep_score, rass_score, psychosis_risk FROM vitals ORDER BY timestamp DESC LIMIT 3")
    vitals = cursor.fetchall()
    print("  Latest 3 records:")
    for v in vitals:
        print(f"    Session {v[0]}: Time={v[1]}, HR={v[2]}, SpO2={v[3]}, Sleep={v[4]}, RASS={v[5]}, Risk={v[6]*100:.1f}%")

# Check alerts count
cursor.execute("SELECT COUNT(*) FROM alerts")
alerts_count = cursor.fetchone()[0]
print(f"\n🚨 Alerts: {alerts_count}")

if alerts_count > 0:
    cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 3")
    alerts = cursor.fetchall()
    print("  Latest 3 alerts:")
    for a in alerts:
        ack = "✅" if a[6] else "⏳"
        print(f"    {ack} {a[3]} - {a[4]}: {a[5]} at {a[7]}")

# Check system events
cursor.execute("SELECT COUNT(*) FROM system_events")
events_count = cursor.fetchone()[0]
print(f"\n📋 System events: {events_count}")

if events_count > 0:
    cursor.execute("SELECT * FROM system_events ORDER BY timestamp DESC LIMIT 3")
    events = cursor.fetchall()
    print("  Latest 3 events:")
    for e in events:
        print(f"    {e[1]}: {e[2]} at {e[3]}")

conn.close()
print("\n✅ Database check complete!")
