import os
import sqlite3
from datetime import datetime
from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
import io
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# === Setup Tracking DB ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKING_DB = os.path.join(BASE_DIR, "tracking_logs.db")


# === Helper to Inject Tracking Pixel ===
def add_tracking_to_body(body: str, email_id: str) -> str:
    NGROK_URL = os.getenv("NGROK_URL", "http://localhost:8000").rstrip("/")
    formatted_body = body.replace("\n", "<br>")
    tracking_url = f"{NGROK_URL}/track/open/{email_id}"
    tracking_img = f'<img src="{tracking_url}" width="1" height="1" style="opacity:0;" alt="" />'

    return f"<html><body>{tracking_img}<br>{formatted_body}</body></html>"


@router.get("/track/open/{email_id}")
def track_open(email_id: str):
    print("📡 /track/open route called", flush=True)
    print(f"📥 Tracking pixel hit for email ID: {email_id}", flush=True)

    try:
        conn = sqlite3.connect(TRACKING_DB)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE tracking_logs
            SET status = 'opened',
                sent_time = ?
            WHERE id = ?
        ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), email_id))

        if cursor.rowcount == 0:
            print(f"⚠️ No record found with ID {email_id}", flush=True)
        else:
            print(f"✅ Tracking status updated for {email_id}", flush=True)

        conn.commit()
        conn.close()

        pixel = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01"
            b"\xe2!\xbc\x33\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        print("📦 Returning 1x1 tracking pixel", flush=True)
        return StreamingResponse(io.BytesIO(pixel), media_type="image/png")

    except Exception as e:
        print(f"❌ Tracking error: {e}", flush=True)
        return Response(status_code=500)



# === View Logs ===
@router.get("/track/logs")
def get_tracking_logs():
    try:
        conn = sqlite3.connect(TRACKING_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT id, recipient, subject, status, sent_time FROM tracking_logs ORDER BY sent_time DESC")
        rows = cursor.fetchall()
        conn.close()

        return {
            "logs": [
                {
                    "id": row[0],
                    "recipient": row[1],
                    "subject": row[2],
                    "status": row[3],
                    "sent_time": row[4]
                } for row in rows
            ]
        }
    except Exception as e:
        return {"error": str(e)}


# === Log Entry Creation ===
def log_email(email_id: str, recipient: str, subject: str):
    try:
        conn = sqlite3.connect(TRACKING_DB)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracking_logs (
                id TEXT PRIMARY KEY,
                recipient TEXT,
                subject TEXT,
                status TEXT DEFAULT 'sent',
                sent_time TEXT
            )
        ''')
        cursor.execute('''
            INSERT INTO tracking_logs (id, recipient, subject, status, sent_time)
            VALUES (?, ?, ?, ?, ?)
        ''', (email_id, recipient, subject, "sent", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ DB Error (log_email): {e}")
