from google.auth.transport.requests import Request as GoogleRequest
import pytz
from fastapi import Body

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from mail_merge import perform_mail_merge
from email_utils import send_email
from fastapi import FastAPI, BackgroundTasks
from datetime import datetime, timedelta
from state_utils import load_state, save_state
from datetime import datetime
import time
from apscheduler.schedulers.background import BackgroundScheduler
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
from threading import Lock
state_lock = Lock()
import random
import csv
import base64
import sqlite3
import os
from datetime import datetime, timedelta , timezone
import uvicorn
from typing import Optional, List
from dotenv import load_dotenv
import openai
import threading
import time
import uuid
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import requests
from tracking import router as tracking_router, add_tracking_to_body, log_email
from dotenv import load_dotenv
from fastapi import Response
from fastapi.responses import StreamingResponse

import io
app = FastAPI()


def create_otp_table():
    conn = sqlite3.connect("otp.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS otps (
            email TEXT PRIMARY KEY,
            otp TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

create_otp_table()
class OTPRequest(BaseModel):
    email: str
    otp: str
import random

def generate_otp():
    return str(random.randint(100000, 999999))

def save_otp(email, otp):
    expires_at = (datetime.utcnow() + timedelta(minutes=5)).isoformat()
    conn = sqlite3.connect("otp.db")
    cursor = conn.cursor()
    cursor.execute("REPLACE INTO otps (email, otp, expires_at) VALUES (?, ?, ?)", (email, otp, expires_at))
    conn.commit()
    conn.close()

import smtplib
from email.mime.text import MIMEText

SMTP_EMAIL = "ayushsinghrajput55323@gmail.com"               # Replace with your sender email
SMTP_PASSWORD = "hrhs mrsr deho lfng"         # Use Gmail app password

def send_otp_email(to_email, otp):
    subject = "🔐 Your OTP Code"
    body = f"Your login OTP is: {otp}\n\nIt will expire in 5 minutes."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_EMAIL
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.sendmail(SMTP_EMAIL, to_email, msg.as_string())

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/send-otp")
def send_otp(email: str = Query(..., description="Gmail address to send OTP")):
    otp = generate_otp()
    # 🔐 Log the generated OTP for debugging (remove in production)
    print(f"🔐 Generated OTP for {email}: {otp}")
    save_otp(email, otp)
    send_otp_email(email, otp)
    return JSONResponse(content={"message": f"OTP sent to {email}"})

from fastapi import HTTPException

# Simulated in-memory OTP store — replace with DB in production
otp_store = {}  # email: { otp: "123456", expires_at: datetime }

# Simple in-memory login store — replace with DB or JWT later
logged_in_users = {}  # email: True after OTP verified
bound_gmail_users = {}  # user_email -> set of allowed Gmail addresses


import sqlite3

@app.post("/verify-otp")
def verify_otp(payload: OTPRequest):
    email = payload.email
    otp_input = payload.otp

    if not email or not otp_input:
        raise HTTPException(status_code=400, detail="Email and OTP required")

    conn = sqlite3.connect("otp.db")
    cursor = conn.cursor()
    cursor.execute("SELECT otp, expires_at FROM otps WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="OTP not found")

    saved_otp, expires_at_str = row
    expires_at = datetime.fromisoformat(expires_at_str)

    if saved_otp != otp_input:
        raise HTTPException(status_code=401, detail="Invalid OTP")

    if datetime.utcnow() > expires_at:
        raise HTTPException(status_code=403, detail="OTP expired")

    # ✅ OTP valid
    logged_in_users[email] = True
    if email not in bound_gmail_users:
        bound_gmail_users[email] = set()
    bound_gmail_users[email].add(email)

    print(f"✅ OTP verified and Gmail {email} bound to user session.")

    return {"success": True, "message": f"{email} verified"}



@app.post("/logout")
def logout_user(payload: dict = Body(...)):
    email = payload.get("email")

    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    if email in logged_in_users:
        del logged_in_users[email]
        print(f"👋 Logged out user: {email}")

        # Optionally remove Gmail associations for this user:
        # bound_gmail_users.pop(email, None)

        return {"message": f"Successfully logged out {email}"}
    else:
        raise HTTPException(status_code=404, detail="User not logged in or already logged out")

def email_to_env_key(email: str) -> str:
    return f"CLIENT_SECRET_{email.replace('@', '_at_').replace('.', '_')}"



def get_client_secret_from_file(email: str) -> dict:
    # Replace special characters to match your Render secret filename
    safe_email = email.replace("@", "_at_").replace(".", "_")
    filename = f"CLIENT_SECRET_{safe_email}"
    filepath = f"/etc/secrets/{filename}"

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Client secret file not found at {filepath}")

    with open(filepath, "r") as f:
        return json.load(f)




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Setup: SQLite DB for token storage
TOKEN_DB = os.path.join(BASE_DIR, "token_store.db")

def init_token_db():
    conn = sqlite3.connect(TOKEN_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tokens (
            email TEXT PRIMARY KEY,
            token TEXT,
            refresh_token TEXT,
            token_uri TEXT,
            client_id TEXT,
            client_secret TEXT,
            scopes TEXT,
            expiry TEXT
        )
    """)
    conn.commit()
    conn.close()

# Call on startup
init_token_db()

load_dotenv()  # Load variables from .env file once
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()
from google_auth_oauthlib.flow import Flow
from fastapi.responses import RedirectResponse

GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly"
]


REDIRECT_URI = "https://ai-email-backend-1-m0vj.onrender.com/oauth2callback"


class TokenPayload(BaseModel):
    email: str
    token_data: dict

from google_auth_oauthlib.flow import Flow

@app.get("/get-auth-url")
def get_auth_url(user_email: str, gmail: str):
    print(f"📥 Request to link Gmail {gmail} from logged-in user {user_email}")

    # ✅ Check if user is logged in
    if not logged_in_users.get(user_email):
        return {"error": "User is not logged in or verified"}

    # ✅ Check if Gmail is allowed
    allowed_gmails = bound_gmail_users.get(user_email, set())
    if gmail not in allowed_gmails:
        return {"error": "This Gmail is not linked to your account"}

    try:
        client_config = get_client_secret_from_file(gmail)
        flow = Flow.from_client_config(
            client_config,
            scopes=GOOGLE_SCOPES,
            redirect_uri=REDIRECT_URI,
        )

        auth_url, _ = flow.authorization_url(
            access_type='offline',
            prompt='consent',
            state=gmail
        )

        return {"auth_url": auth_url}

    except FileNotFoundError as e:
        print("❌", e)
        return {"error": str(e)}
    except Exception as e:
        print("❌ Unexpected error:", e)
        return {"error": "Something went wrong generating auth URL"}


        
def get_user_email(credentials):
    """
    Fetch the authenticated user's email address using Gmail API.
    """
    try:
        service = build("gmail", "v1", credentials=credentials)
        profile = service.users().getProfile(userId="me").execute()
        return profile["emailAddress"]
    except Exception as e:
        print(f"❌ Failed to get user email: {e}")
        raise


@app.get("/oauth2callback")
def oauth2callback(request: Request):
    code = request.query_params.get("code")
    email_hint = request.query_params.get("state")  # This is the original email passed in state

    if not code or not email_hint:
        return {"error": "Missing code or email (state) in callback URL"}

    client_secret_file = get_client_secret_from_file(email_hint)

    flow = Flow.from_client_config(
    client_secret_file,  # this is now a dict, not a file path
    scopes=GOOGLE_SCOPES,
    redirect_uri=REDIRECT_URI,
    )


    flow.fetch_token(code=code)
    credentials = flow.credentials

    actual_email = get_user_email(credentials)  # Gmail account from token, not hint

    token_data = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes,
        "expiry": credentials.expiry.isoformat()
    }

    save_client_token(actual_email, token_data)

    return RedirectResponse(url=f"http://localhost:3000/?success=true&email={actual_email}")



def save_client_token(email, token_dict):
    conn = sqlite3.connect(TOKEN_DB)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO tokens (
            email, token, refresh_token, token_uri, client_id, client_secret, scopes, expiry
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        email,
        token_dict.get("token"),
        token_dict.get("refresh_token"),
        token_dict.get("token_uri"),
        token_dict.get("client_id"),
        token_dict.get("client_secret"),
        json.dumps(token_dict.get("scopes", [])),
        token_dict.get("expiry"),
    ))

    conn.commit()
    conn.close()


def load_client_token(email):
    conn = sqlite3.connect(TOKEN_DB)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM tokens WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise FileNotFoundError(f"No token found for {email}")

    token_dict = {
        "token": row[1],
        "refresh_token": row[2],
        "token_uri": row[3],
        "client_id": row[4],
        "client_secret": row[5],
        "scopes": json.loads(row[6]),
        "expiry": row[7],
    }
    return token_dict

def send_email(recipient, subject, body, client_token_data: dict):
    try:
        creds = Credentials.from_authorized_user_info(client_token_data)

        # Refresh token if expired
        if creds.expired and creds.refresh_token:
            creds.refresh(GoogleRequest())


        service = build("gmail", "v1", credentials=creds)

        email_id = str(uuid.uuid4())
        html_body = add_tracking_to_body(body, email_id)

        message = MIMEMultipart("alternative")
        message["to"] = recipient
        message["subject"] = subject
        message.attach(MIMEText(body, "plain"))
        message.attach(MIMEText(html_body, "html"))

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        service.users().messages().send(userId="me", body={"raw": raw}).execute()

        print(f"✅ Email sent to {recipient}")
        log_email(email_id, recipient, subject)

    except Exception as e:
        print(f"❌ Error sending email: {e}")


# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_BASE_URL")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKING_DB = os.path.join(BASE_DIR, "tracking_logs.db")
SCHEDULED_DB = os.path.join(BASE_DIR, "scheduled_emails.db")


scheduler = BackgroundScheduler()
scheduler.start()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def send_email_from(sender_email, recipient, subject, body):
    try:
        client_token = load_client_token(sender_email)
        send_email(recipient, subject, body, client_token_data=client_token)
    except Exception as e:
        print(f"❌ Error sending email from {sender_email} to {recipient}: {e}")


# === Warmup State Handling ===
def load_state(filename="warmup_state.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"enabled": False, "progress": 0}

def save_state(state, filename="warmup_state.json"):
    abs_path = os.path.abspath(filename)
    print(f"💾 Saving state to: {abs_path}")
    with open(abs_path, "w") as f:
        json.dump(state, f)


warmup_state = load_state()

@app.on_event("shutdown")
def shutdown_event():
    print("Shutting down gracefully...")
    warmup_state["enabled"] = False
    save_state(warmup_state)

# === Load Warmup Pairs from Config ===
def load_warmup_pairs():
    try:
        with open("warmup_config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# === Save Warmup Pairs via API ===
@app.post("/update-warmup-pairs")
async def update_warmup_pairs(pairs: list):
    try:
        with open("warmup_config.json", "w") as f:
            json.dump(pairs, f, indent=4)
        return {"status": "Updated", "pairs": pairs}
    except Exception as e:
        return {"error": str(e)}

# === Warmup Control APIs ===
@app.post("/start-warmup")
def start_warmup():
    print("📥 /start-warmup called")
    state = load_state()  # Always load fresh

    if state.get("enabled"):
        print("⚠️ Warmup already running")
        return {"status": "Already Running"}

    state["enabled"] = True
    state["pairs"] = [
        {"sender": "ayushsinghrajput55323@gmail.com", "recipient": "redmicare72@gmail.com"},
        {"sender": "redmicare72@gmail.com", "recipient": "ayushsinghrajput55323@gmail.com"}
    ]
    state.setdefault("progress", 0)  # Don’t reset if already exists

    save_state(state)
    print("💾 Warmup state saved")

    # Run first batch immediately
    print("🚀 Triggering first warmup queue")
    queue_warmup_emails()

    # Schedule every 5 hours
    print("⏰ Scheduling warmup every 5 hours")
    scheduler.add_job(
        queue_warmup_emails,
        trigger="interval",
        hours=5,
        id="warmup-job",
        replace_existing=True
    )

    return {"status": "Warmup Started"}



@app.post("/stop-warmup")
def stop_warmup():
    warmup_state["enabled"] = False
    save_state(warmup_state)
    scheduler.remove_all_jobs()
    return {"status": "Warmup Stopped"}



@app.get("/warmup-status")
def get_status():
    return load_state()  # ✅ Always returns latest saved state


# === Gmail Sender ===
def send_warmup_email(*, to, subject, body, sender):
    print(f"📤 Attempting to send warmup email from {sender} to {to}")
    
    try:
        send_email_from(sender, to, subject, body)
    except Exception as e:
        print(f"❌ Failed to send warmup email: {e}")
        return

    print("🟡 Reached progress update block")
    with state_lock:
        state = load_state()
        state["progress"] = state.get("progress", 0) + 1
        save_state(state)
        print(f"📊 Progress updated to: {state['progress']}")


@app.on_event("startup")
def resume_if_warmup_enabled():
    state = load_state()
    if state.get("enabled"):
        print("🔁 Resuming warmup after restart")
        scheduler.add_job(
            queue_warmup_emails,
            trigger="interval",
            hours=5,
            id="warmup-job",
            replace_existing=True
        )


# === Reply Simulator ===
def simulate_reply(sender_email, from_email):
    reply_subject = "Re: Warmup Email"
    reply_body = "Thanks for reaching out! This is a simulated warmup reply."

    send_warmup_email(
        to=sender_email,
        subject=reply_subject,
        body=reply_body,
        sender=from_email
    )

def queue_warmup_emails():
    print("⚙️ queue_warmup_emails() called")

    state = load_state()
    if not state.get("enabled"):
        print("⛔ Warmup not enabled")
        return

    warmup_pairs = state.get("pairs", [])
    print(f"📦 Loaded {len(warmup_pairs)} warmup pairs")

    for idx, pair in enumerate(warmup_pairs):
        sender = pair["sender"]
        recipient = pair["recipient"]

        print(f"📤 Scheduling email from {sender} to {recipient}")
        scheduler.add_job(
            send_warmup_email,
            trigger="date",
            run_date=datetime.now() + timedelta(seconds=idx * 5),
            kwargs={
                "to": recipient,
                "subject": "Warmup Email",
                "body": "Hi! This is a warmup email.",
                "sender": sender
            },
            id=f"warmup-send-{idx}-{int(time.time())}",  # avoid ID collision
            replace_existing=False
        )

        print(f"📥 Scheduling reply from {recipient} to {sender}")
        scheduler.add_job(
            send_warmup_email,
            trigger="date",
            run_date=datetime.now() + timedelta(seconds=idx * 5 + 60),
            kwargs={
                "to": sender,
                "subject": "Re: Warmup Email",
                "body": "Thanks for reaching out!",
                "sender": recipient
            },
            id=f"warmup-reply-{idx}-{int(time.time())}",  # avoid ID collision
            replace_existing=False
        )


        
class EmailRequest(BaseModel):
    action: str
    sender_email: Optional[str] = None  # <-- NEW: required for sending
    recipient: Optional[str] = None
    recipients: Optional[List[str]] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    schedule_time: Optional[str] = None
    tone: Optional[str] = None
    delay: Optional[int] = 30
    timezone: str = "UTC"  # Default to UTC if not provided


def generate_email_with_ai(prompt: str, tone: str) -> str:
    full_prompt = f"Generate an email in a {tone} tone with the following content:\n\n{prompt}"
    try:
        response = openai.ChatCompletion.create(
            model="mistralai/mistral-7b-instruct",
            messages=[{"role": "user", "content": full_prompt}]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ AI Error: {str(e)}"

def store_scheduled_email(sender_email, recipient, subject, body, schedule_time, timezone_str):
    # Convert local time to UTC
    local_tz = pytz.timezone(timezone_str)
    local_dt = local_tz.localize(datetime.strptime(schedule_time, "%Y-%m-%d %H:%M:%S"))
    utc_dt = local_dt.astimezone(pytz.utc)

    conn = sqlite3.connect(SCHEDULED_DB)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scheduled_emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT NOT NULL,
            recipient TEXT NOT NULL,
            subject TEXT NOT NULL,
            body TEXT NOT NULL,
            schedule_time TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        INSERT INTO scheduled_emails (sender, recipient, subject, body, schedule_time)
        VALUES (?, ?, ?, ?, ?)
    ''', (sender_email, recipient, subject, body, utc_dt.strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    
def init_scheduled_db():
    conn = sqlite3.connect(SCHEDULED_DB)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scheduled_emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT NOT NULL,
            recipient TEXT NOT NULL,
            subject TEXT NOT NULL,
            body TEXT NOT NULL,
            schedule_time TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def check_and_send_scheduled_emails():
    print("🚀 check_and_send_scheduled_emails started")

    while True:
        conn = sqlite3.connect(SCHEDULED_DB, check_same_thread=False)
        cursor = conn.cursor()
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n🕒 Scheduler running at: {current_time}")

        # Log all scheduled emails in the DB
        cursor.execute('SELECT * FROM scheduled_emails')
        rows = cursor.fetchall()
        if rows:
            print("📋 Scheduled emails in DB:")
            for row in rows:
                print(f"📝 ID: {row[0]}, Sender: {row[1]}, Recipient: {row[2]}, Time: {row[5]}")
        else:
            print("📭 No emails scheduled in DB.")

        # Get emails due for sending
        cursor.execute('SELECT id, sender, recipient, subject, body FROM scheduled_emails WHERE schedule_time <= ?', (current_time,))
        emails = cursor.fetchall()

        for email in emails:
            email_id, sender, recipient, subject, body = email
            try:
                client_token = load_client_token(sender)
                send_email(recipient, subject, body, client_token_data=client_token)
                print(f"✅ Scheduled email sent from {sender} to {recipient}")
            except Exception as e:
                print(f"❌ Failed to send scheduled email from {sender} to {recipient}: {e}")

            # Remove sent email from DB
            cursor.execute('DELETE FROM scheduled_emails WHERE id = ?', (email_id,))
            conn.commit()

        conn.close()
        print(f"🔁 Scheduler loop complete. Sleeping for 60 seconds...\n")
        time.sleep(60)


@app.on_event("startup")
def start_scheduled_email_worker():
    print("🕒 Starting scheduled email processor...")
    init_scheduled_db()  # 🔑 Make sure table exists before thread starts
    threading.Thread(target=check_and_send_scheduled_emails, daemon=True).start()


def fetch_sheet_data(sheet_url):
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    creds = ServiceAccountCredentials.from_json_keyfile_name("sheets_credentials.json", SCOPES)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(sheet_url)
    worksheet = sheet.get_worksheet(0)
    return worksheet.get_all_records()
@app.post("/email-action")
async def api_email_action(email_data: EmailRequest, background_tasks: BackgroundTasks):
    if email_data.action == "generate":
        if not email_data.body or not email_data.tone:
            return {"status": "Error", "message": "Body and tone are required"}

        prompt = f"Write a {email_data.tone.lower()} email for the following situation:\n\n\"{email_data.body}\""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional email assistant. Write polished, clear, and concise emails based on the user's request."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=300,
            )
            message = response["choices"][0]["message"]["content"].strip()
            return {"status": "Generated", "message": f"📝 AI-Generated Email:\n\n{message}"}
        except Exception as e:
            return {"status": "Error", "message": str(e)}

    elif email_data.action == "schedule":
        if not email_data.sender_email:
            return {"status": "Error", "message": "sender_email is required"}

        if not email_data.timezone:
            return {"status": "Error", "message": "Timezone is required for scheduling"}

        recipients = email_data.recipients or ([email_data.recipient] if email_data.recipient else [])
        if not recipients or not email_data.schedule_time or not email_data.body:
            return {"status": "Error", "message": "Recipient(s), schedule time, and body required"}

        for recipient in recipients:
            store_scheduled_email(
                sender_email=email_data.sender_email,
                recipient=recipient,
                subject=email_data.subject or "Scheduled Email",
                body=email_data.body,
                schedule_time=email_data.schedule_time,
                timezone_str=email_data.timezone   # ✅ New param to convert to UTC
            )

        return {"status": "Scheduled", "message": f"📅 Scheduled for {len(recipients)} recipient(s)"}

    elif email_data.action == "extract":
        if not email_data.body:
            return {"status": "Error", "message": "Body required"}

        extract_prompt = f"""
You are an AI email assistant. Extract only the *important actionable key points* as clear bullet points:

Email:
{email_data.body}
"""
        ai_response = generate_email_with_ai(extract_prompt.strip(), "Concise")
        return {"status": "Extracted", "message": f"📌 Key Points:\n\n{ai_response}"}

    elif email_data.action == "mass-send":
        if not email_data.sender_email:
            return {"status": "Error", "message": "sender_email is required"}

        try:
            client_token = load_client_token(email_data.sender_email)
        except Exception as e:
            return {"status": "Error", "message": f"Failed to load sender token: {e}"}

        if not email_data.recipients or not email_data.body or not email_data.subject:
            return {"status": "Error", "message": "Recipients, subject, and body required"}

        batch_size = 50
        delay_between_batches = 10

        def send_emails_in_batches(recipients, subject, body):
            for i in range(0, len(recipients), batch_size):
                batch = recipients[i:i+batch_size]
                for email in batch:
                    try:
                        send_email(email, subject, body, client_token_data=client_token)
                    except Exception as e:
                        print(f"❌ Failed to send to {email}: {e}")
                time.sleep(delay_between_batches)

        background_tasks.add_task(send_emails_in_batches, email_data.recipients, email_data.subject, email_data.body)
        return {"status": "Mass Sending Started", "message": f"🚀 Sending {len(email_data.recipients)} emails"}

    elif email_data.action == "drip-pairs":
        if not email_data.sender_email:
            return {"status": "Error", "message": "sender_email is required"}

        try:
            client_token = load_client_token(email_data.sender_email)
        except Exception as e:
            return {"status": "Error", "message": f"Failed to load sender token: {e}"}

        if not email_data.body or not email_data.body.startswith("http"):
            return {"status": "Error", "message": "A valid Google Sheet URL is required."}

        delay = int(email_data.delay or 30)

        try:
            rows = fetch_sheet_data(email_data.body)
        except Exception as e:
            return {"status": "Error", "message": f"Failed to fetch Google Sheet: {e}"}

        if not rows or "Email" not in rows[0]:
            return {"status": "Error", "message": "Google Sheet must contain an 'Email' column."}

        all_keys = rows[0].keys()
        message_parts = sorted(set(
            int(key.split("(")[1].split(")")[0])
            for key in all_keys if "Subject(" in key or "Body(" in key
        ))

        def send_drip_pairs_to_all():
            for part_no in message_parts:
                subject_key = f"Subject({part_no})"
                body_key = f"Body({part_no})"
                for row in rows:
                    recipient = row.get("Email", "").strip()
                    subject = row.get(subject_key, "").strip()
                    body = row.get(body_key, "").strip()
                    if recipient and subject and body:
                        try:
                            send_email(recipient, subject, body, client_token_data=client_token)
                            print(f"✅ Sent Part {part_no} to {recipient}")
                        except Exception as e:
                            print(f"❌ Failed to send to {recipient}: {e}")
                print(f"⏳ Waiting {delay}s before next message part...")
                time.sleep(delay)

        background_tasks.add_task(send_drip_pairs_to_all)
        return {
            "status": "Drip Campaign Started",
            "message": f"🚀 Sending {len(message_parts)} emails to {len(rows)} recipients with {delay}s delay."
        }

    else:
        return {"status": "Error", "message": "Invalid action type"}


@app.post("/upload-emails")
async def upload_emails(file: UploadFile = File(...)):
    contents = await file.read()
    decoded = contents.decode("utf-8").splitlines()
    emails = [line.strip() for line in decoded if "@" in line and "." in line]
    return {"emails": emails, "count": len(emails)}

@app.post("/mail-merge")
async def run_mail_merge(request: Request):
    try:
        data = await request.json()
        sheet_url = data.get("sheet_url")
        subject_template = data.get("subject_template")
        body_template = data.get("body_template")
        sender_email = data.get("sender_email")

        # Validate all inputs early
        missing = [k for k in ["sheet_url", "subject_template", "body_template", "sender_email"] if not data.get(k)]
        if missing:
            return {"status": "error", "message": f"Missing fields: {', '.join(missing)}"}

        # Load token and perform mail merge
        token = load_client_token(sender_email)
        result = perform_mail_merge(sheet_url, subject_template, body_template, token)

        return {"status": "completed", "result": result}

    except Exception as e:
        return {"status": "error", "message": f"Server error: {str(e)}"}

@app.get("/campaign-report")
def get_campaign_report():
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
        cursor.execute("SELECT id, recipient, subject, status, sent_time FROM tracking_logs ORDER BY sent_time DESC")
        rows = cursor.fetchall()
        conn.close()

        result = [
            {
                "id": row[0],
                "recipient": row[1],
                "subject": row[2],
                "status": row[3],
                "sent_time": row[4]
            }
            for row in rows
        ]

        return result
    except Exception as e:
        return {"status": "error", "message": f"Failed to fetch report: {str(e)}"}

class ABTestRequest(BaseModel):
    sheet_url: str
    sender_email: str  # <-- NEW


@app.post("/ab-test")
def ab_test(data: ABTestRequest):
    sheet_url = data.sheet_url.replace("/edit#gid=", "/export?format=csv&gid=")

    try:
        response = requests.get(sheet_url)
        decoded = response.content.decode('utf-8')
        lines = list(csv.reader(decoded.splitlines()))
        headers = lines[0]
        rows = lines[1:]

        if len(headers) < 5:
            return {"error": "Sheet must contain: Email, Subject A, Body A, Subject B, Body B"}

        try:
            client_token = load_client_token(data.sender_email)
        except Exception as e:
            return {"error": f"Failed to load sender token: {e}"}

        group_a = []
        group_b = []

        for row in rows:
            if len(row) < 5:
                continue

            email = row[0]
            subject_a = row[1]
            body_a = row[2]
            subject_b = row[3]
            body_b = row[4]

            if random.choice([True, False]):
                group_a.append({"email": email, "subject": subject_a, "body": body_a})
            else:
                group_b.append({"email": email, "subject": subject_b, "body": body_b})

        for user in group_a:
            send_email(user["email"], user["subject"], user["body"], client_token_data=client_token)

        for user in group_b:
            send_email(user["email"], user["subject"], user["body"], client_token_data=client_token)

        return {
            "status": "success",
            "group_a_count": len(group_a),
            "group_b_count": len(group_b),
            "group_a_emails": [user["email"] for user in group_a],
            "group_b_emails": [user["email"] for user in group_b]
        }

    except Exception as e:
        return {"error": str(e)}
@app.get("/")
def root():
    return {"message": "✅ AI Email Assistant Backend is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
