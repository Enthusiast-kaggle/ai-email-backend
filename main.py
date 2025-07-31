from io import BytesIO
from PIL import Image
from google.auth.transport.requests import Request as GoogleRequest
import pytz
from fastapi import Body
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from apscheduler.schedulers.background import BackgroundScheduler
from threading import Lock
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
import re
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
import os
import sqlite3

import sqlite3


# Absolute DB path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_DB = os.path.join(BASE_DIR, "tokens.db")

def init_token_db():
    print(f"📂 Initializing DB at: {TOKEN_DB}")
    conn = sqlite3.connect(TOKEN_DB)
    cursor = conn.cursor()
    cursor.execute('''
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
    ''')
    conn.commit()
    conn.close()
    print("✅ Token DB initialized.")

# Always call at app startup
init_token_db()

import io
app = FastAPI()

import json

# Load warmup pool once
with open("warmup_pool.json", "r") as f:
    WARMUP_POOL = json.load(f)

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

from datetime import datetime

WARMUP_PROGRESS_FILE = "warmup_progress.json"

def load_warmup_progress():
    try:
        with open(WARMUP_PROGRESS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_warmup_progress(progress):
    with open(WARMUP_PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

from email.mime.text import MIMEText
from base64 import urlsafe_b64encode

def create_warmup_message(sender, to_email):
    subject = "Hey! Just checking in 😊"
    body = "Hi there!\n\nThis is a warmup email. Please ignore.\n\nCheers!"

    message = MIMEText(body)
    message["to"] = to_email
    message["from"] = sender
    message["subject"] = subject

    raw_message = urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    return {"raw": raw_message}

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from datetime import datetime

def send_warmup_emails(sender_email, creds):
    global WARMUP_POOL
    warmup_pool = WARMUP_POOL

    warmup_progress = load_warmup_progress()

    # Initialize if not tracked
    if sender_email not in warmup_progress:
        warmup_progress[sender_email] = {"stage": 1, "last_sent": None}

    stage = warmup_progress[sender_email]["stage"]
    last_sent = warmup_progress[sender_email]["last_sent"]

    # Send once per day
    today_str = datetime.now().strftime("%Y-%m-%d")
    if last_sent == today_str:
        print(f"Warmup already sent today for {sender_email}")
        return

    # Determine how many recipients to send based on stage
    step_size = 5
    limit = min(stage * step_size, len(warmup_pool))
    recipients = warmup_pool[:limit]

    # Prepare email subject and body
    subject = "Hey there! Warming up this email 😊"
    body = (
        "Hi!\n\n"
        "Just sending this warmup message to improve deliverability. "
        "No need to respond.\n\nCheers!"
    )

    # Prepare token data from creds
    client_token_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
        "expiry": creds.expiry.isoformat() if creds.expiry else None
    }

    for recipient in recipients:
        if recipient == sender_email:
            continue
        send_email(recipient, subject, body, client_token_data , sender_email)

    # Update progress
    if limit < len(warmup_pool):
        warmup_progress[sender_email]["stage"] += 1
    warmup_progress[sender_email]["last_sent"] = today_str
    save_warmup_progress(warmup_progress)

    print(f"Warmup Stage {stage}: Sent to {limit} recipients for {sender_email}")

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
    send_warmup_emails(actual_email, credentials)

    return RedirectResponse(url=f"http://localhost:3000/?success=true&email={actual_email}")




def save_client_token(email, token_dict):
    print(f"🔐 Saving token to: {TOKEN_DB}")
    try:
        conn = sqlite3.connect(TOKEN_DB)
    except Exception as e:
        print(f"❌ ERROR: Could not open DB file at {TOKEN_DB} — {e}")
        raise
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

def debug_show_all_tokens():
    import sqlite3
    conn = sqlite3.connect(TOKEN_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM tokens")
    rows = cursor.fetchall()
    print("📬 Emails in token DB:")
    for row in rows:
        print(f" - {row[0]}")
    conn.close()

def load_client_token(email):
    print(f"📂 Using DB: {TOKEN_DB}")  # Debug print
    print(f"🔐 Saving token to: {TOKEN_DB}")
    try:
        conn = sqlite3.connect(TOKEN_DB)
    except Exception as e:
        print(f"❌ ERROR: Could not open DB file at {TOKEN_DB} — {e}")
        raise
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
    print(f"📂 Connecting to DB at: {TOKEN_DB}")

    return token_dict
    
def send_email(recipient, subject, body, client_token_data: dict, sender_email: str):
    try:
        creds = Credentials.from_authorized_user_info(client_token_data)

        # Refresh token if expired
        if creds.expired and creds.refresh_token:
            creds.refresh(GoogleRequest())


        service = build("gmail", "v1", credentials=creds)
    

        email_id = str(uuid.uuid4())
        html_body = body  # Already contains tracking

        message = MIMEMultipart("alternative")
        message["to"] = recipient
        message["subject"] = subject
        message.attach(MIMEText(html_body, "html"))

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        service.users().messages().send(userId="me", body={"raw": raw}).execute()

        print(f"✅ Email sent to {recipient}")
        log_email(sender_email, recipient, subject)
        return {"status": "success", "email_id": email_id, "recipient": recipient}

    except Exception as e:
        print(f"❌ Error sending email: {e}")
        return {"status": "error", "error": str(e), "recipient": recipient}

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_BASE_URL")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKING_DB = os.path.join(BASE_DIR, "tracking_logs.db")
SCHEDULED_DB = os.path.join(BASE_DIR, "scheduled_emails.db")


scheduler = BackgroundScheduler()
scheduler.start()

# Paths
@app.get("/download-tokens-db")
def download_db():
    from fastapi.responses import FileResponse
    return FileResponse("tokens.db", media_type='application/octet-stream', filename="tokens.db")


def send_email_from(sender, recipient, subject, body):
    try:
        token_data = load_client_token(sender)
        send_email(recipient, subject, body, client_token_data=token_data)
        print(f"📨 Sent email from {sender} to {recipient}")
    except Exception as e:
        print(f"❌ Error sending from {sender} to {recipient}: {e}")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

def get_latest_token():
    db_path = os.path.join(BASE_DIR, "token_store.db")  # consistent with your saving path
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM tokens ORDER BY expiry DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise FileNotFoundError("No token found in token store.")

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

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

def get_user_email_from_token(creds_dict):
    creds = Credentials.from_authorized_user_info(creds_dict)

    # Automatically refresh if token is expired
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())

    # Build the service
    service = build('oauth2', 'v2', credentials=creds)
    user_info = service.userinfo().get().execute()
    return user_info['email']


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
                send_email(
                             recipient=recipient,
                             subject=subject,
                             body=body,
                             client_token_data=client_token,
                             sender_email=sender
                            )

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
            sender_email = email_data.sender_email
            for i in range(0, len(recipients), batch_size):
                batch = recipients[i:i+batch_size]
                for email in batch:
                    try:
                         send_email(
                             recipient=recipient,
                             subject=subject,
                             body=body,
                             client_token_data=client_token,
                             sender_email=sender_email
                            )
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
            sender_email = email_data.sender_email
            for part_no in message_parts:
                subject_key = f"Subject({part_no})"
                body_key = f"Body({part_no})"
                for row in rows:
                    recipient = row.get("Email", "").strip()
                    subject = row.get(subject_key, "").strip()
                    body = row.get(body_key, "").strip()
                    if recipient and subject and body:
                        try:
                            send_email(
                                       recipient=recipient,
                                       subject=subject,
                                       body=body,
                                       client_token_data=client_token,
                                       sender_email=sender_email
                                      )

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


def get_token_by_email(email):
    conn = sqlite3.connect(TOKEN_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT token, refresh_token, token_uri, client_id, client_secret, scopes, expiry FROM tokens WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()

    if row:
        token_dict = {
            "token": row[0],
            "refresh_token": row[1],
            "token_uri": row[2],
            "client_id": row[3],
            "client_secret": row[4],
            "scopes": json.loads(row[5] or "[]"),
            "expiry": row[6]
        }
        return token_dict
    return None
    
# Initialize tracking table if not exists
def initialize_tracking_db():
    conn = sqlite3.connect(TRACKING_DB)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tracking_logs (
            id TEXT PRIMARY KEY,
            sender TEXT,
            recipient TEXT,
            subject TEXT,
            status TEXT DEFAULT 'sent',
            sent_time TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Call initialization once at startup
initialize_tracking_db()

def log_email(sender, recipient, subject, status="sent"):
    conn = sqlite3.connect(TRACKING_DB)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO tracking_logs (id, sender, recipient, subject, status, sent_time)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        str(uuid.uuid4()),
        sender,
        recipient,
        subject,
        status,
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

@app.get("/get-campaign-report")
async def get_campaign_report(request: Request):
    try:
        # Extract sender email from token (adjust if your logic differs)
        logger.info("🔐 Loading Gmail token for user...")
        token = load_client_token(user_email)
        sender_email = data.user_email

        if not sender_email:
            return JSONResponse(status_code=401, content={"error": "Sender not authenticated"})

        conn = sqlite3.connect(TRACKING_DB)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT recipient, subject, status, sent_time
            FROM tracking_logs
            WHERE sender = ?
            ORDER BY sent_time DESC
        ''', (sender_email,))
        logs = cursor.fetchall()
        conn.close()

        return {"report": [
            {
                "recipient": row[0],
                "subject": row[1],
                "status": row[2],
                "sent_time": row[3]
            }
            for row in logs
        ]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


class ABTestRequest(BaseModel):
    sheet_url: str
    user_email: str
def generate_transparent_pixel():
    img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ✅ Replace with your PostgreSQL URL from Render
DATABASE_URL = "postgresql://emailassistant_db_user:t9XtfTyK3MNV2qgo4hsb2xWOcTWfYqbL@dpg-d246ec7fte5s73apf2mg-a/emailassistant_db"

# ✅ SQLAlchemy Setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ✅ Models
class EmailOpen(Base):
    __tablename__ = "email_opens"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True)
    group = Column(String, index=True)
    sender = Column(String, index=True)
    opened_at = Column(DateTime, default=datetime.utcnow)

class EmailClick(Base):
    __tablename__ = "email_clicks"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True)
    group = Column(String, index=True)
    sender = Column(String, index=True)
    clicked_url = Column(String)
    clicked_at = Column(DateTime, default=datetime.utcnow)

# ✅ Create tables if they don't exist
Base.metadata.create_all(bind=engine)

# ✅ DB Session helper
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ✅ 1. Open Tracking
@app.get("/open-track")
async def open_tracking(email: str, group: str, sender: str):
    logger.info(f"📩 OPEN TRACK RECEIVED: email={email}, group={group}, sender={sender}")
    
    try:
        db = next(get_db())
        new_open = EmailOpen(
            email=email,
            group=group,
            sender=sender,
            opened_at=datetime.utcnow()
        )
        db.add(new_open)
        db.commit()
        db.refresh(new_open)

        logger.info(f"✅ Open tracked: {new_open}")
        return JSONResponse(content={"status": "open tracked", "id": new_open.id})
    
    except Exception as e:
        logger.error(f"❌ Failed to track open: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Failed to track open"})


@app.get("/click-track")
async def click_tracking(email: str, group: str, sender: str, url: str):
    logger.info(f"🔗 CLICK TRACK RECEIVED: email={email}, group={group}, sender={sender}, url={url}")
    
    try:
        db = next(get_db())
        new_click = EmailClick(
            email=email,
            group=group,
            sender=sender,
            clicked_url=url,
            clicked_at=datetime.utcnow()
        )
        db.add(new_click)
        db.commit()
        db.refresh(new_click)

        logger.info(f"✅ Click tracked: {new_click}")
        return JSONResponse(content={"status": "click tracked", "id": new_click.id})
    
    except Exception as e:
        logger.error(f"❌ Failed to track click: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Failed to track click"})


from fastapi import APIRouter, Request
import pandas as pd
import random
import requests
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_csv_url(sheet_url):
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
    if match:
        sheet_id = match.group(1)
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    else:
        return None
        

from google.oauth2.credentials import Credentials
from bs4 import BeautifulSoup
import urllib.parse
@app.post("/ab-test")
async def ab_test(data: ABTestRequest, request: Request):
    try:
        logger.info("🔵 A/B test endpoint called.")
        sheet_url = data.sheet_url
        user_email = data.user_email
        logger.info(f"🟡 Received A/B test request with sheet URL: {sheet_url} for user: {user_email}")
        print("🟡 Received A/B test data:", data.dict())

        # --- Step 1: Convert to CSV ---
        def convert_to_csv_url(sheet_url: str) -> str:
            logger.info("🔧 Converting sheet URL to CSV URL...")
            if "docs.google.com" in sheet_url and "/edit" in sheet_url:
                csv = sheet_url.replace('/edit?usp=sharing', '/export?format=csv')
                logger.info(f"✅ Converted CSV URL: {csv}")
                return csv
            logger.warning("❌ Invalid Google Sheet URL format")
            return None
        

        csv_url = convert_to_csv_url(sheet_url)
        if not csv_url:
            logger.error("❌ CSV URL conversion failed.")
            return {"error": "Invalid Google Sheet URL format."}

        logger.info("📥 Downloading CSV data from URL...")
        response = requests.get(csv_url)
        logger.info(f"📡 Response status code: {response.status_code}")
        if response.status_code != 200:
            logger.error("❌ Failed to fetch CSV from Google Sheets.")
            return {"error": "Failed to download CSV from provided URL."}

        logger.info("🧾 Reading CSV into DataFrame...")
        df = pd.read_csv(io.BytesIO(response.content))
        logger.info("✅ CSV successfully read into DataFrame.")

        df.columns = df.columns.str.strip().str.lower()
        print("📄 CSV Columns:", df.columns.tolist())
        logger.info(f"📊 Cleaned Columns: {df.columns.tolist()}")

        required_columns = {"email", "subject(1)", "body(1)", "subject(2)", "body(2)"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            logger.error(f"❌ Missing required columns: {missing}")
            return {"error": f"Missing required columns: {', '.join(missing)}"}

        logger.info("✅ Required columns found, proceeding to split groups.")

        # --- Step 2: Split into A and B groups ---
        df["email"] = df["email"].astype(str).str.strip()
        emails = df["email"].dropna().unique().tolist()
        logger.info(f"📧 Unique cleaned emails count: {len(emails)}")

        random.shuffle(emails)
        midpoint = len(emails) // 2
        group_a = set(emails[:midpoint])
        logger.info(f"🅰️ Group A size: {len(group_a)}, 🅱️ Group B size: {len(emails) - len(group_a)}")

        df["group"] = df["email"].apply(lambda e: "A" if e in group_a else "B")
        df["subject"] = df.apply(lambda row: row["subject(1)"] if row["group"] == "A" else row["subject(2)"], axis=1)
        df["body"] = df.apply(lambda row: row["body(1)"] if row["group"] == "A" else row["body(2)"], axis=1)

        final_df = df[["email", "subject", "body"]]
        print("🧪 Final DataFrame preview:\n", final_df.head())
        logger.info("✅ Final email data prepared.")

        # Drop invalid rows
        final_df.dropna(subset=["email", "subject", "body"], inplace=True)
        logger.info(f"✅ Rows after dropna: {len(final_df)}")

        # --- Step 3: Fetch token using user_email ---
        logger.info("🔐 Loading Gmail token for user...")
        token = load_client_token(user_email)
        sender_email = data.user_email

        print("📦 Loaded token:", token)
        if not token:
            print("❌ Token not found for:", user_email)
            logger.error("❌ No token found for user.")
            return JSONResponse(status_code=400, content={"error": "Could not fetch token for user email."})

        # --- Step 4: Send Emails ---
        logger.info("📤 Starting to send emails...")
        success_count = 0
        failure_count = 0

        for index, row in final_df.iterrows():
            print(f"📤 Loop {index} — Email: {row['email']}")
            to_email = row["email"]
            subject = row["subject"]
            body = row["body"]

            if not to_email or not subject or not body:
                logger.warning(f"⛔ Skipping email to {to_email}: Empty subject/body")
                print("🚫 Skipped row:", row.to_dict())
                failure_count += 1
                continue

            # Determine group
            group = "A" if to_email in group_a else "B"

            # Inject click tracking into links
            soup = BeautifulSoup(body, "html.parser")
            for link in soup.find_all("a", href=True):
                original_url = link['href']
                tracking_url = f"https://ai-email-backend-1-m0vj.onrender.com/click-track?email={to_email}&group={group}&url={original_url}&sender={sender_email}"

                link['href'] = tracking_url


                tracking_pixel = f'<img src="https://ai-email-backend-1-m0vj.onrender.com/open-track?email={to_email}&group={group}&sender={sender_email}" width="1" height="1" style="display:none;">'


            # Final HTML with click and open tracking
            final_html = str(soup) + tracking_pixel
            print("📧 Final HTML sent:\n", final_html)

            print(f"📤 Sending email to: {to_email}")
            result = send_email(to_email, subject, final_html, token , sender_email)

            print(f"📥 Email send result: {result}")

            if result["status"] == "success":
                success_count += 1
            else:
                logger.warning(f"❌ Failed to send email to {to_email}: {result.get('error')}")
                failure_count += 1

        logger.info("✅ Email sending loop completed.")
        return JSONResponse(content={
            "status": "completed",
            "success": success_count,
            "failed": failure_count
        })

    except Exception as e:
        logger.exception("❗ Unhandled error during A/B test execution")
        return JSONResponse(status_code=500, content={"error": str(e)})
        
from fastapi import Request, Query
from typing import Optional

@app.get("/ab-engagement-report")
async def engagement_report(sender: Optional[str] = Query(None)):
    db = next(get_db())

    try:
        logger.info("📊 Fetching detailed engagement report...")

        if not sender:
            return JSONResponse(status_code=400, content={"error": "Sender email is required as a query parameter"})

        # Get opens by sender
        open_data = db.query(
            EmailOpen.email,
            EmailOpen.group,
            EmailOpen.timestamp.label("opened_at"),
            EmailOpen.target_url
        ).filter(EmailOpen.sender == sender).all()

        # Get clicks by sender
        click_data = db.query(
            EmailClick.email,
            EmailClick.group,
            EmailClick.timestamp.label("clicked_at"),
            EmailClick.target_url
        ).filter(EmailClick.sender == sender).all()

        # Combine by (email + group)
        report_dict = {}

        # First collect opens
        for entry in open_data:
            key = (entry.email, entry.group)
            report_dict[key] = {
                "email": entry.email,
                "group": entry.group,
                "opened": True,
                "opened_at": str(entry.opened_at),
                "clicked": False,
                "clicked_at": None,
                "target_url": entry.target_url
            }

        # Now collect clicks and update or add
        for entry in click_data:
            key = (entry.email, entry.group)
            if key in report_dict:
                report_dict[key]["clicked"] = True
                report_dict[key]["clicked_at"] = str(entry.clicked_at)
            else:
                report_dict[key] = {
                    "email": entry.email,
                    "group": entry.group,
                    "opened": False,
                    "opened_at": None,
                    "clicked": True,
                    "clicked_at": str(entry.clicked_at),
                    "target_url": entry.target_url
                }

        logger.info(f"✅ Report generated with {len(report_dict)} entries")
        return list(report_dict.values())

    except Exception as e:
        logger.error(f"❌ Failed to generate report: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Failed to generate report"})

@app.get("/download-db")
async def download_db():
    return FileResponse(TOKEN_DB, media_type='application/octet-stream', filename="your_database.db")
    
@app.get("/")
def root():
    return {"message": "✅ AI Email Assistant Backend is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
