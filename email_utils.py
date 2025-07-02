# email_utils.py
import base64
import os
import sqlite3
import uuid
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKING_DB = os.path.join(BASE_DIR, "tracking_logs.db")

def add_tracking_to_body(body: str, email_id: str) -> str:
    formatted_body = body.replace("\n", "<br>")
    tracking_url = f"http://192.168.53.99:8000/track/open/{email_id}"  # Replace with your real server IP/domain

    tracking_img = f'<img src="{tracking_url}" width="1" height="1" style="display:none;" alt="." />'
    html_body = f"""
    <html>
        <body>
            {formatted_body}
            {tracking_img}
        </body>
    </html>
    """
    return html_body

def send_email(recipient: str, subject: str, body: str, client_token_data: dict):
    try:
        creds = Credentials.from_authorized_user_info(client_token_data)
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
            INSERT INTO tracking_logs (id, recipient, subject, sent_time)
            VALUES (?, ?, ?, ?)
        ''', (email_id, recipient, subject, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ DB Error: {e}")
