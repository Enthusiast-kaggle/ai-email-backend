import gspread
from oauth2client.service_account import ServiceAccountCredentials
from string import Template

# Reuse your existing email function (make sure this import is correct)
from email_utils import send_email  # ✅ Correct import


SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SHEET_CREDENTIAL_FILE = "sheets_credentials.json"

def fetch_sheet_data(sheet_url):
    creds = ServiceAccountCredentials.from_json_keyfile_name(SHEET_CREDENTIAL_FILE, SCOPES)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(sheet_url)
    worksheet = sheet.get_worksheet(0)  # Use the first sheet
    data = worksheet.get_all_records()
    return data

from string import Template
from email_utils import send_email  # Make sure this version accepts client_token_data

def perform_mail_merge(sheet_url, subject_template, body_template, client_token_data):
    rows = fetch_sheet_data(sheet_url)
    results = []

    for row in rows:
        try:
            subject = Template(subject_template).safe_substitute(row)
            body = Template(body_template).safe_substitute(row)

            # Try both "Email" and "email" to be case-insensitive
            recipient = row.get("Email") or row.get("email")

            if recipient:
                send_email(recipient, subject, body, client_token_data=client_token_data)
                results.append({"email": recipient, "status": "sent"})
            else:
                results.append({"status": "skipped: no email found in row"})
        except Exception as e:
            results.append({"email": row.get("Email", "unknown"), "status": f"error: {str(e)}"})

    return results
