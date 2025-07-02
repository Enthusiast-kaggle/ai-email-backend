from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import os

# Define the scopes your app needs (Gmail Send in this case)
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

def authenticate_gmail_redmicare():
    print("🚀 Starting Gmail authentication for redmicare72@gmail.com...")

    token_path = "token_redmicare.json"  # Store token separately
    creds_path = "client_secret_715127385680-cciu62m3slsre3f1cni7s729kufbs5ga.apps.googleusercontent.com.json"

    if not os.path.exists(creds_path):
        print(f"❌ Credential file {creds_path} not found!")
        return

    print("🌐 Opening OAuth flow for Redmicare with prompt='consent'")
    flow = InstalledAppFlow.from_client_secrets_file(
        creds_path,
        scopes=SCOPES
    )

    creds = flow.run_local_server(
        port=8081,  # Use a different port if 8080 is in use
        prompt='consent',
        access_type='offline'
    )

    with open(token_path, "w") as token:
        token.write(creds.to_json())

    print("✅ Gmail authentication for Redmicare complete. Token saved.")
    return creds

if __name__ == "_main_":
    authenticate_gmail_redmicare()