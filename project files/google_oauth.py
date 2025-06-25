import os
from authlib.integrations.requests_client import OAuth2Session

# ✅ Securely retrieve secrets from environment variables
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8501")

AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v3/userinfo"

SCOPES = [
    "openid", "email", "profile",
    "https://www.googleapis.com/auth/classroom.courses.readonly",
    "https://www.googleapis.com/auth/classroom.courseworkmaterials.readonly",
    "https://www.googleapis.com/auth/classroom.rosters.readonly"
]

def get_authorization_url():
    google = OAuth2Session(
        GOOGLE_CLIENT_ID,
        GOOGLE_CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES
    )
    uri, state = google.create_authorization_url(AUTHORIZATION_ENDPOINT)
    return uri, state

def get_user_info(code):
    google = OAuth2Session(
        GOOGLE_CLIENT_ID,
        GOOGLE_CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
    )
    token = google.fetch_token(TOKEN_ENDPOINT, code=code)

    # Reinitialize session with token for authenticated request
    google = OAuth2Session(
        GOOGLE_CLIENT_ID,
        GOOGLE_CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        token=token
    )
    resp = google.get(USERINFO_ENDPOINT)
    return resp.json(), token["access_token"]

