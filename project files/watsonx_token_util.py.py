import os
import requests
from dotenv import load_dotenv

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Get the API key from environment variable
api_key = os.getenv("WATSONX_API_KEY")

if not api_key:
    raise ValueError("WATSONX_API_KEY not found in environment variables.")

# ✅ IBM IAM token endpoint
url = "https://iam.cloud.ibm.com/identity/token"

# ✅ Request IAM token using API key
response = requests.post(
    url,
    data={
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key
    },
    headers={"Content-Type": "application/x-www-form-urlencoded"}
)

# ✅ Print response
print(response.status_code)
print(response.text)
