import os
import requests
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("WATSONX_API_KEY")

if not api_key:
    raise ValueError("WATSONX_API_KEY not found in environment variables.")

url = "https://iam.cloud.ibm.com/identity/token"

response = requests.post(
    url,
    data={
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key
    },
    headers={"Content-Type": "application/x-www-form-urlencoded"}
)

print(response.status_code)
print(response.text)





