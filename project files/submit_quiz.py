import requests

# Test embedding (same shape, dummy data)
embedding = [round(0.005 * i, 4) for i in range(1024)]  # Slightly different vector

# New quiz payload
payload = {
    "user_id": "user_001",
    "topic": "Computer Vision",
    "score": 7,
    "embedding": embedding
}

response = requests.post("http://localhost:8000/submit-quiz", json=payload)
print("Status Code:", response.status_code)
print("Response:", response.json())
