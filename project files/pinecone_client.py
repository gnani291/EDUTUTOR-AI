import os
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pinecone credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,  # adjust for your embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to index
index = pc.Index(INDEX_NAME)

# ✅ Core upsert function
def upsert_vector(vector_id: str, values: list[float], metadata: dict = None):
    index.upsert(vectors=[{
        "id": vector_id,
        "values": values,
        "metadata": metadata or {}
    }])

# ✅ Core query function
def query_vector(values: list[float], top_k: int = 5):
    return index.query(vector=values, top_k=top_k, include_metadata=True)

# ✅ Store user profile embedding
def store_user_profile_embedding(user_id: str, embedding: list[float], metadata: dict):
    vector_id = f"user_{user_id}"
    upsert_vector(vector_id=vector_id, values=embedding, metadata=metadata)

# ✅ Store quiz metadata after submission
def store_quiz_metadata(user_id: str, topic: str, score: int, embedding: list[float]):
    timestamp = datetime.now().isoformat()
    vector_id = f"quiz_{user_id}_{timestamp}"
    metadata = {
        "user_id": user_id,
        "topic": topic,
        "score": score,
        "date": timestamp
    }
    upsert_vector(vector_id=vector_id, values=embedding, metadata=metadata)

# ✅ Retrieve quiz history for educator dashboard
def get_user_quiz_history(user_id: str, top_k: int = 50):
    return index.query(
        vector=[],
        filter={"user_id": user_id},
        top_k=top_k,
        include_metadata=True
    )["matches"]
