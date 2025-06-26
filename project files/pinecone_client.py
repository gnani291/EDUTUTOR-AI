import os
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pinecone credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,  # ⚠️ Updated to match embedding size
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


# ✅ Store user profile embedding
def store_user_profile_embedding(user_id: str, embedding: list[float], metadata: dict):
    vector_id = f"user_{user_id}"
    upsert_vector(vector_id=vector_id, values=embedding, metadata=metadata)


# ✅ Store quiz metadata with full info
def store_quiz_metadata(user_id: str, topic: str, score: int, embedding: list[float],
                        questions: list[str] = None, answers: list[str] = None):
    timestamp = datetime.now().isoformat()
    vector_id = f"quiz_{user_id}_{timestamp}"
    metadata = {
        "user_id": user_id,
        "topic": topic,
        "score": score,
        "timestamp": timestamp,
        "questions": questions or [],
        "answers": answers or []
    }
    upsert_vector(vector_id=vector_id, values=embedding, metadata=metadata)


# ✅ Retrieve quiz history for a user
def get_user_quiz_history(user_id: str, top_k: int = 50):
    try:
        results = index.query(
            vector=[0.0] * 1024,  # dummy vector with correct dimension
            filter={"user_id": {"$eq": user_id}},
            top_k=top_k,
            include_metadata=True
        )

        history = []
        for match in results.matches:
            metadata = match.get("metadata", {})
            history.append({
                "timestamp": metadata.get("timestamp", "Unknown"),
                "score": metadata.get("score", "N/A"),
                "topic": metadata.get("topic", "N/A"),
                "questions": metadata.get("questions", []),
                "answers": metadata.get("answers", [])
            })
        return history
    except Exception as e:
        print(f"❌ Error fetching history: {e}")
        return []
