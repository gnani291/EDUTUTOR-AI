from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from edututor.quiz_generator import generate_quiz
from pinecone_client import (
    store_quiz_metadata,
    get_user_quiz_history
)

app = FastAPI()

# ✅ Keep existing routes
@app.get("/")
def root():
    return {"message": "EduTutor AI Backend is running."}

@app.get("/quiz")
def get_quiz(topic: str = Query(..., description="Topic to generate quiz for")):
    try:
        quiz = generate_quiz(topic)
        return {"topic": topic, "quiz": quiz}
    except Exception as e:
        return {"error": str(e)}

# ✅ New: Pydantic model for quiz submission
class QuizSubmission(BaseModel):
    user_id: str
    topic: str
    score: int
    embedding: list[float]  # 768-dim vector

# ✅ New: Submit quiz result to Pinecone
@app.post("/submit-quiz")
def submit_quiz(data: QuizSubmission):
    try:
        store_quiz_metadata(
            user_id=data.user_id,
            topic=data.topic,
            score=data.score,
            embedding=data.embedding
        )
        return {"status": "success", "message": "Quiz stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ New: Get quiz history for a user
@app.get("/user/{user_id}/quiz-history")
def get_quiz_history(user_id: str):
    try:
        history = get_user_quiz_history(user_id)
        return {"user_id": user_id, "quiz_history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

