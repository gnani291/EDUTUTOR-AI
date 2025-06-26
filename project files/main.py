from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
from edututor.quiz_generator import generate_quiz
from pinecone_client import (
    store_quiz_metadata,
    get_user_quiz_history
)

app = FastAPI()


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


# ✅ Updated model to include questions & answers
class QuizSubmission(BaseModel):
    user_id: str
    topic: str
    score: int
    embedding: List[float]  # 1024-dim vector expected
    questions: List[str]
    answers: List[str]


@app.post("/submit-quiz")
def submit_quiz(data: QuizSubmission):
    try:
        store_quiz_metadata(
            user_id=data.user_id,
            topic=data.topic,
            score=data.score,
            embedding=data.embedding,
            questions=data.questions,
            answers=data.answers
        )
        return {"status": "success", "message": "Quiz stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user/{user_id}/quiz-history")
def get_quiz_history(user_id: str):
    try:
        history = get_user_quiz_history(user_id)
        return {"user_id": user_id, "quiz_history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

