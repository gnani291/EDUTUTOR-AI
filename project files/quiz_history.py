import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import os
import json

# Initialize Pinecone
pinecone = Pinecone(api_key="YOUR_API_KEY")  # Replace with your actual API key
index = pinecone.Index("quiz-submissions")  # Replace with your actual index name

# Utility function to fetch quiz history for a user
def fetch_quiz_history(user_id):
    try:
        results = index.query(
            namespace="quiz-history",
            filter={"user_id": {"$eq": user_id}},
            top_k=100,
            include_metadata=True
        )

        if not results.matches:
            return []

        # Extract metadata from results
        history = []
        for match in results.matches:
            metadata = match['metadata']
            history.append({
                "timestamp": metadata.get("timestamp", "Unknown"),
                "score": metadata.get("score", "N/A"),
                "questions": metadata.get("questions", []),
                "answers": metadata.get("answers", [])
            })
        return history
    except Exception as e:
        st.error(f"Error fetching history: {str(e)}")
        return []

# Streamlit App
def main():
    st.title("📜 Quiz History")

    # Ask user for email or ID (in real app, this should come from login session)
    user_id = st.text_input("Enter your Email or Student ID", key="quiz_history_input")

    if st.button("Fetch Quiz History"):
        if not user_id:
            st.warning("Please enter your email or student ID.")
            return

        with st.spinner("Fetching your quiz history..."):
            history = fetch_quiz_history(user_id)

        if not history:
            st.info("No quiz history found for this user.")
        else:
            for i, record in enumerate(sorted(history, key=lambda x: x["timestamp"], reverse=True)):
                with st.expander(f"🗓️ Attempt on {record['timestamp']} | Score: {record['score']}"):
                    for q_num, (q, a) in enumerate(zip(record["questions"], record["answers"]), start=1):
                        st.markdown(f"**Q{q_num}:** {q}")
                        st.markdown(f"**Your Answer:** {a}")
                    st.markdown("---")

if __name__ == "__main__":
    main()
