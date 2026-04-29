import streamlit as st
from pinecone_client import get_quiz_by_topic


st.set_page_config(page_title="EduTutor Quiz Viewer", layout="centered")
st.title("📚 EduTutor - View Quiz by Topic")

topic = st.text_input("Enter a quiz topic:", placeholder="e.g., Machine Learning")

if st.button("Fetch Quiz"):
    if not topic.strip():
        st.warning("Please enter a topic to search.")
    else:
        with st.spinner("Fetching quiz from Pinecone..."):
            quizzes = get_quiz_by_topic(topic.strip())

        if not quizzes:
            st.error("❌ No quiz found for this topic.")
        else:
            st.success(f"✅ Found {len(quizzes)} quiz item(s) for topic: {topic}")
            for i, quiz in enumerate(quizzes):
                st.markdown(f"### 🔹 Question {i + 1}")
                st.markdown(f"**Q:** {quiz['question']}")
                for option in quiz['options']:
                    st.markdown(f"- {option}")
                st.markdown(f"**✅ Answer:** {quiz['answer']}")
                st.caption(f"🔎 Relevance Score: {quiz['score']:.4f}")

