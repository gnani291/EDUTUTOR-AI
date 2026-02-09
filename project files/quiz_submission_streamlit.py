import streamlit as st
import requests
from edututor.watsonx_client import generate_quiz  

st.set_page_config(page_title="EduTutor Quiz", layout="centered")
st.title("🧠 EduTutor - Take a Quiz")
if "user_id" not in st.session_state:
    st.session_state["user_id"] = st.text_input("👤 Enter your User ID", "user_001")
topic = st.text_input("📚 Enter topic for quiz:", "Machine Learning")
num_questions = st.slider("🔢 Number of questions:", 1, 10, 5)

if st.button("🚀 Generate Quiz"):
    with st.spinner("Generating quiz from WatsonX..."):
        try:
            quiz_data = generate_quiz(topic, num_questions)
            st.session_state["quiz"] = quiz_data
            st.session_state["submitted"] = False
        except Exception as e:
            st.error(f"❌ Failed to generate quiz: {e}")

if "quiz" in st.session_state and not st.session_state.get("submitted", False):
    st.header(f"📄 Quiz on: {topic}")
    answers = []

    with st.form("quiz_form"):
        for idx, q in enumerate(st.session_state["quiz"]):
            st.subheader(f"Q{idx+1}: {q['question']}")
            selected = st.radio(f"Choose your answer for Q{idx+1}", q["options"], key=f"q{idx}")
            answers.append((q, selected))

        submitted = st.form_submit_button("✅ Submit Quiz")

    if submitted:
        correct = 0
        result_display = []
        for idx, (q, selected) in enumerate(answers):
            selected_letter = selected.split(".")[0].strip()  # Extract option letter (e.g., "A")
            is_correct = selected_letter == q["answer"]
            if is_correct:
                correct += 1
            result_display.append((idx + 1, q["question"], selected, q["answer"], is_correct))

        st.session_state["submitted"] = True
        st.success(f"🎯 Your Score: {correct}/{len(answers)}")

        # Submit to backend
        embedding = [round(0.01 * i, 4) for i in range(1024)]  # Simulated embedding
        payload = {
            "user_id": st.session_state["user_id"],
            "topic": topic,
            "score": correct,
            "embedding": embedding
        }

        try:
            res = requests.post("http://localhost:8000/submit-quiz", json=payload)
            if res.status_code == 200:
                st.success("📝 Quiz data successfully stored in Pinecone!")
            else:
                st.error(f"❌ Failed to store quiz. Status code: {res.status_code}")
        except Exception as e:
            st.error(f"❌ Backend error: {e}")

        st.header("📊 Review:")
        for i, question, chosen, correct_ans, status in result_display:
            st.write(f"**Q{i}: {question}**")
            st.write(f"- Your Answer: `{chosen}`")
            st.write(f"- Correct Answer: `{correct_ans}`")
            st.markdown(f"- ✅ **Correct!**" if status else "- ❌ **Incorrect**")
            st.markdown("---")

if st.session_state.get("submitted", False):
    if st.button("🔄 Try Another Quiz"):
        del st.session_state["quiz"]
        st.session_state["submitted"] = False








