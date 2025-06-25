@echo off
echo 🚀 Starting EduTutor Backend...
start cmd /k "cd /d %cd% && uvicorn edututor.main:app --reload"

timeout /t 3 >nul

echo 🧠 Starting EduTutor Unified Frontend...
start cmd /k "cd /d %cd% && streamlit run streamlit_app.py"

echo ✅ All systems launched!
