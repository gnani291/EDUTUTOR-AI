# watsonx_client.py
import os
import re
import json
import logging
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference

# Load .env variables
load_dotenv()

# WatsonX config
API_KEY = os.getenv("WATSONX_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
API_URL = os.getenv("WATSONX_URL")
MODEL_ID = os.getenv("WATSONX_MODEL_ID", "ibm/granite-13b-instruct-v2")

credentials = {
    "apikey": API_KEY,
    "url": API_URL,
}

logging.basicConfig(level=logging.INFO)

def parse_multiple_json_arrays(raw_text):
    try:
        # Find all separate JSON arrays
        matches = re.findall(r'\[\s*{.*?}\s*\]', raw_text, re.DOTALL)
        logging.info(f"🔍 Found {len(matches)} JSON arrays in output.")

        all_questions = []
        for i, block in enumerate(matches):
            try:
                data = json.loads(block)
                all_questions.extend(data)
            except json.JSONDecodeError as e:
                logging.warning(f"⚠️ Skipping malformed block {i+1}: {e}")

        return all_questions if all_questions else None

    except Exception as e:
        logging.error(f"❌ Error parsing JSON arrays: {e}")
        return None

def get_model_response(prompt, topic):
    try:
        model = ModelInference(
            model_id=MODEL_ID,
            credentials=credentials,
            project_id=PROJECT_ID
        )

        params = {
            "temperature": 0.3,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "decoding_method": "sample",
            "max_new_tokens": 1024
        }

        response = model.generate_text(prompt=prompt, params=params)

        if isinstance(response, str):
            generated_text = response.strip()
        elif isinstance(response, dict):
            generated_text = response.get("results", [{}])[0].get("generated_text", "").strip()
        else:
            generated_text = ""

        logging.info("\n🔍 RAW GENERATED TEXT:\n" + generated_text)

        if not generated_text:
            return {
                "topic": topic,
                "quiz": [],
                "error": "Model returned empty or unrecognized response.",
                "raw_response": str(response)
            }

        parsed = parse_multiple_json_arrays(generated_text)

        if not parsed:
            return {
                "topic": topic,
                "quiz": [],
                "error": "Failed to parse any valid questions.",
                "raw_text": generated_text
            }

        final = []
        for q in parsed:
            final.append({
                "question": q["question"],
                "options": {
                    "A": q["options"][0][3:],
                    "B": q["options"][1][3:],
                    "C": q["options"][2][3:],
                    "D": q["options"][3][3:]
                },
                "answer": q["answer"]
            })

        return {
            "topic": topic,
            "quiz": final
        }

    except Exception as e:
        logging.error(f"❌ Error generating quiz: {e}")
        return {
            "topic": topic,
            "quiz": [],
            "error": str(e)
        }

# ✅ Add this: Function to be used in Streamlit UI
def generate_quiz(topic: str, num_questions: int):
    prompt = f"""
    Generate {num_questions} multiple-choice questions on the topic "{topic}".
    Format the output as a single JSON list like:
    [
      {{
        "question": "...",
        "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
        "answer": "A"
      }},
      ...
    ]
    Only return valid JSON. No explanations.
    """

    result = get_model_response(prompt, topic)
    if "quiz" in result and isinstance(result["quiz"], list) and len(result["quiz"]) > 0:
        quiz_data = []
        for q in result["quiz"]:
            options = [
                f"A. {q['options']['A']}",
                f"B. {q['options']['B']}",
                f"C. {q['options']['C']}",
                f"D. {q['options']['D']}"
            ]
            quiz_data.append({
                "question": q["question"],
                "options": options,
                "answer": q["answer"]
            })
        return quiz_data
    else:
        raise Exception(result.get("error", "Unknown error during quiz generation."))

