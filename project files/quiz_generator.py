from edututor.watsonx_client import get_model_response

def generate_quiz(topic: str):
    prompt = f"""
Generate exactly 5 multiple choice questions on the topic "{topic}" in strict JSON format.

[
  {{
    "question": "What is ...?",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "answer": "A"
  }},
  ...
]

Only return valid JSON. No explanation. No preface. No markdown. Just the JSON array.
"""
    return get_model_response(prompt, topic)


