# app.py

import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

app = Flask(__name__)

messages = [
    {"role": "system", "content": "You are a friendly, concise assistant. Keep replies short and clear."}
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_text = request.json["message"]
    messages.append({"role": "user", "content": user_text})

    try:
        resp = client.chat_completion(
            messages=messages,
            max_tokens=300,
            temperature=0.7,
            top_p=0.9
        )

        bot_text = resp.choices[0].message["content"]
        messages.append({"role": "assistant", "content": bot_text})

        return jsonify({"reply": bot_text})

    except Exception as e:
        return jsonify({"reply": "Error: " + str(e)})

if __name__ == "__main__":
    app.run(debug=True)
