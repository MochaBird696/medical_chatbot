import os
import json
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ─── Load environment variables ────────────────────────────────────────────────
load_dotenv()  # reads .env into os.environ
# If you need to authenticate private models:
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# ─── App & Model setup ────────────────────────────────────────────────────────
app = Flask(__name__)

MODEL_NAME = os.getenv("HF_MODEL", "google/flan-t5-base")

# auto-detect GPU vs CPU
device = 0 if torch.cuda.is_available() else -1

# load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# build the chat pipeline
chatbot = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_length=256,
    do_sample=False,
)

# simple in‐memory session store
sessions = {}

# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")  # make sure this exists

@app.route("/chat", methods=["POST"])
def chat():
    data       = request.get_json(force=True)
    session_id = data.get("session_id")
    user_msg   = data.get("message")

    # 1) Initialize session with system prompt if first time
    if session_id not in sessions:
        sessions[session_id] = [{
            "role": "system",
            "content": (
              "You are MediChat, a medical assistant. "
              "Ask structured follow-up questions (return JSON with “question” and “options”) "
              "until you can propose a diagnosis."
            )
        }]

    history = sessions[session_id]
    history.append({"role": "user", "content": user_msg})

    # 2) Build prompt from history
    prompt = ""
    for turn in history:
        prompt += f"{turn['role']}: {turn['content']}\n"
    prompt += "assistant:"

    # 3) Generate
    raw = chatbot(prompt)[0]["generated_text"].strip()

    # 4) Try to parse JSON; if that fails, treat as free text
    try:
        reply = json.loads(raw)
        history.append({"role": "assistant", "content": raw})
        return jsonify({"structured": reply})
    except ValueError:
        # plain text fallback
        history.append({"role": "assistant", "content": raw})
        return jsonify({"reply": raw})


# ─── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
