"""
Web interface for PDR Bot — Telegram-like chat UI.

Run:
    python web.py

Then open: http://localhost:5000
"""

import os
import json
import sys
from dotenv import load_dotenv

try:
    from flask import Flask, render_template, request, jsonify
except ImportError:
    print("Ошибка: Flask не установлен. Выполните: pip install flask")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("Ошибка: openai не установлен. Выполните: pip install openai")
    sys.exit(1)

load_dotenv()

app = Flask(__name__)

PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "prompts.json")
DEFAULT_MODEL = "gpt-3.5-turbo"


def load_prompts() -> list[dict]:
    if not os.path.exists(PROMPTS_FILE):
        return []
    try:
        with open(PROMPTS_FILE, encoding="utf-8") as f:
            return json.load(f).get("prompts", [])
    except (json.JSONDecodeError, OSError) as e:
        print(f"Ошибка чтения {PROMPTS_FILE}: {e}")
        return []


def build_system_message(prompt: dict) -> str:
    parts = [
        "СТРОГОЕ ОГРАНИЧЕНИЕ: Ты являешься виртуальным менеджером автосервиса по удалению вмятин без покраски (PDR). "
        "Ты НЕ являешься универсальным ассистентом и НЕ умеешь делать ничего, кроме консультации по услугам PDR. "
        "На любой вопрос вне темы услуг отвечай только: что занимаешься исключительно удалением вмятин без покраски. "
        "Никогда не перечисляй свои возможности как ИИ и не обсуждай темы вне автосервиса."
    ]
    if prompt.get("role"):
        parts.append(prompt["role"] + ".")
    if prompt.get("context"):
        parts.append("Контекст: " + prompt["context"] + ".")
    if prompt.get("question"):
        parts.append("Задача: " + prompt["question"] + ".")
    if prompt.get("format"):
        parts.append("Формат ответа: " + prompt["format"] + ".")
    return "\n".join(parts)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/prompts")
def get_prompts():
    return jsonify(load_prompts())


@app.route("/api/model")
def get_model():
    return jsonify({"model": os.getenv("OPENAI_MODEL", DEFAULT_MODEL)})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = (data.get("message") or "").strip()
    history = data.get("history") or []
    prompt_id = data.get("prompt_id")

    if not user_message:
        return jsonify({"error": "Пустое сообщение"}), 400

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY не задан в .env"}), 500

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    prompts = load_prompts()
    selected = next((p for p in prompts if p["id"] == prompt_id), None)

    messages = []
    if selected:
        messages.append({"role": "system", "content": build_system_message(selected)})

    for msg in history:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_completion_tokens=1024,
            frequency_penalty=0.5,
        )
        reply = response.choices[0].message.content
        usage = response.usage
        return jsonify({
            "reply": reply,
            "tokens": {
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
                "total": usage.total_tokens,
            },
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("WEB_PORT", 5000))
    print("=" * 50)
    print(f"  PDR Bot Web UI  →  http://localhost:{port}")
    print("=" * 50)
    app.run(debug=False, port=port, host="0.0.0.0")
