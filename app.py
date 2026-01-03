from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Load summarization model (CPU only)
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1
)

MAX_WORDS = 5000

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Text is empty"}), 400

    if len(text.split()) > MAX_WORDS:
        return jsonify({"error": "Text exceeds 5000 words"}), 400

    # 1️⃣ Summary
    summary = summarizer(
        text,
        max_length=150,
        min_length=40,
        do_sample=False
    )[0]["summary_text"]

    # 2️⃣ Important Points (shorter summary)
    key_points = summarizer(
        text,
        max_length=90,
        min_length=30,
        do_sample=False
    )[0]["summary_text"]

    # 3️⃣ Questions (rule-based, CPU-safe)
    questions = [
        "What is the main idea of the paragraph?",
        "What are the key applications discussed?",
        "Why is this topic important?",
        "What methods are mentioned?",
        "Summarize the paragraph in your own words."
    ]

    return jsonify({
        "summary": summary,
        "important_points": key_points,
        "questions": questions
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5010)
