from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask_cors import CORS
import torch
import torch.nn.functional as F

import pandas as pd
import pickle
import numpy as np
from Recommend_Anime.ml_recommendationsystemanime import *

app = Flask(__name__)
CORS(app, supports_credentials=True) 

model_path = "./toxic_classification/toxic-model-xlm-roberta/checkpoint-375" # Please change this depends on your model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# --- API Endpoint ---
@app.route("/recommend_collaborative", methods=["POST"])
def recommend():
    data = request.json
    print("📥 Received from frontend:", data)
    watched_anime = data.get("watched_anime", [])
    top_n = data.get("top_n", 5)

    rekomendasi = recommend_collaborative(watched_anime, top_n=top_n)
    print("📤 Sending back:", rekomendasi)
    return jsonify({"recommendations": rekomendasi})


@app.route('/recommend_anime', methods=['POST'])
def recommend_anime_api():
    data = request.get_json()
    title = data.get("title", "")
    results = recommend_anime(title, top_n=12)
    return jsonify({"recommendations": results})

@app.route("/toxic_classification", methods=["POST"])
def predict():
    data = request.get_json()
    texts = data.get("texts") or data.get("text")

    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        return jsonify({"error": "No text provided"}), 400
    
    device = next(model.parameters()).device

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        predicted_classes = torch.argmax(probs, dim=1).tolist()
        confidences = torch.max(probs, dim=1).values.tolist()

    results = []
    for i in range(len(texts)):
        results.append({
            "text": texts[i],
            "predicted_class": predicted_classes[i],
            "label": "toxic" if predicted_classes[i] == 1 else "non-toxic",
            "confidence": confidences[i]
        })

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, port=5000)