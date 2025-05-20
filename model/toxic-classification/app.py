from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = Flask(__name__)

model_path = ".\model\toxic-classification\model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    texts = data.get("texts", [])

    if not texts:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    return jsonify({
    "text": texts,
    "predicted_class": predicted_class,
    "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
