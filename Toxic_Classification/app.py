from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask_cors import CORS
import torch
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)

model_path = "./toxic-model-xlm-roberta/checkpoint-375" # Please change this depends on your model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

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
    app.run(debug=True, port=5001)