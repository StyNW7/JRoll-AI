from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from ml_recommendationsystemanime import *

# Load model/data yang kamu pakai di notebook
# Contoh:
# df = pd.read_csv('../dataset_anime/anime-dataset-2023.csv')
# ratings = pd.read_csv('../dataset_anime/users-score-2023.csv')

# Kalau ada model hasil training/pickle, load di sini
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)

from flask_cors import CORS

app = Flask(__name__)  # Inisialisasi Flask
CORS(app, supports_credentials=True) 


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

if __name__ == "__main__":
    app.run(debug=True)
