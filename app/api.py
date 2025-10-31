from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Загрузка модели и порога при старте
MODEL_PATH = "models/lightgbm_model.pkl"
THRESHOLD_PATH = "models/lightgbm_best_threshold.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
best_threshold = joblib.load(THRESHOLD_PATH)
feature_names = model.feature_name_  # список признаков, на которых обучалась модель

print(f"✅ Модель загружена. Порог: {best_threshold:.4f}. Признаков: {len(feature_names)}")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not isinstance(data, list):
            data = [data]  # поддержка одного или нескольких объектов

        # Создаём DataFrame с правильными колонками
        X = pd.DataFrame(data)[feature_names]
        
        # Предсказания
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= best_threshold).astype(int)

        result = [
            {"prediction": int(p), "risk_probability": float(pr)}
            for p, pr in zip(pred, proba)
        ]
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)