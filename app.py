from flask import Flask, request, jsonify
import numpy as np
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "models/lstm_model_prediction.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Make sure it's present.")

model = load_model(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "LSTM Flask API running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    if "data" not in payload:
        return jsonify({"error": "Missing 'data' key"}), 400

    arr = np.array(payload["data"])
    # reshape for LSTM input: batch, timesteps, features
    if arr.ndim == 1:
        arr = arr.reshape((1, arr.shape[0], 1))
    elif arr.ndim == 2:
        arr = arr.reshape((1, arr.shape[0], arr.shape[1]))

    try:
        preds = model.predict(arr)
        return jsonify({"predictions": preds.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
