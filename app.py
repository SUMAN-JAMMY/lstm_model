from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "models/lstm_model_prediction.py"
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

    # Adjust shape for LSTM input: (batch, timesteps, features)
    if arr.ndim == 1:
        arr = arr.reshape((1, arr.shape[0], 1))
    elif arr.ndim == 2:
        arr = arr.reshape((1, arr.shape[0], arr.shape[1]))

    try:
        preds = model.predict(arr)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({"predictions": preds.tolist()})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT env variable
    app.run(host="0.0.0.0", port=port, debug=False)

