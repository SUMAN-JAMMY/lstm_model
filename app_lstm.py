from flask import Flask, request, jsonify
import numpy as np
import pickle
from lstm_student_prediction import model  # import your trained model

app = Flask(__name__)

# Load preprocessing objects if available
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
except:
    scaler, encoder = None, None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Student Prediction API is running with Flask!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)

        if scaler:
            features = scaler.transform(features)

        prediction = model.predict(features)
        probability = float(prediction[0][0])
        result = 1 if probability >= 0.5 else 0

        return jsonify({"prediction": result, "probability": probability})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

