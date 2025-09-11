from fastapi import FastAPI, HTTPException, Request
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# Load the trained model at startup
MODEL_PATH = "models/lstm_model_prediction.h5"
model = load_model(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "LSTM API running!"}

@app.post("/predict")
async def predict(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if "data" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'data' key")

    arr = np.array(payload["data"])

    # Adjust shape for LSTM input: (batch, timesteps, features)
    if arr.ndim == 1:
        arr = arr.reshape((1, arr.shape[0], 1))
    elif arr.ndim == 2:
        arr = arr.reshape((1, arr.shape[0], arr.shape[1]))

    try:
        preds = model.predict(arr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {"predictions": preds.tolist()}
