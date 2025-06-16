from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib 
import numpy as np

app = FastAPI()

class ModelInput(BaseModel):
    day_of_week: int
    month: int
    holiday: bool
    weather: int

# ✅ Load model with joblib
modelin = joblib.load("./AI/decision_tree.pkl")
@app.post("/predict")
def predict(inputs: List[ModelInput]):
    # Prepare features for prediction
    features = np.array([
        [
            inp.day_of_week,
            inp.month,
            int(inp.holiday),
            inp.weather
        ] for inp in inputs
    ], dtype=np.float32)

    # Make prediction
    preds = modelin.predict(features)

    return {"predictions": preds.tolist()}
