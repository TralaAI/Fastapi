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
    temperature_celcius: int
    label: str

class ModelRawData(BaseModel):
    timestamp: str
    type: str
    holiday: bool
    weather: str
    temperature: int

modelin = joblib.load("./AI/decision_tree.pkl")
litter_types = ["plastic", "paper", "metal", "glass", "organic"]

@app.post("/predict")
def predict(inputs: List[ModelInput]):
    features = np.array([
        [
            inp.day_of_week,
            inp.month,
            int(inp.holiday),
            inp.weather,
            inp.temperature_celcius
        ] for inp in inputs
    ], dtype=np.float32)

    preds = modelin.predict(features)  # Assume shape (n_samples, 5)

    litter_types = ["plastic", "paper", "metal", "glass", "organic"]

    predictions = []
    for inp, pred_row in zip(inputs, preds):
        # Map each litter type to its predicted value for that input
        litter_prediction = {lt: float(pred) for lt, pred in zip(litter_types, pred_row)}
        predictions.append({inp.label: litter_prediction})

    return {"predictions": predictions}


# TODO May add endpoint to retrain the model on database data

# @app.post("/retrain")