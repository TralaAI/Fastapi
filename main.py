from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

class ModelInput(BaseModel):
    day_of_week: int
    month: int
    holiday: bool
    weather: int

# Load once at startup
with open("./AI/decision_tree.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(input: ModelInput):
    # pack features in the order your model expects
    features = np.array([[
        input.day_of_week,
        input.month,
        int(input.holiday),
        input.weather
    ]], dtype=np.float32)

    # run prediction
    pred = model.predict(features)

    return {"prediction": float(pred[0])}