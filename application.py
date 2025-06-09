import pickle as pkl
from fastapi import FastAPI, HTTPException
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import Any, List

app = FastAPI()

# Load your ML model
with open("model.pkl", "rb") as f:
    model = pkl.load(f)

# Authentication dependency
security = HTTPBearer()

class PredictRequest(BaseModel):
    features: List[Any]

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        prediction = model.predict([request.features])
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))