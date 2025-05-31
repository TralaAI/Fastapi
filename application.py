from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Any, List
from auth import verify_token
import pickle

app = FastAPI()

# Load your ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Authentication dependency
security = HTTPBearer()

class PredictRequest(BaseModel):
    features: List[Any]

@app.post("/predict")
def predict(
    request: PredictRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    verify_token(credentials.credentials)
    try:
        prediction = model.predict([request.features])
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))