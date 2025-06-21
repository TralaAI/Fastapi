from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy import MetaData, Table, create_engine
from starlette.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from fastapi.requests import Request
from sqlalchemy.sql import select
from dotenv import load_dotenv
from typing import Optional
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import subprocess
import joblib
import uuid
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'Model_Generator' / 'Model.py'
DATABASE_URL = os.getenv("connStr")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set.")

engine = create_engine(DATABASE_URL)
metadata = MetaData()
f_api_keys = Table("FApiKeys", metadata, autoload_with=engine)

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        x_api_key = request.headers.get("X-API-Key")
        if not x_api_key:
            return JSONResponse({"detail": "Missing API key header"}, status_code=401)

        try:
            with engine.connect() as conn:
                query = select(f_api_keys.c.Key).where(f_api_keys.c.Key == x_api_key)
                result = conn.execute(query).fetchone()
                if not result:
                    return JSONResponse({"detail": "Invalid API key"}, status_code=401)
        except SQLAlchemyError:
            return JSONResponse({"detail": "Database error"}, status_code=500)

        return await call_next(request)

app = FastAPI()
app.add_middleware(APIKeyMiddleware)

class APIKey(BaseModel):
    id: int
    key: uuid.UUID = Field(default_factory=uuid.uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    is_active: bool = True

class ModelInput(BaseModel):
    day_of_week: int
    month: int
    holiday: bool
    weather: int
    temperature_celcius: int
    is_weekend: bool
    label: str

class ModelInputRequest(BaseModel):
    inputs: List[ModelInput]
    cameraId: int

class DataEnrichment(BaseModel):
    timestamp: str
    detected_object: str
    holiday: bool
    weather: str
    temperature_celsius: int

class RetrainRequest(BaseModel):
    data: List[DataEnrichment]
    cameraId: int

litter_types = ["plastic", "paper", "metal", "glass", "organic"]

@app.post("/predict")
def predict(request: ModelInputRequest):
    inputs = request.inputs
    cameraId = request.cameraId

    pkl_path = BASE_DIR / 'AI_Models' / f"Camera{cameraId}_tree.pkl" 

    if not pkl_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found for Camera ID: {cameraId}")

    modelin = joblib.load(pkl_path)

    features = np.array([
        [
            inp.day_of_week,
            inp.month,
            int(inp.holiday),
            inp.weather,
            inp.temperature_celcius,
            inp.is_weekend,
        ] for inp in inputs
    ], dtype=np.float32)

    preds = modelin.predict(features)

    litter_types = ["plastic", "paper", "metal", "glass", "organic"]
    predictions = []
    for inp, pred_row in zip(inputs, preds):
        litter_prediction = {lt: float(pred) for lt, pred in zip(litter_types, pred_row)}
        predictions.append({inp.label: litter_prediction})

    return {"predictions": predictions}

# TODO: retrain needs to add new data to the database instead of CSV
@app.post("/retrain")
def retrain_model(request: RetrainRequest):
    cameraId = request.cameraId

    MODEL_PATH = BASE_DIR / 'Model_Generator' / 'Model.py'

    if MODEL_PATH.exists():
        try:
            result = subprocess.run(
                ['python3', str(MODEL_PATH), str(cameraId)], 
                capture_output=True,
                text=True,
                check=True
            )
            exec_output = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "message": "Model.py executed successfully."
            }
        except subprocess.CalledProcessError as e:
            exec_output = {"error": f"Execution failed: {e.stderr}"}
    else:
        exec_output = {"error": "Model.py not found"}

    return {
        "status": "success",
        "added_rows": len(new_data),
        "exec_output": exec_output
    }

@app.get("/status")
def status():
        return {"status": "API is running"}