from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy import MetaData, Table, create_engine
import Model_Generator.Model as ModelGenerator
from starlette.responses import JSONResponse
from typing import Callable, List, Optional
from fastapi import FastAPI, Query, Request
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from sqlalchemy.sql import select
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import joblib
import uuid
import os

load_dotenv()

if not os.getenv("connStr"):
    raise RuntimeError("The 'connStr' environment variable is required but not set. Please set it before running the application.")

BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = os.getenv("connStr")
if not DATABASE_URL:
    raise ValueError("connStr environment variable is not set.")

engine = create_engine(DATABASE_URL)
metadata = MetaData()
api_keys = Table(
    "ApiKeys",
    metadata,
    autoload_with=engine,
    schema="dbo"  # Explicitly specify schema for SQL Server
)


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        x_api_key = request.headers.get("X-API-Key")
        if not x_api_key:
            return JSONResponse({"detail": "Missing API key header"}, status_code=401)

        try:
            with engine.connect() as conn:
                # Build query to validate API key: must match, be active, not expired, and of type 'fastapi'
                query = (
                    select(api_keys.c.Key)
                    .where(api_keys.c.Key == x_api_key)
                    .where(api_keys.c.IsActive == True)
                    .where(
                        (api_keys.c.ExpiresAt.is_(None)) |
                        (api_keys.c.ExpiresAt > datetime.now(timezone.utc))
                    )
                    .where(api_keys.c.Type == "fastapi")
                )
                result = conn.execute(query).fetchone()
                if not result:
                    return JSONResponse({"detail": "Invalid API key"}, status_code=401)
        except SQLAlchemyError as ex:
            print("Database error occurred while validating API key : ", ex)
            return JSONResponse({"detail": "Database error"}, status_code=500)

        return await call_next(request)

ModelGenerator.train_and_save_model('1')  # Initial training for sensoring group

app = FastAPI()
app.add_middleware(APIKeyMiddleware)

class APIKey(BaseModel):
    id: int
    key: uuid.UUID = Field(default_factory=uuid.uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    is_active: bool = True
    type: str = "fastapi"

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

litter_types = ["plastic", "paper", "metal", "glass", "organic"]

@app.post("/predict")
def predict(request: ModelInputRequest):
    inputs = request.inputs
    cameraId = request.cameraId

    pkl_path = BASE_DIR / 'AI_Models' / f"Camera{cameraId}_tree.pkl" 

    if not pkl_path.exists():
        return JSONResponse({"error": f"Model file not found: {pkl_path}"}, status_code=404)

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

    preds = modelin.predict(features)  # shape (n_samples, 5)

    results = []
    for inp, pred_row in zip(inputs, preds):
        litter_prediction = {lt: float(pred) for lt, pred in zip(litter_types, pred_row)}
        results.append({
            "date": inp.label,
            "predictions": litter_prediction
        })

    return results

@app.post("/retrain")
def retrain_model(cameraLocation: int = Query(..., description="Camera location ID")):
    ModelGenerator.train_and_save_model(cameraLocation)
    return {"status": "success"}

@app.get("/status")
def status():
        return {"status": "API is running"}