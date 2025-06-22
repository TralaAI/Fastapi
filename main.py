from sqlalchemy import MetaData, Table, create_engine, text
from starlette.middleware.base import BaseHTTPMiddleware
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

def get_unique_camera_ids():
    query = text("SELECT DISTINCT CameraId FROM litters")
    with engine.connect() as conn:
        result = conn.execute(query)
        camera_ids = [row[0] for row in result.fetchall()]
    return camera_ids

def train_initial_models(camera_ids: List[int]):
    for camera_id in camera_ids:
        try:
            ModelGenerator.train_and_save_model(camera_id)
            print(f"Model trained and saved for Camera ID: {camera_id}")
            print()
        except Exception as e:
            print(f"Error training model for Camera ID {camera_id}: {e}")

def get_last_updated_time(camera_id: int):
    model_path = BASE_DIR / 'AI_Models' / f"Camera{camera_id}_tree.pkl"
    if model_path.exists():
        return datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc)
    return None

camera_ids = get_unique_camera_ids()
train_results = [ModelGenerator.train_and_save_model(camera_id) for camera_id in camera_ids]

# Variables for status endpoint for 5 different models (indexed by cameraId)
model_status = {
    int(result["camera"]): {
        "current_rmse": float(result["rmse"]),
        "last_updated": get_last_updated_time(int(result["camera"]))
    }
    for result in train_results
}

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

class RetrainRequest(BaseModel):
    cameraLocation: int
class ModelStatusResponse(BaseModel):
    status: str
    current_rmse: float
    last_updated: datetime

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
def retrain_model(request: RetrainRequest):
    print(request.cameraLocation)
    ModelGenerator.train_and_save_model(request.cameraLocation)
    return {"status": "success"}

@app.get("/status")
def status():
        return {"status": "API is running"}

@app.get("/status/model")
def status(cameraId: int = Query(..., description="Camera ID to check model status for")):
    pkl_path = BASE_DIR / 'AI_Models' / f"Camera{cameraId}_tree.pkl"
    if not pkl_path.exists():
        return JSONResponse({"error": f"Model file not found for Camera ID {cameraId}"}, status_code=404)
    if cameraId not in model_status:
        return JSONResponse({"error": f"No status found for Camera ID {cameraId}"}, status_code=404)
    
    status_info = model_status[cameraId]
    return ModelStatusResponse(
        status="success",
        current_rmse=status_info["current_rmse"],
        last_updated=status_info["last_updated"]
    )