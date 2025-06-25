import os
import uuid
import pytz
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any
from typing import Union
from typing import Awaitable
from dotenv import load_dotenv
from sqlalchemy.sql import select
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from starlette.responses import Response
from sqlalchemy.exc import SQLAlchemyError
from fastapi import FastAPI, Query, Request
from typing import Callable, List, Optional
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy import MetaData, Table, create_engine, text

import Model as ModelGenerator

# --- Environment Setup ---
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = os.getenv("connStr")
if not DATABASE_URL:
    raise RuntimeError("The 'connStr' environment variable is required but not set. Please set it before running the application.")

# --- Database Setup ---
engine = create_engine(DATABASE_URL)
metadata = MetaData()
api_keys = Table(
    "ApiKeys",
    metadata,
    autoload_with=engine,
    schema="dbo"
)

# --- Middleware ---
class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        x_api_key = request.headers.get("X-API-Key")
        if not x_api_key:
            return JSONResponse({"detail": "Missing API key header"}, status_code=401)
        try:
            with engine.connect() as conn:
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
            print("Database error occurred while validating API key:", ex)
            return JSONResponse({"detail": "Database error"}, status_code=500)
        return await call_next(request)

# --- Utility Functions ---
def get_unique_camera_ids() -> List[int]:
    query = text("SELECT DISTINCT CameraId FROM litters")
    with engine.connect() as conn:
        result = conn.execute(query)
        return [row[0] for row in result.fetchall()]

def build_model_status(train_results: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    nl_tz = pytz.timezone("Europe/Amsterdam")
    return {
        int(result["camera"]): {
            "current_rmse": float(result["rmse"]),
            "last_updated": datetime.now(nl_tz)
        }
        for result in train_results
    }

# --- Model Training ---
def train_initial_models(camera_ids: List[int]):
    for camera_id in camera_ids:
        try:
            ModelGenerator.train_and_save_model(camera_id)
            print(f"Model trained and saved for Camera ID: {camera_id}")
        except Exception as e:
            print(f"Error training model for Camera ID {camera_id}: {e}")

camera_ids = get_unique_camera_ids()
train_results = [ModelGenerator.train_and_save_model(camera_id) for camera_id in camera_ids]
model_status = build_model_status(train_results)

# --- FastAPI App Setup ---
app = FastAPI()
app.add_middleware(APIKeyMiddleware)

# --- Pydantic Models ---
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

# --- Constants ---
LITTER_TYPES = ["glass", "metal", "organic", "paper", "plastic"]

# --- API Endpoints ---
@app.post("/predict")
def predict(request: ModelInputRequest):
    camera_id = request.cameraId
    pkl_path = BASE_DIR / 'AI_Models' / f"Camera{camera_id}_tree.pkl"
    if not pkl_path.exists():
        return JSONResponse({"error": f"Model file not found: {pkl_path}"}, status_code=404)
    try:
        model = joblib.load(pkl_path)  # type: ignore
    except Exception as e:
        return JSONResponse({"error": f"Failed to load model: {str(e)}"}, status_code=500)

    try:
        features = np.array([
            [
                inp.day_of_week,
                inp.month,
                int(inp.holiday),
                inp.weather,
                inp.temperature_celcius,
                inp.is_weekend,
            ] for inp in request.inputs
        ], dtype=np.float32)
    except Exception as e:
        return JSONResponse({"error": f"Invalid input data: {str(e)}"}, status_code=400)

    try:
        preds = model.predict(features)
    except Exception as e:
        return JSONResponse({"error": f"Prediction failed: {str(e)}"}, status_code=500)

    results: List[Dict[str, Any]] = []
    try:
        for inp, pred_row in zip(request.inputs, preds):
            litter_prediction = {lt: float(pred) for lt, pred in zip(LITTER_TYPES, pred_row)}
            results.append({
                "date": inp.label,
                "predictions": litter_prediction
            })
    except Exception as e:
        return JSONResponse({"error": f"Error formatting prediction results: {str(e)}"}, status_code=500)
    return JSONResponse(content=results, status_code=200)

@app.post("/retrain")
def retrain_model(request: RetrainRequest):
    try:
        retrain_result = ModelGenerator.train_and_save_model(request.cameraLocation)
        if not retrain_result or "rmse" not in retrain_result:
            return JSONResponse({"error": "Model retraining failed or invalid result."}, status_code=500)
        global model_status
        model_status = build_model_status([retrain_result])
        return JSONResponse({"status": "success"}, status_code=200)
    except FileNotFoundError:
        return JSONResponse({"error": f"Data or model file not found for Camera ID {request.cameraLocation}"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": f"Unexpected error: {str(e)}"}, status_code=500)

@app.get("/status")
def status():
    return {"status": "API is running"}

@app.get("/status/model")
def status_model(cameraId: int = Query(..., description="Camera ID to check model status for")):
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
