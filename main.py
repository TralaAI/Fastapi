from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy import MetaData, Table, create_engine
from starlette.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, Field
from fastapi.requests import Request
from sqlalchemy.sql import select
from dotenv import load_dotenv
from datetime import datetime
from fastapi import FastAPI
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

# load_dotenv()
# connection_string = os.getenv("connStr")
# engine = create_engine(connection_string)
# query = "SELECT * FROM FApiKeys"

# with engine.connect() as conn:
#     result = conn.execute(text(query)).fetchone()

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
    key: uuid = Field(default_factory=uuid.uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
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
    modelIndex: str

class DataEnrichment(BaseModel):
    timestamp: str
    detected_object: str
    holiday: bool
    weather: str
    temperature_celsius: int

class RetrainRequest(BaseModel):
    data: List[DataEnrichment]
    cameraLocation: str

litter_types = ["plastic", "paper", "metal", "glass", "organic"]

@app.post("/predict")
def predict(request: ModelInputRequest):
    inputs = request.inputs
    modelIndex = request.modelIndex

    match modelIndex:
        case '0':
            pkl_path = BASE_DIR / 'AI_Models' / 'developing_phase_tree.pkl'
        case '1':
            pkl_path = BASE_DIR / 'AI_Models' / 'sensoring_group_tree.pkl'
        case '2':
            pkl_path = BASE_DIR / 'AI_Models' / 'generated_city_tree.pkl'
        case '3':
            pkl_path = BASE_DIR / 'AI_Models' / 'generated_industrial_tree.pkl'
        case '4':
            pkl_path = BASE_DIR / 'AI_Models' / 'generated_suburbs_tree.pkl'
        case _:
            return {"error": "Invalid modelIndex"}

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

    litter_types = ["plastic", "paper", "metal", "glass", "organic"]
    predictions = []
    for inp, pred_row in zip(inputs, preds):
        litter_prediction = {lt: float(pred) for lt, pred in zip(litter_types, pred_row)}
        predictions.append({inp.label: litter_prediction})

    return {"predictions": predictions}


@app.post("/retrain")
def retrain_model(request: RetrainRequest):
    data = request.data
    cameraLocation = request.cameraLocation 

    match cameraLocation:
        case '0':
            CSV_PATH = BASE_DIR / 'Data' / '0_developing_data.csv'
        case '1':
            CSV_PATH = BASE_DIR / 'Data' / '1_sensoring_data.csv'
        case '2':
            CSV_PATH = BASE_DIR / 'Data' / '2_city_data.csv'
        case '3':
            CSV_PATH = BASE_DIR / 'Data' / '3_industrial_data.csv'
        case '4':
            CSV_PATH = BASE_DIR / 'Data' / '4_suburbs_data.csv'
        case _:
            return {"error": "Invalid cameraLocation"}
    
    df_existing = pd.read_csv(CSV_PATH)
    last_id = df_existing['id'].max()

    new_data = pd.DataFrame([item.dict() for item in data])
    new_data['holiday'] = new_data['holiday'].astype(int)
    new_data.insert(0, 'id', range(last_id + 1, last_id + 1 + len(new_data)))
    new_data = new_data[['id', 'detected_object', 'timestamp', 'weather', 'temperature_celsius', 'holiday']]
    new_data.to_csv(CSV_PATH, mode='a', index=False, header=False)

    if MODEL_PATH.exists():
        try:
            result = subprocess.run(
                ['python3', str(MODEL_PATH), str(cameraLocation)], 
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