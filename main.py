from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pathlib import Path
import joblib 
import numpy as np
import pandas as pd
import subprocess

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / 'Model_Generator' / 'Model.py'

app = FastAPI()

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
            CSV_PATH = BASE_DIR / 'Data' / 'developing_data.csv'
        case '1':
            CSV_PATH = BASE_DIR / 'Data' / 'sensoring_data.csv'
        case '2':
            CSV_PATH = BASE_DIR / 'Data' / 'city_data.csv'
        case '3':
            CSV_PATH = BASE_DIR / 'Data' / 'industrial_data.csv'
        case '4':
            CSV_PATH = BASE_DIR / 'Data' / 'suburbs_data.csv'
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