from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pathlib import Path
import joblib 
import numpy as np
import pandas as pd
import subprocess

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / 'Data' / 'Example_insane.csv'
MODEL_PATH = BASE_DIR / 'AI' / 'Model.py'

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

class DataEnrichment(BaseModel):
    timestamp: str
    detected_object: str
    holiday: bool
    weather: str
    temperature_celsius: int

class RetrainRequest(BaseModel):
    data: List[DataEnrichment]
    parameter: str

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

@app.post("/retrain")
def retrain_model(request: RetrainRequest):
    data = request.data
    parameter = request.parameter 
    
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
                ['python3', str(MODEL_PATH), str(parameter)], 
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