from fastapi import FastAPI
import pickle

app = FastAPI()

# Load your ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(data: dict):
    # Convert input to model-friendly format
    prediction = model.predict([data["features"]])
    return {"prediction": prediction}