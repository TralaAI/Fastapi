from fastapi import FastAPI

app = FastAPI()

class ModelInput(BaseModel):
    day_of_week: int
    month: int
    holiday: bool
    weather: int

@app.post("/predict")
def predict(input: ModelInput):
        model_input = [
        input.day_of_week,
        input.month,
        int(input.holiday),
        input.weather
    ]
        # Here you would typically call your model's prediction method