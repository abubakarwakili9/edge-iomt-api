
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define input schema
class SensorInput(BaseModel):
    Heart_Rate_bpm: float
    BatteryRatio: float
    Temperature_C: float
    Device_Battery_Level: float
    Battery_Level: float
    HealthPressureGap: float
    Target_Heart_Rate: float

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Health Status Prediction API is running."}

@app.post("/predict")
def predict(data: SensorInput):
    input_array = np.array([
        data.Heart_Rate_bpm,
        data.BatteryRatio,
        data.Temperature_C,
        data.Device_Battery_Level,
        data.Battery_Level,
        data.HealthPressureGap,
        data.Target_Heart_Rate
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    return {"prediction": "Unhealthy" if prediction == 1 else "Healthy"}
