from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# 1. Initialize FastAPI
app = FastAPI(title="CarMarket Pro AI Service")

# 2. DYNAMIC PATHING (The Fix)
# This finds the folder where main.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# This joins that folder with the 'models' subfolder
MODEL_PATH = os.path.join(BASE_DIR, "models", "car_price_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoders.pkl")

# Load the artifacts using the full absolute paths
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)

# 3. Define Input Format
class CarFeatures(BaseModel):
    brand: str
    model: str
    year: int
    engine_size: float
    fuel_type: str
    transmission: str
    mileage: int
    car_type: str
    drive_type: str

@app.get("/")
def home():
    return {
        "message": "Welcome to CarMarket Pro AI Pricing Engine",
        "status": "Online",
        "currency": "USD"
    }

@app.post("/predict")
def predict_car_price(car: CarFeatures):
    input_dict = car.model_dump() 
    df_input = pd.DataFrame([input_dict])
    
    for col, le in encoders.items():
        df_input[col] = le.transform(df_input[col])
    
    raw_prediction = model.predict(df_input)[0]
    
    # Currency Conversion (PKR to USD)
    usd_price = raw_prediction / 280
    
    return {
        "estimated_price_usd": round(float(usd_price), 2),
        "currency": "USD",
        "formatted_price": f"${round(float(usd_price), 2):,.2f}"
    }