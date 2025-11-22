from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from data_pipeline import DataPipeline
from price_optimizer import PriceOptimizer
import json

app = FastAPI(title="Fuel Price Optimization API", version="1.0.0")

class TodayInput(BaseModel):
    date: str
    price: float  # Last observed price
    cost: float   # Today's cost
    comp1_price: float
    comp2_price: float
    comp3_price: float

class PriceRecommendation(BaseModel):
    recommended_price: float
    expected_volume: int
    expected_profit: float
    expected_margin: float
    expected_margin_pct: float
    price_vs_competitors: float

# Global variables for loaded models
pipeline = None
optimizer = None
historical_df = None

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global pipeline, optimizer, historical_df
    
    try:
        # Initialize pipeline
        pipeline = DataPipeline()
        
        # Load historical data
        historical_df = pipeline.load_historical_data('../data/oil_retail_history.csv')
        historical_df = pipeline.engineer_features(historical_df)
        historical_df = pipeline.prepare_features(historical_df)
        
        # Load trained model
        optimizer = PriceOptimizer()
        optimizer.load_model('../models/price_optimizer.pkl')
        
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.get("/")
async def root():
    return {"message": "Fuel Price Optimization API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if optimizer is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "healthy", "models_loaded": True}

@app.post("/optimize-price", response_model=PriceRecommendation)
async def optimize_price(today_input: TodayInput):
    """Get optimal price recommendation"""
    if optimizer is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert input to dictionary
        today_data = today_input.dict()
        
        # Process today's features
        today_features = pipeline.process_today_input(today_data, historical_df)
        
        # Get competitor prices
        comp_prices = [today_data['comp1_price'], today_data['comp2_price'], today_data['comp3_price']]
        
        # Optimize price
        result = optimizer.optimize_price(
            features=today_features,
            cost=today_data['cost'],
            last_price=today_data['price'],
            comp_prices=comp_prices
        )
        
        return PriceRecommendation(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.post("/predict-volume")
async def predict_volume(today_input: TodayInput, price: float):
    """Predict volume for a specific price"""
    if optimizer is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert input to dictionary
        today_data = today_input.dict()
        
        # Process today's features
        today_features = pipeline.process_today_input(today_data, historical_df)
        
        # Predict volume
        volume = optimizer.predict_volume(today_features, price)
        profit = optimizer.calculate_profit(price, today_data['cost'], volume)
        
        return {
            "price": price,
            "predicted_volume": int(volume),
            "predicted_profit": round(profit, 2),
            "margin": round(price - today_data['cost'], 3)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)