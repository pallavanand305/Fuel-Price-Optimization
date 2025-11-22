import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_pipeline import DataPipeline, load_today_data
from price_optimizer import PriceOptimizer
import json

def explore_data(df: pd.DataFrame):
    """Perform exploratory data analysis"""
    print("=== Data Exploration ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print("\nBasic statistics:")
    print(df[['price', 'cost', 'volume', 'comp1_price', 'comp2_price', 'comp3_price']].describe())
    
    # Calculate correlations
    corr_cols = ['price', 'cost', 'volume', 'comp1_price', 'comp2_price', 'comp3_price']
    correlations = df[corr_cols].corr()
    print("\nCorrelation matrix:")
    print(correlations.round(3))

def train_model():
    """Train the price optimization model"""
    print("=== Training Price Optimization Model ===")
    
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Load and process data
    # Try complete_data.csv first, fallback to generated data
    try:
        df = pipeline.load_historical_data('../data/complete_data.csv')
    except FileNotFoundError:
        df = pipeline.load_historical_data('../data/oil_retail_history.csv')
    print(f"Loaded {len(df)} records")
    
    # Explore data
    explore_data(df)
    
    # Engineer features
    df = pipeline.engineer_features(df)
    df = pipeline.prepare_features(df)
    
    print(f"Created {len(pipeline.feature_columns)} features")
    print("Features:", pipeline.feature_columns)
    
    # Initialize and train optimizer
    optimizer = PriceOptimizer(model_type='xgboost')
    optimizer.train_volume_model(df, pipeline.feature_columns)
    
    # Save model
    optimizer.save_model('../models/price_optimizer.pkl')
    print("Model saved successfully")
    
    return optimizer, pipeline, df

def predict_optimal_price(today_file: str = '../data/today_example.json'):
    """Predict optimal price for today"""
    print("=== Price Prediction ===")
    
    # Load today's data
    today_data = load_today_data(today_file)
    print("Today's input:", today_data)
    
    # Load historical data and pipeline
    pipeline = DataPipeline()
    historical_df = pipeline.load_historical_data('../data/oil_retail_history.csv')
    historical_df = pipeline.engineer_features(historical_df)
    historical_df = pipeline.prepare_features(historical_df)
    
    # Process today's features
    today_features = pipeline.process_today_input(today_data, historical_df)
    
    # Load trained model
    optimizer = PriceOptimizer()
    optimizer.load_model('../models/price_optimizer.pkl')
    
    # Get optimization results
    comp_prices = [today_data['comp1_price'], today_data['comp2_price'], today_data['comp3_price']]
    
    result = optimizer.optimize_price(
        features=today_features,
        cost=today_data['cost'],
        last_price=today_data['price'],
        comp_prices=comp_prices
    )
    
    print("\n=== OPTIMIZATION RESULTS ===")
    print(f"Recommended Price: INR {result['recommended_price']:.2f}")
    print(f"Expected Volume: {result['expected_volume']:,} liters")
    print(f"Expected Profit: INR {result['expected_profit']:,.2f}")
    print(f"Expected Margin: INR {result['expected_margin']:.2f} ({result['expected_margin_pct']:.1f}%)")
    print(f"Price vs Competitors: INR {result['price_vs_competitors']:+.2f}")
    
    # Compare with competitors
    avg_comp = np.mean(comp_prices)
    print(f"\nCompetitor Analysis:")
    print(f"  Competitor 1: INR {today_data['comp1_price']:.2f}")
    print(f"  Competitor 2: INR {today_data['comp2_price']:.2f}")
    print(f"  Competitor 3: INR {today_data['comp3_price']:.2f}")
    print(f"  Average: INR {avg_comp:.2f}")
    
    return result

def validate_model():
    """Validate model performance on historical data"""
    print("=== Model Validation ===")
    
    pipeline = DataPipeline()
    df = pipeline.load_historical_data('../data/oil_retail_history.csv')
    df = pipeline.engineer_features(df)
    df = pipeline.prepare_features(df)
    
    # Load model
    optimizer = PriceOptimizer()
    optimizer.load_model('../models/price_optimizer.pkl')
    
    # Test on last 30 days
    test_df = df.tail(30).copy()
    predictions = []
    
    for idx, row in test_df.iterrows():
        if idx == 0:
            continue
            
        # Create features for this day
        features = pd.DataFrame([row[pipeline.feature_columns]])
        
        # Predict volume for actual price
        predicted_volume = optimizer.predict_volume(features, row['price'])
        actual_volume = row['volume']
        
        predictions.append({
            'date': row['date'],
            'actual_volume': actual_volume,
            'predicted_volume': predicted_volume,
            'error': abs(actual_volume - predicted_volume),
            'error_pct': abs(actual_volume - predicted_volume) / actual_volume * 100
        })
    
    pred_df = pd.DataFrame(predictions)
    print(f"Mean Absolute Error: {pred_df['error'].mean():.0f} liters")
    print(f"Mean Absolute Percentage Error: {pred_df['error_pct'].mean():.1f}%")
    
    return pred_df

if __name__ == "__main__":
    # Train model
    optimizer, pipeline, df = train_model()
    
    # Validate model
    validation_results = validate_model()
    
    # Predict optimal price for today
    result = predict_optimal_price()
    
    # Save results (convert numpy types to native Python types)
    result_serializable = {
        k: float(v) if isinstance(v, (np.float32, np.float64)) else 
           int(v) if isinstance(v, (np.int32, np.int64)) else v 
        for k, v in result.items() if k != 'all_scenarios'
    }
    
    with open('../data/prediction_result.json', 'w') as f:
        json.dump(result_serializable, f, indent=2)
    
    print("\nResults saved to ../data/prediction_result.json")