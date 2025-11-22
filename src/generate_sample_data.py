import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

def generate_sample_data():
    """Generate realistic fuel retail data"""
    np.random.seed(42)
    
    # Generate 2 years of daily data
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    n_days = len(dates)
    
    # Base parameters
    base_cost = 85.0
    base_price = 94.0
    
    # Generate correlated data
    data = []
    
    for i, date in enumerate(dates):
        # Seasonal and trend effects
        seasonal_factor = 0.1 * np.sin(2 * np.pi * i / 365.25)
        trend_factor = 0.0001 * i
        
        # Cost varies with oil prices (some volatility)
        cost = base_cost + seasonal_factor + trend_factor + np.random.normal(0, 2.0)
        
        # Company price (with some stickiness)
        if i == 0:
            price = base_price + seasonal_factor + trend_factor
        else:
            # Price changes are sticky - only change 30% of the time
            if np.random.random() < 0.3:
                price = cost * (1.25 + np.random.normal(0, 0.1))  # 25% markup with noise
            else:
                price = data[i-1]['price'] + np.random.normal(0, 0.5)  # Small drift
        
        # Competitor prices (correlated but with different strategies)
        comp1_price = price + np.random.normal(-1.0, 1.5)  # Slightly cheaper
        comp2_price = price + np.random.normal(0.5, 2.0)   # Slightly more expensive
        comp3_price = price + np.random.normal(0.0, 2.5)   # Similar pricing
        
        # Volume depends on price competitiveness and seasonality
        price_competitiveness = np.mean([comp1_price, comp2_price, comp3_price]) - price
        volume_base = 10000 + 2000 * seasonal_factor  # Seasonal demand
        volume = volume_base + 5000 * price_competitiveness + np.random.normal(0, 1000)
        volume = max(volume, 1000)  # Minimum volume
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'price': round(price, 3),
            'cost': round(cost, 3),
            'comp1_price': round(comp1_price, 3),
            'comp2_price': round(comp2_price, 3),
            'comp3_price': round(comp3_price, 3),
            'volume': int(volume)
        })
    
    return pd.DataFrame(data)

def generate_today_example():
    """Generate today's input example"""
    return {
        "date": "2024-12-31",
        "price": 94.45,  # Last observed price in INR
        "cost": 85.77,   # Today's cost in INR
        "comp1_price": 95.01,
        "comp2_price": 95.7,
        "comp3_price": 95.21
    }

if __name__ == "__main__":
    # Generate and save historical data
    df = generate_sample_data()
    df.to_csv('../data/oil_retail_history.csv', index=False)
    print(f"Generated {len(df)} days of historical data")
    
    # Generate today's example
    today_data = generate_today_example()
    with open('../data/today_example.json', 'w') as f:
        json.dump(today_data, f, indent=2)
    print("Generated today's example data")