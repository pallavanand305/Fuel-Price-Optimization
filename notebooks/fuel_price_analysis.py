# Fuel Price Optimization Analysis
# This script demonstrates the complete analysis workflow

import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_pipeline import DataPipeline, load_today_data
from price_optimizer import PriceOptimizer
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load and explore the dataset"""
    print("=== FUEL PRICE OPTIMIZATION ANALYSIS ===\n")
    
    # Load data
    pipeline = DataPipeline()
    df = pipeline.load_historical_data('../data/oil_retail_history.csv')
    
    print(f"Dataset Overview:")
    print(f"- Records: {len(df):,}")
    print(f"- Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"- Columns: {list(df.columns)}")
    
    # Basic statistics
    print(f"\nPrice Statistics:")
    print(f"- Company Price: INR {df['price'].mean():.2f} ± INR {df['price'].std():.2f}")
    print(f"- Cost: INR {df['cost'].mean():.2f} ± INR {df['cost'].std():.2f}")
    print(f"- Average Margin: INR {(df['price'] - df['cost']).mean():.2f}")
    print(f"- Volume: {df['volume'].mean():,.0f} ± {df['volume'].std():,.0f} liters")
    
    return df

def analyze_price_relationships(df):
    """Analyze relationships between price, volume, and competitors"""
    print(f"\n=== PRICE RELATIONSHIP ANALYSIS ===")
    
    # Calculate correlations
    corr_matrix = df[['price', 'cost', 'volume', 'comp1_price', 'comp2_price', 'comp3_price']].corr()
    
    print(f"\nKey Correlations:")
    print(f"- Price vs Volume: {corr_matrix.loc['price', 'volume']:.3f}")
    print(f"- Price vs Cost: {corr_matrix.loc['price', 'cost']:.3f}")
    print(f"- Price vs Comp1: {corr_matrix.loc['price', 'comp1_price']:.3f}")
    
    # Price competitiveness analysis
    df['avg_comp_price'] = df[['comp1_price', 'comp2_price', 'comp3_price']].mean(axis=1)
    df['price_advantage'] = df['avg_comp_price'] - df['price']
    df['profit'] = (df['price'] - df['cost']) * df['volume']
    
    print(f"\nCompetitive Analysis:")
    print(f"- Average price advantage: INR {df['price_advantage'].mean():.2f}")
    print(f"- Days with price advantage: {(df['price_advantage'] > 0).sum()} ({(df['price_advantage'] > 0).mean()*100:.1f}%)")
    print(f"- Average daily profit: INR {df['profit'].mean():,.2f}")
    
    return df

def demonstrate_optimization():
    """Demonstrate the price optimization process"""
    print(f"\n=== PRICE OPTIMIZATION DEMONSTRATION ===")
    
    # Load today's example
    today_data = load_today_data('../data/today_example.json')
    print(f"\nToday's Market Conditions:")
    print(f"- Current Price: INR {today_data['price']:.2f}")
    print(f"- Cost: INR {today_data['cost']:.2f}")
    print(f"- Competitor 1: INR {today_data['comp1_price']:.2f}")
    print(f"- Competitor 2: INR {today_data['comp2_price']:.2f}")
    print(f"- Competitor 3: INR {today_data['comp3_price']:.2f}")
    
    # Load model and optimize
    pipeline = DataPipeline()
    historical_df = pipeline.load_historical_data('../data/oil_retail_history.csv')
    historical_df = pipeline.engineer_features(historical_df)
    historical_df = pipeline.prepare_features(historical_df)
    
    today_features = pipeline.process_today_input(today_data, historical_df)
    
    optimizer = PriceOptimizer()
    optimizer.load_model('../models/price_optimizer.pkl')
    
    comp_prices = [today_data['comp1_price'], today_data['comp2_price'], today_data['comp3_price']]
    result = optimizer.optimize_price(
        features=today_features,
        cost=today_data['cost'],
        last_price=today_data['price'],
        comp_prices=comp_prices
    )
    
    print(f"\nOptimization Results:")
    print(f"- Recommended Price: INR {result['recommended_price']:.2f}")
    print(f"- Expected Volume: {result['expected_volume']:,} liters")
    print(f"- Expected Profit: INR {result['expected_profit']:,.2f}")
    print(f"- Margin: {result['expected_margin_pct']:.1f}%")
    print(f"- vs Competitors: INR {result['price_vs_competitors']:+.2f}")
    
    return result

def analyze_scenarios(optimizer, today_features, today_data):
    """Analyze different pricing scenarios"""
    print(f"\n=== SCENARIO ANALYSIS ===")
    
    # Test different price points
    base_price = today_data['price']
    price_scenarios = [base_price - 0.05, base_price, base_price + 0.05, base_price + 0.10]
    
    print(f"\nPrice Scenario Analysis:")
    print(f"{'Price':<12} {'Volume':<8} {'Profit':<14} {'Margin%':<8}")
    print("-" * 45)
    
    for price in price_scenarios:
        volume = optimizer.predict_volume(today_features, price)
        profit = optimizer.calculate_profit(price, today_data['cost'], volume)
        margin_pct = (price - today_data['cost']) / today_data['cost'] * 100
        
        print(f"INR {price:<8.2f} {volume:<8.0f} INR {profit:<10.2f} {margin_pct:<7.1f}%")

def main():
    """Run complete analysis"""
    # Load and explore data
    df = load_and_explore_data()
    
    # Analyze relationships
    df = analyze_price_relationships(df)
    
    # Demonstrate optimization
    result = demonstrate_optimization()
    
    # Load optimizer for scenario analysis
    pipeline = DataPipeline()
    historical_df = pipeline.load_historical_data('../data/oil_retail_history.csv')
    historical_df = pipeline.engineer_features(historical_df)
    historical_df = pipeline.prepare_features(historical_df)
    
    today_data = load_today_data('../data/today_example.json')
    today_features = pipeline.process_today_input(today_data, historical_df)
    
    optimizer = PriceOptimizer()
    optimizer.load_model('../models/price_optimizer.pkl')
    
    # Scenario analysis
    analyze_scenarios(optimizer, today_features, today_data)
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"The system successfully demonstrates:")
    print(f"[+] Data pipeline with feature engineering")
    print(f"[+] ML model training and validation")
    print(f"[+] Price optimization with business rules")
    print(f"[+] Scenario analysis capabilities")
    print(f"[+] Production-ready API service")

if __name__ == "__main__":
    main()