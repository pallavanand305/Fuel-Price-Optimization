import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json

class DataPipeline:
    """Data ingestion and transformation pipeline for fuel price optimization"""
    
    def __init__(self):
        self.feature_columns = []
        
    def load_historical_data(self, filepath: str) -> pd.DataFrame:
        """Load and validate historical data"""
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        return self._validate_data(df)
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        # Remove rows with missing critical data
        df = df.dropna(subset=['price', 'cost', 'volume'])
        
        # Remove outliers (prices/costs outside reasonable bounds)
        df = df[(df['price'] > 50.0) & (df['price'] < 150.0)]
        df = df[(df['cost'] > 30.0) & (df['cost'] < 120.0)]
        df = df[df['volume'] > 0]
        
        return df.sort_values('date').reset_index(drop=True)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for ML model"""
        df = df.copy()
        
        # Price features
        df['margin'] = df['price'] - df['cost']
        df['margin_pct'] = (df['price'] - df['cost']) / df['cost']
        
        # Competitor analysis
        df['avg_comp_price'] = df[['comp1_price', 'comp2_price', 'comp3_price']].mean(axis=1)
        df['price_vs_comp'] = df['price'] - df['avg_comp_price']
        df['price_rank'] = df[['price', 'comp1_price', 'comp2_price', 'comp3_price']].rank(axis=1)['price']
        
        # Lag features
        for lag in [1, 3, 7]:
            df[f'price_lag_{lag}'] = df['price'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'margin_lag_{lag}'] = df['margin'].shift(lag)
        
        # Moving averages
        for window in [3, 7, 14]:
            df[f'price_ma_{window}'] = df['price'].rolling(window).mean()
            df[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
            df[f'comp_price_ma_{window}'] = df['avg_comp_price'].rolling(window).mean()
        
        # Volatility features
        df['price_volatility_7d'] = df['price'].rolling(7).std()
        df['volume_volatility_7d'] = df['volume'].rolling(7).std()
        
        # Time features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Target variable (profit)
        df['profit'] = df['margin'] * df['volume']
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare final feature set for modeling"""
        feature_cols = [
            'cost', 'comp1_price', 'comp2_price', 'comp3_price',
            'margin_pct', 'avg_comp_price', 'price_vs_comp', 'price_rank',
            'price_lag_1', 'price_lag_3', 'price_lag_7',
            'volume_lag_1', 'volume_lag_3', 'volume_lag_7',
            'margin_lag_1', 'margin_lag_3', 'margin_lag_7',
            'price_ma_3', 'price_ma_7', 'price_ma_14',
            'volume_ma_3', 'volume_ma_7', 'volume_ma_14',
            'comp_price_ma_3', 'comp_price_ma_7', 'comp_price_ma_14',
            'price_volatility_7d', 'volume_volatility_7d',
            'day_of_week', 'month', 'is_weekend'
        ]
        
        # Store feature columns for later use
        self.feature_columns = [col for col in feature_cols if col in df.columns]
        
        # Fill missing values with forward fill then backward fill
        df[self.feature_columns] = df[self.feature_columns].fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def process_today_input(self, today_data: Dict, historical_df: pd.DataFrame) -> pd.DataFrame:
        """Process today's input data for prediction"""
        # Create a row for today
        today_row = pd.DataFrame([today_data])
        today_row['date'] = pd.to_datetime(today_row['date'])
        
        # Combine with historical data to compute features
        combined_df = pd.concat([historical_df, today_row], ignore_index=True)
        combined_df = self.engineer_features(combined_df)
        combined_df = self.prepare_features(combined_df)
        
        # Return only today's row with features
        return combined_df.iloc[-1:][self.feature_columns]

def load_today_data(filepath: str) -> Dict:
    """Load today's input data"""
    with open(filepath, 'r') as f:
        return json.load(f)