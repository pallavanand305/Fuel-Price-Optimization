import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from typing import Dict, Tuple, List
import joblib
import warnings
warnings.filterwarnings('ignore')

class PriceOptimizer:
    """ML-based fuel price optimization system"""
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.volume_model = None
        self.feature_columns = None
        self.business_rules = {
            'max_daily_change': 2.0,   # Max ₹2 change per day
            'min_margin_pct': 0.15,    # Minimum 15% margin
            'max_margin_pct': 0.40,    # Maximum 40% margin
        }
        
    def _create_model(self):
        """Create the appropriate model"""
        if self.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    
    def train_volume_model(self, df: pd.DataFrame, feature_columns: List[str]):
        """Train model to predict volume based on price and features"""
        self.feature_columns = feature_columns
        
        # Prepare training data
        X = df[feature_columns].copy()
        y = df['volume'].copy()
        
        # Remove rows with missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Simple validation for small dataset
        if len(X) < 10:
            print("Small dataset - using simple train/test split")
            scores = [0.5]  # Placeholder score
        else:
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                model = self._create_model()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                scores.append(score)
        
        print(f"Validation scores: {scores}")
        print(f"Mean score: {np.mean(scores):.3f}")
        
        # Train final model on all data
        self.volume_model = self._create_model()
        self.volume_model.fit(X, y)
        
        return self.volume_model
    
    def predict_volume(self, features: pd.DataFrame, price: float) -> float:
        """Predict volume for a given price"""
        # Update price in features
        features_copy = features.copy()
        if 'price' in features_copy.columns:
            features_copy['price'] = price
        
        # Recalculate price-dependent features
        if 'margin' in features_copy.columns and 'cost' in features_copy.columns:
            features_copy['margin'] = price - features_copy['cost']
            features_copy['margin_pct'] = (price - features_copy['cost']) / features_copy['cost']
        
        if 'avg_comp_price' in features_copy.columns:
            features_copy['price_vs_comp'] = price - features_copy['avg_comp_price']
        
        # Predict volume
        volume = self.volume_model.predict(features_copy[self.feature_columns])[0]
        return max(volume, 0)  # Ensure non-negative volume
    
    def calculate_profit(self, price: float, cost: float, volume: float) -> float:
        """Calculate profit for given price, cost, and volume"""
        return (price - cost) * volume
    
    def apply_business_rules(self, price: float, cost: float, last_price: float) -> float:
        """Apply business constraints to price"""
        # Maximum daily change constraint
        max_change = self.business_rules['max_daily_change']
        price = max(price, last_price - max_change)
        price = min(price, last_price + max_change)
        
        # Margin constraints
        min_price = cost * (1 + self.business_rules['min_margin_pct'])
        max_price = cost * (1 + self.business_rules['max_margin_pct'])
        
        price = max(price, min_price)
        price = min(price, max_price)
        
        return price
    
    def optimize_price(self, features: pd.DataFrame, cost: float, last_price: float, 
                      comp_prices: List[float]) -> Dict:
        """Find optimal price that maximizes profit"""
        
        # Define price search range
        avg_comp_price = np.mean(comp_prices)
        min_search_price = max(cost * 1.1, avg_comp_price - 5.0)
        max_search_price = min(cost * 1.5, avg_comp_price + 5.0)
        
        # Grid search for optimal price
        price_range = np.linspace(min_search_price, max_search_price, 50)
        best_profit = -np.inf
        best_price = last_price
        best_volume = 0
        
        results = []
        
        for price in price_range:
            # Apply business rules
            constrained_price = self.apply_business_rules(price, cost, last_price)
            
            # Predict volume
            volume = self.predict_volume(features, constrained_price)
            
            # Calculate profit
            profit = self.calculate_profit(constrained_price, cost, volume)
            
            results.append({
                'price': constrained_price,
                'volume': volume,
                'profit': profit,
                'margin': constrained_price - cost,
                'margin_pct': (constrained_price - cost) / cost
            })
            
            if profit > best_profit:
                best_profit = profit
                best_price = constrained_price
                best_volume = volume
        
        return {
            'recommended_price': round(best_price, 3),
            'expected_volume': int(best_volume),
            'expected_profit': round(best_profit, 2),
            'expected_margin': round(best_price - cost, 3),
            'expected_margin_pct': round((best_price - cost) / cost * 100, 1),
            'price_vs_competitors': round(best_price - avg_comp_price, 3),
            'all_scenarios': results
        }
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'volume_model': self.volume_model,
            'feature_columns': self.feature_columns,
            'business_rules': self.business_rules,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.volume_model = model_data['volume_model']
        self.feature_columns = model_data['feature_columns']
        self.business_rules = model_data['business_rules']
        self.model_type = model_data['model_type']