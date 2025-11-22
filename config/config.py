"""Configuration settings for fuel price optimization system"""

# Model Configuration
MODEL_CONFIG = {
    'model_type': 'xgboost',  # 'xgboost' or 'random_forest'
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42
}

# Business Rules
BUSINESS_RULES = {
    'max_daily_change': 0.05,  # Maximum price change per day (dollars)
    'min_margin_pct': 0.15,    # Minimum profit margin (15%)
    'max_margin_pct': 0.40,    # Maximum profit margin (40%)
    'price_bounds': {
        'min_price': 0.50,     # Minimum allowed price
        'max_price': 3.00      # Maximum allowed price
    }
}

# Feature Engineering
FEATURE_CONFIG = {
    'lag_periods': [1, 3, 7],           # Lag features to create
    'moving_avg_windows': [3, 7, 14],   # Moving average windows
    'volatility_window': 7               # Volatility calculation window
}

# Data Validation
DATA_VALIDATION = {
    'required_columns': ['date', 'price', 'cost', 'volume', 'comp1_price', 'comp2_price', 'comp3_price'],
    'price_range': (0.5, 3.0),
    'cost_range': (0.3, 2.5),
    'volume_min': 0
}

# API Configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'title': 'Fuel Price Optimization API',
    'version': '1.0.0'
}

# File Paths
PATHS = {
    'data_dir': '../data/',
    'models_dir': '../models/',
    'historical_data': '../data/oil_retail_history.csv',
    'today_example': '../data/today_example.json',
    'model_file': '../models/price_optimizer.pkl'
}