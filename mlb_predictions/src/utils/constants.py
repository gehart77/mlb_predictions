"""
Constants Module

This module contains constants used throughout the application, including:
- MLB team information
- API endpoints
- Configuration settings
"""

# MLB Teams
MLB_TEAMS = {
    'ARI': {'name': 'Arizona Diamondbacks', 'city': 'Phoenix', 'state': 'AZ'},
    'ATL': {'name': 'Atlanta Braves', 'city': 'Atlanta', 'state': 'GA'},
    'BAL': {'name': 'Baltimore Orioles', 'city': 'Baltimore', 'state': 'MD'},
    'BOS': {'name': 'Boston Red Sox', 'city': 'Boston', 'state': 'MA'},
    'CHC': {'name': 'Chicago Cubs', 'city': 'Chicago', 'state': 'IL'},
    'CHW': {'name': 'Chicago White Sox', 'city': 'Chicago', 'state': 'IL'},
    'CIN': {'name': 'Cincinnati Reds', 'city': 'Cincinnati', 'state': 'OH'},
    'CLE': {'name': 'Cleveland Guardians', 'city': 'Cleveland', 'state': 'OH'},
    'COL': {'name': 'Colorado Rockies', 'city': 'Denver', 'state': 'CO'},
    'DET': {'name': 'Detroit Tigers', 'city': 'Detroit', 'state': 'MI'},
    'HOU': {'name': 'Houston Astros', 'city': 'Houston', 'state': 'TX'},
    'KCR': {'name': 'Kansas City Royals', 'city': 'Kansas City', 'state': 'MO'},
    'LAA': {'name': 'Los Angeles Angels', 'city': 'Anaheim', 'state': 'CA'},
    'LAD': {'name': 'Los Angeles Dodgers', 'city': 'Los Angeles', 'state': 'CA'},
    'MIA': {'name': 'Miami Marlins', 'city': 'Miami', 'state': 'FL'},
    'MIL': {'name': 'Milwaukee Brewers', 'city': 'Milwaukee', 'state': 'WI'},
    'MIN': {'name': 'Minnesota Twins', 'city': 'Minneapolis', 'state': 'MN'},
    'NYM': {'name': 'New York Mets', 'city': 'New York', 'state': 'NY'},
    'NYY': {'name': 'New York Yankees', 'city': 'New York', 'state': 'NY'},
    'OAK': {'name': 'Oakland Athletics', 'city': 'Oakland', 'state': 'CA'},
    'PHI': {'name': 'Philadelphia Phillies', 'city': 'Philadelphia', 'state': 'PA'},
    'PIT': {'name': 'Pittsburgh Pirates', 'city': 'Pittsburgh', 'state': 'PA'},
    'SDP': {'name': 'San Diego Padres', 'city': 'San Diego', 'state': 'CA'},
    'SEA': {'name': 'Seattle Mariners', 'city': 'Seattle', 'state': 'WA'},
    'SFG': {'name': 'San Francisco Giants', 'city': 'San Francisco', 'state': 'CA'},
    'STL': {'name': 'St. Louis Cardinals', 'city': 'St. Louis', 'state': 'MO'},
    'TBR': {'name': 'Tampa Bay Rays', 'city': 'St. Petersburg', 'state': 'FL'},
    'TEX': {'name': 'Texas Rangers', 'city': 'Arlington', 'state': 'TX'},
    'TOR': {'name': 'Toronto Blue Jays', 'city': 'Toronto', 'state': 'ON'},
    'WSN': {'name': 'Washington Nationals', 'city': 'Washington', 'state': 'DC'}
}

# API Endpoints
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
WEATHER_API_BASE = "http://api.weatherapi.com/v1"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Model Configuration
MODEL_CONFIG = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5
}

# Feature Configuration
FEATURE_CONFIG = {
    'batting_features': [
        'AVG', 'OBP', 'SLG', 'OPS', 'R', 'G',
        'H', '2B', '3B', 'HR', 'RBI', 'BB', 'SO'
    ],
    'pitching_features': [
        'ERA', 'WHIP', 'SO', 'BB', 'IP',
        'H', 'ER', 'HR', 'SV', 'HLD'
    ],
    'schedule_features': [
        'W', 'L', 'W-Home', 'L-Home', 'W-Road', 'L-Road'
    ],
    'weather_features': [
        'temperature', 'humidity', 'wind_speed',
        'wind_direction', 'precipitation'
    ]
}

# File Paths
DATA_DIR = "data"
MODELS_DIR = "models"
LOGS_DIR = "logs"

# Create directories if they don't exist
import os
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True) 