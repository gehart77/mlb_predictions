"""
Model Training Module

This module handles the training and evaluation of the MLB prediction model.
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

from src.data.mlb_data_collector import MLBDataCollector
from src.features.feature_engineering import create_team_features
from src.utils.constants import MODEL_CONFIG, MODELS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('prediction_model')

def collect_training_data(start_year=2018, end_year=2024):
    """
    Collect training data for the model.
    
    Args:
        start_year (int): Start year for data collection
        end_year (int): End year for data collection
        
    Returns:
        tuple: (features, target) DataFrames
    """
    logger.info(f"Collecting data for {start_year}")
    
    all_features = []
    all_targets = []
    
    data_collector = MLBDataCollector()
    
    # MLB team IDs
    team_ids = [
        'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET',
        'HOU', 'KCR', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'OAK',
        'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN'
    ]
    
    for year in range(start_year, end_year + 1):
        logger.info(f"Collecting data for {year}")
        
        for team in team_ids:
            try:
                # Get team statistics
                batting_stats, pitching_stats = data_collector.get_team_stats(team, year)
                
                # Get team schedule
                schedule = data_collector.get_team_schedule(team, year)
                
                # Create features
                features = create_team_features(batting_stats, pitching_stats, schedule)
                
                # Create target (win/loss)
                target = (schedule['W'] > schedule['L']).astype(int)
                
                all_features.append(features)
                all_targets.append(target)
                
            except Exception as e:
                logger.error(f"Error: {str(e)}", extra={'team': team, 'year': year})
                continue
    
    if not all_features:
        raise ValueError("No data collected for any team/year combination")
    
    features = pd.concat(all_features, ignore_index=True)
    target = pd.concat(all_targets, ignore_index=True)
    
    return features, target

def train_model(features, target):
    """
    Train the MLB prediction model.
    
    Args:
        features (DataFrame): Training features
        target (Series): Target variable
        
    Returns:
        RandomForestClassifier: Trained model
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target,
        test_size=0.2,
        random_state=42
    )
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=MODEL_CONFIG['n_estimators'],
        max_depth=MODEL_CONFIG['max_depth'],
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model accuracy: {accuracy:.2f}")
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
    
    return model

def save_model(model, model_path):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model
        model_path (str): Path to save the model
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

def main():
    """
    Main function to train and save the model.
    """
    logger.info("Starting model training process")
    
    try:
        # Collect training data
        start_year = 2018
        end_year = 2024
        features, target = collect_training_data(start_year, end_year)
        
        # Train model
        model = train_model(features, target)
        
        # Save model
        model_path = os.path.join(MODELS_DIR, 'mlb_prediction_model.pkl')
        save_model(model, model_path)
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

if __name__ == "__main__":
    main() 