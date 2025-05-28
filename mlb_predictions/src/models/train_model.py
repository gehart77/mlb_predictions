"""
Model Training Script

This script trains the MLB prediction model using historical data.
It includes:
- Data collection
- Feature engineering
- Model training
- Model evaluation
- Model saving
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.mlb_data_collector import MLBDataCollector
from src.features.feature_engineering import FeatureEngineer
from src.models.prediction_model import MLBPredictionModel
from src.utils.logger import model_logger, log_error, log_performance
from src.utils.constants import MLB_TEAMS, MODEL_CONFIG, MODELS_DIR

def collect_training_data(start_year, end_year):
    """
    Collect historical data for model training.
    
    Args:
        start_year (int): Start year for data collection
        end_year (int): End year for data collection
        
    Returns:
        tuple: (features, target) DataFrames
    """
    collector = MLBDataCollector()
    engineer = FeatureEngineer()
    
    all_features = []
    all_targets = []
    
    for year in range(start_year, end_year + 1):
        model_logger.info(f"Collecting data for {year}")
        
        for team_id in MLB_TEAMS.keys():
            try:
                # Collect team data
                batting_stats, pitching_stats = collector.get_team_stats(team_id, year)
                schedule = collector.get_team_schedule(team_id, year)
                
                # Process features
                team_features = engineer.create_team_features(
                    batting_stats,
                    pitching_stats,
                    schedule
                )
                
                # Add to collection
                all_features.append(team_features)
                all_targets.append(schedule['R'])  # Use runs scored as target
                
            except Exception as e:
                log_error(
                    model_logger,
                    e,
                    {"team": team_id, "year": year}
                )
    
    # Combine all data
    features = pd.concat(all_features, ignore_index=True)
    target = pd.concat(all_targets, ignore_index=True)
    
    return features, target

def train_and_evaluate_model(features, target):
    """
    Train and evaluate the prediction model.
    
    Args:
        features (DataFrame): Input features
        target (Series): Target variable
        
    Returns:
        tuple: (model, metrics) Trained model and evaluation metrics
    """
    model = MLBPredictionModel()
    
    # Train model
    start_time = datetime.now()
    metrics = model.train(features, target)
    duration = (datetime.now() - start_time).total_seconds()
    
    log_performance(
        model_logger,
        "model_training",
        duration,
        {"n_samples": len(features)}
    )
    
    # Evaluate model
    eval_metrics = model.evaluate_model(features, target)
    metrics.update(eval_metrics)
    
    return model, metrics

def save_model_and_metrics(model, metrics, year):
    """
    Save trained model and metrics.
    
    Args:
        model (MLBPredictionModel): Trained model
        metrics (dict): Model metrics
        year (int): Year of training data
    """
    # Save model
    model_path = os.path.join(MODELS_DIR, f"mlb_model_{year}.joblib")
    model.save_model(model_path)
    
    # Save metrics
    metrics_path = os.path.join(MODELS_DIR, f"model_metrics_{year}.json")
    pd.Series(metrics).to_json(metrics_path)
    
    model_logger.info(f"Model and metrics saved for {year}")

def main():
    """Main training script."""
    try:
        # Set training parameters
        start_year = 2018
        end_year = datetime.now().year - 1
        
        model_logger.info("Starting model training process")
        
        # Collect data
        features, target = collect_training_data(start_year, end_year)
        model_logger.info(f"Collected {len(features)} training samples")
        
        # Train and evaluate model
        model, metrics = train_and_evaluate_model(features, target)
        model_logger.info("Model training completed")
        model_logger.info(f"Model metrics: {metrics}")
        
        # Save model and metrics
        save_model_and_metrics(model, metrics, end_year)
        
        model_logger.info("Training process completed successfully")
        
    except Exception as e:
        log_error(model_logger, e)
        raise

if __name__ == "__main__":
    main() 