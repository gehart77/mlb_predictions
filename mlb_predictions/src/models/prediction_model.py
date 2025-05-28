"""
Prediction Model Module

This module implements the machine learning model for predicting MLB game outcomes.
It includes:
- Model training
- Feature importance analysis
- Prediction generation
- Model evaluation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from datetime import datetime

class MLBPredictionModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.feature_importance = None
        
    def prepare_training_data(self, features, target):
        """
        Prepare data for model training.
        
        Args:
            features (DataFrame): Input features
            target (Series): Target variable (runs scored)
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, features, target):
        """
        Train the prediction model.
        
        Args:
            features (DataFrame): Input features
            target (Series): Target variable (runs scored)
            
        Returns:
            dict: Training metrics
        """
        X_train, X_test, y_train, y_test = self.prepare_training_data(features, target)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def predict_game(self, home_features, away_features):
        """
        Predict the score for a game.
        
        Args:
            home_features (DataFrame): Home team features
            away_features (DataFrame): Away team features
            
        Returns:
            dict: Prediction results including scores and probabilities
        """
        # Combine features
        game_features = pd.concat([home_features, away_features], axis=1)
        
        # Make predictions
        home_runs = self.model.predict(game_features)
        away_runs = self.model.predict(game_features)
        
        # Calculate win probability
        home_win_prob = self._calculate_win_probability(home_runs, away_runs)
        
        return {
            'home_team_runs': round(home_runs[0], 1),
            'away_team_runs': round(away_runs[0], 1),
            'home_win_probability': home_win_prob,
            'away_win_probability': 1 - home_win_prob,
            'prediction_confidence': self._calculate_confidence(home_runs, away_runs)
        }
    
    def evaluate_model(self, features, target):
        """
        Evaluate model performance using cross-validation.
        
        Args:
            features (DataFrame): Input features
            target (Series): Target variable
            
        Returns:
            dict: Evaluation metrics
        """
        cv_scores = cross_val_score(
            self.model,
            features,
            target,
            cv=5,
            scoring='neg_mean_squared_error'
        )
        
        return {
            'cv_rmse': np.sqrt(-cv_scores.mean()),
            'cv_std': np.sqrt(-cv_scores.std())
        }
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = joblib.load(filepath)
    
    def _calculate_win_probability(self, home_runs, away_runs):
        """
        Calculate win probability based on predicted runs.
        
        Args:
            home_runs (float): Predicted home team runs
            away_runs (float): Predicted away team runs
            
        Returns:
            float: Home team win probability
        """
        # Simple probability calculation based on run differential
        run_diff = home_runs - away_runs
        return 1 / (1 + np.exp(-run_diff))
    
    def _calculate_confidence(self, home_runs, away_runs):
        """
        Calculate prediction confidence based on run differential.
        
        Args:
            home_runs (float): Predicted home team runs
            away_runs (float): Predicted away team runs
            
        Returns:
            float: Confidence score (0-1)
        """
        run_diff = abs(home_runs - away_runs)
        return min(run_diff / 5, 1.0)  # Normalize to 0-1 range

if __name__ == "__main__":
    # Example usage
    model = MLBPredictionModel()
    
    # Create sample data
    features = pd.DataFrame({
        'team_avg': [0.250, 0.260],
        'team_era': [4.00, 3.90],
        'win_pct': [0.500, 0.550]
    })
    
    target = pd.Series([4.5, 5.0])
    
    # Train model
    metrics = model.train(features, target)
    print("Training Metrics:")
    print(metrics)
    
    # Make prediction
    home_features = pd.DataFrame({
        'team_avg': [0.250],
        'team_era': [4.00],
        'win_pct': [0.500]
    })
    
    away_features = pd.DataFrame({
        'team_avg': [0.260],
        'team_era': [3.90],
        'win_pct': [0.550]
    })
    
    prediction = model.predict_game(home_features, away_features)
    print("\nGame Prediction:")
    print(prediction) 