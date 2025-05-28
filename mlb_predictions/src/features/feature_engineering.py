"""
Feature Engineering Module

This module handles the processing of raw MLB data into features suitable for
machine learning models. It includes functions for:
- Creating team performance features
- Processing weather data
- Incorporating betting odds
- Handling injury impacts
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = []
        
    def create_team_features(self, batting_stats, pitching_stats, schedule):
        """
        Create features from team statistics and schedule.
        
        Args:
            batting_stats (DataFrame): Team batting statistics
            pitching_stats (DataFrame): Team pitching statistics
            schedule (DataFrame): Team schedule and record
            
        Returns:
            DataFrame: Processed features
        """
        features = pd.DataFrame()
        
        # Batting features
        features['team_avg'] = batting_stats['AVG']
        features['team_obp'] = batting_stats['OBP']
        features['team_slg'] = batting_stats['SLG']
        features['team_ops'] = batting_stats['OPS']
        features['team_runs_per_game'] = batting_stats['R'] / batting_stats['G']
        
        # Pitching features
        features['team_era'] = pitching_stats['ERA']
        features['team_whip'] = pitching_stats['WHIP']
        features['team_k_per_9'] = pitching_stats['SO'] * 9 / pitching_stats['IP']
        features['team_bb_per_9'] = pitching_stats['BB'] * 9 / pitching_stats['IP']
        
        # Schedule features
        features['win_pct'] = schedule['W'] / (schedule['W'] + schedule['L'])
        features['last_10_games'] = self._calculate_last_n_games(schedule, 10)
        features['home_win_pct'] = schedule['W-Home'] / (schedule['W-Home'] + schedule['L-Home'])
        features['away_win_pct'] = schedule['W-Road'] / (schedule['W-Road'] + schedule['L-Road'])
        
        return features
    
    def process_weather_data(self, weather_data):
        """
        Process weather data into features.
        
        Args:
            weather_data (dict): Raw weather data
            
        Returns:
            DataFrame: Weather features
        """
        features = pd.DataFrame()
        
        # Extract relevant weather features
        features['temperature'] = weather_data['current']['temp_f']
        features['humidity'] = weather_data['current']['humidity']
        features['wind_speed'] = weather_data['current']['wind_mph']
        features['wind_direction'] = weather_data['current']['wind_degree']
        features['precipitation'] = weather_data['current']['precip_in']
        
        # Create derived features
        features['wind_factor'] = self._calculate_wind_factor(
            features['wind_speed'],
            features['wind_direction']
        )
        
        return features
    
    def incorporate_betting_odds(self, odds_data):
        """
        Process betting odds into features.
        
        Args:
            odds_data (dict): Raw betting odds data
            
        Returns:
            DataFrame: Odds features
        """
        features = pd.DataFrame()
        
        # Extract odds features
        features['money_line'] = odds_data['h2h'][0]['price']
        features['spread'] = odds_data['spreads'][0]['price']
        features['total'] = odds_data['totals'][0]['price']
        
        # Calculate implied probabilities
        features['implied_probability'] = self._calculate_implied_probability(
            features['money_line']
        )
        
        return features
    
    def process_injury_data(self, injury_data):
        """
        Process injury data into features.
        
        Args:
            injury_data (DataFrame): Raw injury data
            
        Returns:
            DataFrame: Injury impact features
        """
        features = pd.DataFrame()
        
        # Calculate injury impact scores
        features['key_player_injuries'] = self._calculate_key_player_injuries(injury_data)
        features['pitching_injury_impact'] = self._calculate_pitching_injury_impact(injury_data)
        features['batting_injury_impact'] = self._calculate_batting_injury_impact(injury_data)
        
        return features
    
    def _calculate_last_n_games(self, schedule, n):
        """Calculate win percentage for last n games."""
        recent_games = schedule.tail(n)
        return recent_games['W'].sum() / n
    
    def _calculate_wind_factor(self, wind_speed, wind_direction):
        """Calculate wind impact factor based on speed and direction."""
        # TODO: Implement wind factor calculation
        return 1.0
    
    def _calculate_implied_probability(self, money_line):
        """Calculate implied probability from money line odds."""
        if money_line > 0:
            return 100 / (money_line + 100)
        else:
            return abs(money_line) / (abs(money_line) + 100)
    
    def _calculate_key_player_injuries(self, injury_data):
        """Calculate impact of key player injuries."""
        # TODO: Implement key player injury impact calculation
        return 0.0
    
    def _calculate_pitching_injury_impact(self, injury_data):
        """Calculate impact of pitching injuries."""
        # TODO: Implement pitching injury impact calculation
        return 0.0
    
    def _calculate_batting_injury_impact(self, injury_data):
        """Calculate impact of batting injuries."""
        # TODO: Implement batting injury impact calculation
        return 0.0

if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()
    
    # Create sample data
    batting_stats = pd.DataFrame({
        'AVG': [0.250],
        'OBP': [0.320],
        'SLG': [0.400],
        'OPS': [0.720],
        'R': [700],
        'G': [162]
    })
    
    pitching_stats = pd.DataFrame({
        'ERA': [4.00],
        'WHIP': [1.30],
        'SO': [1200],
        'BB': [400],
        'IP': [1400]
    })
    
    schedule = pd.DataFrame({
        'W': [81],
        'L': [81],
        'W-Home': [45],
        'L-Home': [36],
        'W-Road': [36],
        'L-Road': [45]
    })
    
    # Process features
    team_features = engineer.create_team_features(batting_stats, pitching_stats, schedule)
    print("Team Features:")
    print(team_features) 