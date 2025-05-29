"""
MLB Data Collector Module

This module handles the collection of MLB data from various sources including:
- Team statistics
- Player statistics
- Game schedules
- Weather data
- Betting odds
"""

import os
from datetime import datetime, timedelta
import pandas as pd
import requests
from pybaseball import team_batting, team_pitching, schedule_and_record
from dotenv import load_dotenv
import time
from typing import Tuple, Optional

load_dotenv()

class MLBDataCollector:
    def __init__(self):
        self.weather_api_key = os.getenv('WEATHER_API_KEY')
        self.odds_api_key = os.getenv('ODDS_API_KEY')
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        self.mlb_stats_api_base = "https://statsapi.mlb.com/api/v1"
        
    def _get_mlb_stats_api_data(self, team_id: str, year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch team statistics from MLB Stats API as a fallback.
        
        Args:
            team_id (str): MLB team ID
            year (int): Year to fetch stats for
            
        Returns:
            tuple: (batting_stats, pitching_stats) DataFrames
        """
        try:
            # Get team ID from team abbreviation
            team_lookup_url = f"{self.mlb_stats_api_base}/teams"
            print(f"Fetching team info from: {team_lookup_url}")
            response = requests.get(team_lookup_url)
            teams_data = response.json()
            
            team_info = next((team for team in teams_data['teams'] 
                            if team['abbreviation'] == team_id), None)
            
            if not team_info:
                raise Exception(f"Team {team_id} not found in MLB Stats API")
                
            team_id_mlb = team_info['id']
            print(f"Found team ID {team_id_mlb} for {team_id}")
            
            # Get batting stats
            batting_url = f"{self.mlb_stats_api_base}/teams/{team_id_mlb}/stats"
            params = {
                'season': year,
                'stats': 'season',
                'group': 'hitting'
            }
            print(f"Fetching batting stats from: {batting_url} with params: {params}")
            batting_response = requests.get(batting_url, params=params)
            batting_data = batting_response.json()
            
            # Get pitching stats
            params['group'] = 'pitching'
            print(f"Fetching pitching stats from: {batting_url} with params: {params}")
            pitching_response = requests.get(batting_url, params=params)
            pitching_data = pitching_response.json()
            
            # Print raw data for debugging
            print("\nBatting data structure:")
            print(batting_data.keys())
            if 'stats' in batting_data and batting_data['stats']:
                print("First level stats keys:", batting_data['stats'][0].keys())
                if 'splits' in batting_data['stats'][0]:
                    print("Splits keys:", batting_data['stats'][0]['splits'][0].keys())
            
            print("\nPitching data structure:")
            print(pitching_data.keys())
            if 'stats' in pitching_data and pitching_data['stats']:
                print("First level stats keys:", pitching_data['stats'][0].keys())
                if 'splits' in pitching_data['stats'][0]:
                    print("Splits keys:", pitching_data['stats'][0]['splits'][0].keys())
            
            # Convert to DataFrames with proper index
            batting_stats = pd.DataFrame([batting_data['stats'][0]['splits'][0]['stat']])
            pitching_stats = pd.DataFrame([pitching_data['stats'][0]['splits'][0]['stat']])
            
            # Add team and year columns
            batting_stats['team'] = team_id
            batting_stats['year'] = year
            pitching_stats['team'] = team_id
            pitching_stats['year'] = year
            
            print("\nCreated DataFrames:")
            print("Batting stats columns:", batting_stats.columns.tolist())
            print("Pitching stats columns:", pitching_stats.columns.tolist())
            
            return batting_stats, pitching_stats
            
        except Exception as e:
            print(f"\nDetailed error in MLB Stats API:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            raise Exception(f"Error fetching data from MLB Stats API: {str(e)}")
        
    def get_team_stats(self, team_id: str, year: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch team batting and pitching statistics with retry logic and fallback.
        
        Args:
            team_id (str): MLB team ID
            year (int, optional): Year to fetch stats for. Defaults to current year.
            
        Returns:
            tuple: (batting_stats, pitching_stats) DataFrames
            
        Raises:
            Exception: If data cannot be fetched from any source
        """
        if year is None:
            year = datetime.now().year
            
        # Ensure team_id is a string
        team_id = str(team_id)
        
        # Try MLB Stats API first (more reliable)
        try:
            print(f"Fetching data from MLB Stats API for team {team_id} for year {year}")
            return self._get_mlb_stats_api_data(team_id, year)
        except Exception as e:
            print(f"Error fetching data from MLB Stats API: {str(e)}")
            print("Trying Fangraphs as fallback...")
            
            # Try Fangraphs as fallback
            for attempt in range(self.max_retries):
                try:
                    print(f"Attempting to fetch data from Fangraphs for team {team_id} (attempt {attempt + 1}/{self.max_retries})")
                    batting_stats = team_batting(team_id, year)
                    pitching_stats = team_pitching(team_id, year)
                    
                    # Verify we got valid data
                    if batting_stats is not None and not batting_stats.empty and \
                       pitching_stats is not None and not pitching_stats.empty:
                        return batting_stats, pitching_stats
                        
                except Exception as e:
                    print(f"Error fetching team data from Fangraphs (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                    if attempt < self.max_retries - 1:
                        print(f"Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
            
            raise Exception(f"Failed to fetch team data from both MLB Stats API and Fangraphs: {str(e)}")
    
    def get_team_schedule(self, team_id: str, year: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch team schedule and record using MLB Stats API.
        
        Args:
            team_id (str): MLB team ID
            year (int, optional): Year to fetch schedule for. Defaults to current year.
            
        Returns:
            DataFrame: Team schedule and record
        """
        if year is None:
            year = datetime.now().year
            
        try:
            print(f"Fetching schedule for team {team_id} for year {year}")
            
            # Get team ID from MLB Stats API
            team_lookup_url = f"{self.mlb_stats_api_base}/teams"
            print(f"Team lookup URL: {team_lookup_url}")
            response = requests.get(team_lookup_url)
            print(f"Team lookup response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error response: {response.text}")
                raise Exception(f"Failed to fetch team data: {response.status_code}")
                
            teams_data = response.json()
            print(f"Found {len(teams_data.get('teams', []))} teams in API response")
            
            team_info = next((team for team in teams_data['teams'] 
                            if team['abbreviation'] == team_id), None)
            
            if not team_info:
                print(f"Team {team_id} not found in API response")
                print("Available teams:", [team['abbreviation'] for team in teams_data['teams']])
                raise ValueError(f"Team {team_id} not found in MLB Stats API")
                
            team_id_mlb = team_info['id']
            print(f"Found team ID {team_id_mlb} for {team_id}")
            
            # Get schedule
            schedule_url = f"{self.mlb_stats_api_base}/schedule"
            params = {
                'sportId': 1,  # MLB
                'teamId': team_id_mlb,
                'season': year,
                'hydrate': 'team,venue,game'
            }
            print(f"Schedule URL: {schedule_url}")
            print(f"Schedule params: {params}")
            
            response = requests.get(schedule_url, params=params)
            print(f"Schedule response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error response: {response.text}")
                raise Exception(f"Failed to fetch schedule: {response.status_code}")
                
            schedule_data = response.json()
            print(f"Found {len(schedule_data.get('dates', []))} dates in schedule")
            
            if not schedule_data.get('dates'):
                print("No schedule dates found in response")
                return pd.DataFrame(columns=['Date', 'Team', 'Opponent', 'Location', 'Time', 'game_id', 'HomeTeam', 'AwayTeam'])
            
            # Process schedule data
            games = []
            for date in schedule_data['dates']:
                print(f"Processing date: {date.get('date')}")
                for game in date['games']:
                    print(f"Processing game: {game.get('gamePk')}")
                    game_date = pd.to_datetime(game['gameDate'])
                    games.append({
                        'Date': game_date,
                        'Time': game_date.time(),
                        'game_id': str(game['gamePk']),
                        'HomeTeam': game['teams']['home']['team']['abbreviation'],
                        'AwayTeam': game['teams']['away']['team']['abbreviation'],
                        'Location': game['venue']['name'],
                        'Team': team_id,
                        'Opponent': game['teams']['away']['team']['abbreviation'] if game['teams']['home']['team']['abbreviation'] == team_id else game['teams']['home']['team']['abbreviation']
                    })
            
            schedule = pd.DataFrame(games)
            print(f"Processed schedule shape: {schedule.shape}")
            if not schedule.empty:
                print(f"Date range: {schedule['Date'].min()} to {schedule['Date'].max()}")
            
            return schedule
            
        except Exception as e:
            import traceback
            print(f"Error fetching schedule for team {team_id}: {str(e)}")
            print("Full error details:")
            print(traceback.format_exc())
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['Date', 'Team', 'Opponent', 'Location', 'Time', 'game_id', 'HomeTeam', 'AwayTeam'])
    
    def get_weather_forecast(self, location, date):
        """
        Fetch weather forecast for a specific location and date.
        
        Args:
            location (str): City name
            date (str): Date in YYYY-MM-DD format
            
        Returns:
            dict: Weather forecast data
        """
        base_url = "http://api.weatherapi.com/v1/forecast.json"
        params = {
            'key': self.weather_api_key,
            'q': location,
            'dt': date,
            'aqi': 'no'
        }
        print(f"Weather API Key: {self.weather_api_key}")
        print(f"Weather location query: {location}")
        print(f"Weather date: {date}")
        try:
            response = requests.get(base_url, params=params)
            print("Weather API response:", response.text)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Weather API error: {str(e)}")
            return {"error": str(e), "response": response.text if 'response' in locals() else None}
    
    def get_betting_odds(self, game_id):
        """
        Fetch betting odds for a specific game.
        
        Args:
            game_id (str): Unique game identifier
            
        Returns:
            dict: Betting odds data
        """
        base_url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
        params = {
            'apiKey': self.odds_api_key,
            'regions': 'us',
            'markets': 'h2h,spreads,totals',
            'gameId': game_id
        }
        
        response = requests.get(base_url, params=params)
        return response.json()
    
    def get_injury_report(self, team_id):
        """
        Fetch current injury report for a team.
        
        Args:
            team_id (str): MLB team ID
            
        Returns:
            DataFrame: Injury report data
        """
        # TODO: Implement injury report fetching
        # This will require scraping from MLB.com or using a paid API
        pass
    
    def get_all_teams_data(self):
        """
        Fetch data for all MLB teams.
        
        Returns:
            dict: Dictionary containing data for all teams
        """
        teams_data = {}
        # TODO: Implement fetching data for all teams
        return teams_data
    
    def update_all_data(self):
        """
        Update all data sources with the latest information.
        
        Returns:
            bool: True if update was successful
        """
        try:
            # TODO: Implement comprehensive data update
            return True
        except Exception as e:
            print(f"Error updating data: {str(e)}")
            return False

if __name__ == "__main__":
    # Example usage
    collector = MLBDataCollector()
    
    # Get data for a specific team (e.g., NYY for New York Yankees)
    batting_stats, pitching_stats = collector.get_team_stats('NYY')
    schedule = collector.get_team_schedule('NYY')
    
    print("Batting Stats:")
    print(batting_stats.head())
    print("\nPitching Stats:")
    print(pitching_stats.head())
    print("\nSchedule:")
    print(schedule.head()) 