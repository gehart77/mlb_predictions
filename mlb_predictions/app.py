"""
MLB Game Predictions Application

This is the main application file that provides a Streamlit interface for the
MLB prediction system. It integrates all components and provides a user-friendly
way to view predictions and analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.mlb_data_collector import MLBDataCollector
from features.feature_engineering import FeatureEngineer
from models.prediction_model import MLBPredictionModel

# Initialize components
collector = MLBDataCollector()
engineer = FeatureEngineer()
model = MLBPredictionModel()

def main():
    st.set_page_config(
        page_title="MLB Game Predictions",
        page_icon="⚾",
        layout="wide"
    )
    
    st.title("MLB Game Predictions")
    st.write("Predict scores for upcoming MLB games using advanced analytics and machine learning.")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Today's Predictions", "Team Analysis", "Model Performance", "About"]
    )
    
    if page == "Today's Predictions":
        show_todays_predictions()
    elif page == "Team Analysis":
        show_team_analysis()
    elif page == "Model Performance":
        show_model_performance()
    else:
        show_about()

def show_todays_predictions():
    st.header("Today's Game Predictions")
    
    # Get today's date
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Fetch today's games
    try:
        # TODO: Implement game fetching
        games = []  # Placeholder for actual game data
        
        if not games:
            st.info("No games scheduled for today.")
            return
        
        # Display predictions for each game
        for game in games:
            with st.expander(f"{game['home_team']} vs {game['away_team']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Predicted Score")
                    st.write(f"Home: {game['prediction']['home_team_runs']}")
                    st.write(f"Away: {game['prediction']['away_team_runs']}")
                
                with col2:
                    st.subheader("Win Probability")
                    fig = px.pie(
                        values=[
                            game['prediction']['home_win_probability'],
                            game['prediction']['away_win_probability']
                        ],
                        names=[game['home_team'], game['away_team']],
                        title="Win Probability"
                    )
                    st.plotly_chart(fig)
                
                # Show additional game information
                st.subheader("Game Details")
                st.write(f"Weather: {game['weather']}")
                st.write(f"Venue: {game['venue']}")
                st.write(f"Start Time: {game['start_time']}")
                
    except Exception as e:
        st.error(f"Error fetching game data: {str(e)}")

def show_team_analysis():
    st.header("Team Analysis")
    
    # Team selection
    team = st.selectbox(
        "Select Team",
        ["NYY", "BOS", "LAD", "SFG"]  # TODO: Add all MLB teams
    )
    
    try:
        # Fetch team data
        batting_stats, pitching_stats = collector.get_team_stats(team)
        schedule = collector.get_team_schedule(team)
        
        # Display team statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Batting Statistics")
            st.dataframe(batting_stats)
        
        with col2:
            st.subheader("Pitching Statistics")
            st.dataframe(pitching_stats)
        
        # Show team performance trends
        st.subheader("Team Performance")
        fig = px.line(
            schedule,
            x='Date',
            y=['W', 'L'],
            title=f"{team} Season Record"
        )
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error fetching team data: {str(e)}")

def show_model_performance():
    st.header("Model Performance")
    
    # TODO: Implement model performance visualization
    st.info("Model performance metrics and visualizations will be displayed here.")

def show_about():
    st.header("About")
    
    st.write("""
    This application uses machine learning to predict MLB game outcomes. It considers:
    
    - Team statistics and performance
    - Player availability and injuries
    - Weather conditions
    - Historical matchups
    - Betting odds
    
    The predictions are generated using a Random Forest model trained on historical MLB data.
    """)
    
    st.subheader("Data Sources")
    st.write("""
    - MLB Stats API
    - Weather API
    - Odds API
    - Baseball Reference
    """)
    
    st.subheader("Contact")
    st.write("For questions or feedback, please contact the development team.")

if __name__ == "__main__":
    main() 