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
import traceback
import numpy as np
from pybaseball import schedule_and_record

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.mlb_data_collector import MLBDataCollector
from features.feature_engineering import FeatureEngineer
from models.prediction_model import MLBPredictionModel

# Initialize components
collector = MLBDataCollector()
engineer = FeatureEngineer()
model = MLBPredictionModel()

STADIUM_TO_CITY = {
    "Yankee Stadium": "New York",
    "Fenway Park": "Boston",
    "Dodger Stadium": "Los Angeles",
    "Oracle Park": "San Francisco",
    "Minute Maid Park": "Houston",
    "Truist Park": "Atlanta",
    "Wrigley Field": "Chicago",
    "Busch Stadium": "St. Louis",
    "Rogers Centre": "Toronto",
    "Tropicana Field": "St. Petersburg",
    "Oriole Park at Camden Yards": "Baltimore",
    "Progressive Field": "Cleveland",
    "Target Field": "Minneapolis",
    "Comerica Park": "Detroit",
    "Kauffman Stadium": "Kansas City",
    "Guaranteed Rate Field": "Chicago",
    "Oakland Coliseum": "Oakland",
    "T-Mobile Park": "Seattle",
    "Globe Life Field": "Arlington",
    "Angel Stadium": "Anaheim",
    "Citi Field": "New York",
    "Citizens Bank Park": "Philadelphia",
    "Nationals Park": "Washington",
    "loanDepot park": "Miami",
    "PNC Park": "Pittsburgh",
    "Great American Ball Park": "Cincinnati",
    "American Family Field": "Milwaukee",
    "Coors Field": "Denver",
    "Chase Field": "Phoenix",
    "Petco Park": "San Diego"
}

@st.cache_data(ttl=3600)
def get_team_schedule_cached(team):
    return collector.get_team_schedule(team)

@st.cache_data(ttl=3600)
def get_team_stats_cached(team):
    return collector.get_team_stats(team)

def main():
    st.set_page_config(
        page_title="MLB Game Predictions",
        page_icon="⚾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
        <style>
        /* Main content area */
        .main {
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        
        /* Headers */
        h1 {
            font-size: 2.5rem !important;
            margin-bottom: 1.5rem !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }
        
        h2 {
            font-size: 1.8rem !important;
            margin: 1.5rem 0 1rem 0 !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        
        h3 {
            font-size: 1.4rem !important;
            margin: 1rem 0 !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        
        /* Cards/Expanders */
        .streamlit-expanderHeader {
            font-size: 1.2rem !important;
            padding: 1rem !important;
            background-color: #2D2D2D !important;
            border-radius: 8px !important;
            margin: 0.5rem 0 !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        
        /* Text */
        p {
            font-size: 1.1rem !important;
            line-height: 1.6 !important;
            color: #FFFFFF !important;
            font-weight: 400 !important;
        }
        
        /* Buttons and controls */
        .stButton>button {
            min-height: 44px !important;
            padding: 0.5rem 1rem !important;
            border-radius: 8px !important;
            font-size: 1.1rem !important;
            font-weight: 500 !important;
            color: #FFFFFF !important;
            background-color: #0066CC !important;
        }
        
        /* DataFrames */
        .dataframe {
            font-size: 1rem !important;
            border-radius: 8px !important;
            overflow: hidden !important;
            color: #FFFFFF !important;
            background-color: #2D2D2D !important;
        }
        
        /* Sidebar */
        .css-1d391kg {
            padding: 2rem 1rem !important;
            background-color: #2D2D2D !important;
            color: #FFFFFF !important;
        }
        
        /* Charts */
        .js-plotly-plot {
            border-radius: 8px !important;
            overflow: hidden !important;
            background-color: #2D2D2D !important;
        }
        
        /* Metrics */
        .stMetric {
            background-color: #2D2D2D !important;
            padding: 1rem !important;
            border-radius: 8px !important;
            color: #FFFFFF !important;
        }
        
        .stMetric label {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        
        .stMetric div {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background-color: #2D2D2D !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #2D2D2D;
            border-radius: 8px 8px 0 0;
            gap: 1rem;
            padding-top: 10px;
            padding-bottom: 10px;
            color: #FFFFFF !important;
            font-weight: 500 !important;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #1E1E1E !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        
        /* Select boxes */
        .stSelectbox label {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        
        /* Radio buttons */
        .stRadio label {
            color: #FFFFFF !important;
            font-weight: 500 !important;
        }
        
        /* Info messages */
        .stInfo {
            background-color: #1E3A5F !important;
            color: #FFFFFF !important;
            font-weight: 500 !important;
        }
        
        /* Warning messages */
        .stWarning {
            background-color: #5C3D00 !important;
            color: #FFFFFF !important;
            font-weight: 500 !important;
        }
        
        /* Error messages */
        .stError {
            background-color: #5C0000 !important;
            color: #FFFFFF !important;
            font-weight: 500 !important;
        }
        
        /* Table headers and cells */
        th {
            background-color: #2D2D2D !important;
            color: #FFFFFF !important;
        }
        
        td {
            background-color: #1E1E1E !important;
            color: #FFFFFF !important;
        }
        
        /* Plotly chart text */
        .js-plotly-plot .plotly .main-svg text {
            fill: #FFFFFF !important;
        }
        
        /* Plotly chart background */
        .js-plotly-plot .plotly .main-svg {
            background-color: #2D2D2D !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("MLB Game Predictions")
    st.write("Predict scores for upcoming MLB games using advanced analytics and machine learning.")
    
    # Sidebar with improved styling
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Go to",
            ["Today's Predictions", "Team Analysis", "Model Performance", "About"],
            label_visibility="collapsed"
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
    today = datetime.now().date()
    
    try:
        teams = ["NYY", "BOS", "LAD", "SF", "HOU", "ATL", "CHC", "STL", "TOR", "TB", 
                "BAL", "CLE", "MIN", "DET", "KC", "CWS", "OAK", "SEA", "TEX", "LAA",
                "NYM", "PHI", "WSH", "MIA", "PIT", "CIN", "MIL", "COL", "ARI", "SD"]
        games = []
        for team in teams:
            try:
                schedule = get_team_schedule_cached(team)
                if 'Date' in schedule.columns:
                    today_games = schedule[schedule['Date'].dt.date == today]
                else:
                    today_games = pd.DataFrame()
                for _, game in today_games.iterrows():
                    if not any(g['game_id'] == game.get('game_id', '') for g in games):
                        home_stats = get_team_stats_cached(game['HomeTeam'])
                        away_stats = get_team_stats_cached(game['AwayTeam'])
                        location = game.get('Location', 'Unknown')
                        city = STADIUM_TO_CITY.get(location, location)
                        weather = collector.get_weather_forecast(city, today.strftime("%Y-%m-%d"))
                        prediction = {
                            'home_team_runs': 4,  # Placeholder - replace with actual prediction
                            'away_team_runs': 3,  # Placeholder - replace with actual prediction
                            'home_win_probability': 0.55,  # Placeholder - replace with actual prediction
                            'away_win_probability': 0.45  # Placeholder - replace with actual prediction
                        }
                        games.append({
                            'game_id': game.get('game_id', ''),
                            'home_team': game['HomeTeam'],
                            'away_team': game['AwayTeam'],
                            'venue': game.get('Location', 'Unknown'),
                            'start_time': game.get('Time', 'TBD'),
                            'weather': weather.get('current', {}).get('condition', {}).get('text', 'Unknown'),
                            'prediction': prediction
                        })
            except Exception as e:
                continue
        if not games:
            st.info("No games scheduled for today.")
            return
        cols = st.columns(2)
        for idx, game in enumerate(games):
            with cols[idx % 2]:
                with st.expander(f"{game['home_team']} vs {game['away_team']}", expanded=True):
                    st.subheader("Predicted Score")
                    score_col1, score_col2 = st.columns(2)
                    with score_col1:
                        st.metric("Home", game['prediction']['home_team_runs'])
                    with score_col2:
                        st.metric("Away", game['prediction']['away_team_runs'])
                    st.subheader("Win Probability")
                    fig = px.pie(
                        values=[
                            game['prediction']['home_win_probability'],
                            game['prediction']['away_win_probability']
                        ],
                        names=[game['home_team'], game['away_team']],
                        title="Win Probability",
                        color_discrete_sequence=['#1f77b4', '#ff7f0e']
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        showlegend=True,
                        legend=dict(
                            bgcolor='rgba(0,0,0,0)',
                            bordercolor='white',
                            borderwidth=1
                        )
                    )
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        marker=dict(line=dict(color='white', width=2))
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"{game['game_id']}_win_prob")
                    st.subheader("Game Details")
                    details_col1, details_col2 = st.columns(2)
                    with details_col1:
                        st.write("📍 **Venue**")
                        st.write(game['venue'])
                        st.write("⏰ **Start Time**")
                        st.write(game['start_time'])
                    with details_col2:
                        st.write("🌤️ **Weather**")
                        st.write(game['weather'])
    except Exception as e:
        st.error(f"Error fetching game data: {str(e)}")
        st.error("Full error details:")
        st.error(traceback.format_exc())

def show_team_analysis():
    st.header("Team Analysis")
    
    # Team selection with correct MLB Stats API abbreviations
    team = st.selectbox(
        "Select Team",
        ["NYY", "BOS", "LAD", "SF", "HOU", "ATL", "CHC", "STL", "TOR", "TB", 
         "BAL", "CLE", "MIN", "DET", "KC", "CWS", "OAK", "SEA", "TEX", "LAA",
         "NYM", "PHI", "WSH", "MIA", "PIT", "CIN", "MIL", "COL", "ARI", "SD"]
    )
    
    # Time frame selection
    time_frame = st.selectbox(
        "Select Time Frame",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Season to Date", "Full Season"]
    )
    
    try:
        # Fetch team data
        st.write("Fetching team data...")
        batting_stats, pitching_stats = get_team_stats_cached(team)
        schedule = get_team_schedule_cached(team)
        
        # Convert date columns to datetime if they aren't already
        if 'Date' in schedule.columns and not pd.api.types.is_datetime64_any_dtype(schedule['Date']):
            schedule['Date'] = pd.to_datetime(schedule['Date'])
        
        # Calculate date range based on selected time frame
        end_date = pd.Timestamp.now(tz='UTC')
        if time_frame == "Last 7 Days":
            start_date = end_date - pd.Timedelta(days=7)
        elif time_frame == "Last 30 Days":
            start_date = end_date - pd.Timedelta(days=30)
        elif time_frame == "Last 90 Days":
            start_date = end_date - pd.Timedelta(days=90)
        elif time_frame == "Season to Date":
            start_date = pd.Timestamp(end_date.year, 3, 1, tz='UTC')  # MLB season typically starts in March
        else:  # Full Season
            start_date = pd.Timestamp(end_date.year, 1, 1, tz='UTC')
        
        # Filter data based on date range
        if 'Date' in schedule.columns:
            # Ensure schedule dates are timezone-aware
            if schedule['Date'].dt.tz is None:
                schedule['Date'] = schedule['Date'].dt.tz_localize('UTC')
            
            # Filter the schedule
            schedule = schedule[(schedule['Date'] >= start_date) & (schedule['Date'] <= end_date)]
            
            # Debug information
            st.write(f"Date range: {start_date} to {end_date}")
            st.write(f"Schedule date range: {schedule['Date'].min()} to {schedule['Date'].max()}")
            st.write(f"Number of games in range: {len(schedule)}")
        
        # after filtering or creating the schedule DataFrame, before plotting or selecting columns
        st.write("Schedule columns before deduplication:", list(schedule.columns))
        # Forcefully remove all but the first 'Date' column
        if 'Date' in schedule.columns:
            date_cols = [i for i, col in enumerate(schedule.columns) if col == 'Date']
            if len(date_cols) > 1:
                keep = date_cols[0]
                drop = [i for i in date_cols[1:]]
                schedule = schedule.drop(schedule.columns[drop], axis=1)
        st.write("Schedule columns after deduplication:", list(schedule.columns))
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Batting Statistics", "Pitching Statistics", "Team Performance"])
        
        with tab1:
            st.subheader("Batting Statistics (Per Game)")
            if not schedule.empty:
                batting_metrics = [col for col in schedule.columns if col not in ['ER', 'IP', 'SO.1', 'BB.1', 'H.1', 'HR.1', 'ERA', 'WHIP', 'SV', 'W', 'L']]  # Exclude pitching stats
                metric = st.selectbox("Select Batting Stat", batting_metrics, key="batting_metric")
                # For batting stats debug output and plotting
                st.write("Schedule columns available:", list(schedule.columns))
                if "Date" in schedule.columns and metric in schedule.columns:
                    st.write(f"Plotting {metric}:", schedule.loc[:, ~schedule.columns.duplicated()][["Date", metric]].head())
                    if pd.api.types.is_numeric_dtype(schedule[metric]) and schedule[metric].replace([np.inf, -np.inf], np.nan).dropna().shape[0] >= 2:
                        fig = px.line(
                            schedule,
                            x="Date",
                            y=metric,
                            title=f"{team} {metric} Per Game - {time_frame}"
                        )
                        fig.update_traces(
                            line=dict(color="#00baff", width=4),
                            marker=dict(size=10, color="#00baff", line=dict(width=2, color="white")),
                            mode="lines+markers"
                        )
                        fig.update_layout(
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            font=dict(color="#00baff", size=18, family="Arial"),
                            title=dict(font=dict(size=22, color="#00baff", family="Arial"), x=0.5, xanchor="center"),
                            xaxis=dict(
                                showgrid=False,
                                zeroline=False,
                                showline=False,
                                ticks="outside",
                                tickfont=dict(size=16, color="#333"),
                            ),
                            yaxis=dict(
                                showgrid=True,
                                gridcolor="#e0e0e0",
                                zeroline=False,
                                showline=False,
                                ticks="outside",
                                tickfont=dict(size=16, color="#333"),
                            ),
                            margin=dict(l=40, r=40, t=80, b=40),
                        )
                        latest_x = schedule["Date"].iloc[-1]
                        latest_y = schedule[metric].iloc[-1]
                        fig.add_scatter(
                            x=[latest_x], y=[latest_y],
                            mode="markers+text",
                            marker=dict(size=14, color="#00baff", line=dict(width=2, color="white")),
                            text=[f"{latest_y:.3f}" if isinstance(latest_y, (int, float)) else str(latest_y)],
                            textposition="middle right",
                            textfont=dict(size=18, color="#00baff", family="Arial"),
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"bat_{metric}_plot")
                    else:
                        st.info(f"Not enough numeric data to plot {metric}.")
                else:
                    st.warning(f"Cannot plot: 'Date' or '{metric}' not in schedule columns.")
            else:
                st.warning("No per-game data available for this team and time frame.")
        
        with tab2:
            st.subheader("Pitching Statistics (Per Game)")
            if not schedule.empty:
                pitching_metrics = [col for col in schedule.columns if col in ['ER', 'IP', 'SO.1', 'BB.1', 'H.1', 'HR.1', 'ERA', 'WHIP', 'SV', 'W', 'L']]
                metric = st.selectbox("Select Pitching Stat", pitching_metrics, key="pitching_metric")
                # For pitching stats debug output and plotting
                st.write("Schedule columns available:", list(schedule.columns))
                if "Date" in schedule.columns and metric in schedule.columns:
                    st.write(f"Plotting {metric}:", schedule.loc[:, ~schedule.columns.duplicated()][["Date", metric]].head())
                    if pd.api.types.is_numeric_dtype(schedule[metric]) and schedule[metric].replace([np.inf, -np.inf], np.nan).dropna().shape[0] >= 2:
                        fig = px.line(
                            schedule,
                            x="Date",
                            y=metric,
                            title=f"{team} {metric} Per Game - {time_frame}"
                        )
                        fig.update_traces(
                            line=dict(color="#00baff", width=4),
                            marker=dict(size=10, color="#00baff", line=dict(width=2, color="white")),
                            mode="lines+markers"
                        )
                        fig.update_layout(
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            font=dict(color="#00baff", size=18, family="Arial"),
                            title=dict(font=dict(size=22, color="#00baff", family="Arial"), x=0.5, xanchor="center"),
                            xaxis=dict(
                                showgrid=False,
                                zeroline=False,
                                showline=False,
                                ticks="outside",
                                tickfont=dict(size=16, color="#333"),
                            ),
                            yaxis=dict(
                                showgrid=True,
                                gridcolor="#e0e0e0",
                                zeroline=False,
                                showline=False,
                                ticks="outside",
                                tickfont=dict(size=16, color="#333"),
                            ),
                            margin=dict(l=40, r=40, t=80, b=40),
                        )
                        latest_x = schedule["Date"].iloc[-1]
                        latest_y = schedule[metric].iloc[-1]
                        fig.add_scatter(
                            x=[latest_x], y=[latest_y],
                            mode="markers+text",
                            marker=dict(size=14, color="#00baff", line=dict(width=2, color="white")),
                            text=[f"{latest_y:.3f}" if isinstance(latest_y, (int, float)) else str(latest_y)],
                            textposition="middle right",
                            textfont=dict(size=18, color="#00baff", family="Arial"),
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"pitch_{metric}_plot")
                    else:
                        st.info(f"Not enough numeric data to plot {metric}.")
                else:
                    st.warning(f"Cannot plot: 'Date' or '{metric}' not in schedule columns.")
            else:
                st.warning("No per-game data available for this team and time frame.")
        
        with tab3:
            # Team performance visualization
            st.subheader("Team Performance")
            
            if schedule is not None and not schedule.empty:
                # Calculate cumulative wins and losses
                if 'game_id' in schedule.columns:
                    # Sort by date to ensure correct cumulative calculation
                    schedule = schedule.sort_values('Date')
                    
                    # Initialize wins and losses columns
                    schedule['Wins'] = 0
                    schedule['Losses'] = 0
                    
                    # Calculate wins and losses based on home/away team
                    for idx, row in schedule.iterrows():
                        if row['HomeTeam'] == team:
                            # Home team
                            if row.get('home_team_runs', 0) > row.get('away_team_runs', 0):
                                schedule.at[idx, 'Wins'] = 1
                            else:
                                schedule.at[idx, 'Losses'] = 1
                        else:
                            # Away team
                            if row.get('away_team_runs', 0) > row.get('home_team_runs', 0):
                                schedule.at[idx, 'Wins'] = 1
                            else:
                                schedule.at[idx, 'Losses'] = 1
                    
                    # Calculate cumulative wins and losses
                    schedule['Cumulative_Wins'] = schedule['Wins'].cumsum()
                    schedule['Cumulative_Losses'] = schedule['Losses'].cumsum()
                    
                    # Create line chart for wins and losses
                    performance_data = schedule[['Date', 'Cumulative_Wins', 'Cumulative_Losses']].melt(
                        id_vars=['Date'],
                        value_vars=['Cumulative_Wins', 'Cumulative_Losses'],
                        var_name='Record',
                        value_name='Games'
                    )
                    
                    fig = px.line(
                        performance_data,
                        x='Date',
                        y='Games',
                        color='Record',
                        title=f"{team} Season Record - {time_frame}",
                        color_discrete_sequence=['#1f77b4', '#ff7f0e']
                    )
                    
                    # Update layout for better visualization
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Games",
                        hovermode="x unified",
                        showlegend=True,
                        legend_title="Record",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis=dict(
                            gridcolor='rgba(255,255,255,0.1)',
                            zerolinecolor='rgba(255,255,255,0.1)',
                            tickfont=dict(color='white'),
                            showgrid=True,
                            tickangle=45
                        ),
                        yaxis=dict(
                            gridcolor='rgba(255,255,255,0.1)',
                            zerolinecolor='rgba(255,255,255,0.1)',
                            tickfont=dict(color='white'),
                            showgrid=True
                        )
                    )
                    
                    # Add curved lines and markers
                    fig.update_traces(
                        line=dict(width=3, shape="spline"),
                        mode='lines+markers',
                        marker=dict(size=8)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key="team_performance_record")
                    
                    # Show win percentage
                    win_pct = schedule['Cumulative_Wins'] / (schedule['Cumulative_Wins'] + schedule['Cumulative_Losses'])
                    fig = px.line(
                        x=schedule['Date'],
                        y=win_pct,
                        title=f"{team} Win Percentage - {time_frame}",
                        color_discrete_sequence=['#1f77b4']
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Win Percentage",
                        hovermode="x unified",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis=dict(
                            gridcolor='rgba(255,255,255,0.1)',
                            zerolinecolor='rgba(255,255,255,0.1)',
                            tickfont=dict(color='white'),
                            showgrid=True,
                            tickangle=45
                        ),
                        yaxis=dict(
                            gridcolor='rgba(255,255,255,0.1)',
                            zerolinecolor='rgba(255,255,255,0.1)',
                            tickfont=dict(color='white'),
                            showgrid=True
                        )
                    )
                    
                    # Add curved line and markers
                    fig.update_traces(
                        line=dict(width=3, shape="spline"),
                        mode='lines+markers',
                        marker=dict(size=8)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key="team_performance_win_pct")
                    
                    # Show current record
                    st.subheader("Current Record")
                    record_col1, record_col2, record_col3 = st.columns(3)
                    with record_col1:
                        st.metric("Wins", schedule['Cumulative_Wins'].iloc[-1])
                    with record_col2:
                        st.metric("Losses", schedule['Cumulative_Losses'].iloc[-1])
                    with record_col3:
                        st.metric("Win %", f"{win_pct.iloc[-1]:.3f}")
                else:
                    st.warning("Game results data not available for this team.")
            else:
                st.warning("No schedule data available for this team.")
        
    except Exception as e:
        st.error(f"Error fetching team data: {str(e)}")
        st.error("Full error details:")
        st.error(traceback.format_exc())

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