"""
Application Runner Script

This script runs the MLB prediction application, handling:
- Environment setup
- Model loading
- Application startup
"""

import os
import sys
import subprocess
from datetime import datetime
import streamlit as st

def setup_environment():
    """Set up the application environment."""
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Check for required environment variables
    required_vars = ['WEATHER_API_KEY', 'ODDS_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        sys.exit(1)

def check_model_exists():
    """Check if a trained model exists."""
    model_files = [f for f in os.listdir("models") if f.startswith("mlb_model_")]
    return len(model_files) > 0

def train_model_if_needed():
    """Train the model if no trained model exists."""
    if not check_model_exists():
        print("No trained model found. Training new model...")
        try:
            subprocess.run(
                [sys.executable, "src/models/train_model.py"],
                check=True
            )
            print("Model training completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error training model: {str(e)}")
            sys.exit(1)

def run_application():
    """Run the Streamlit application."""
    try:
        subprocess.run(
            ["streamlit", "run", "app.py"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running application: {str(e)}")
        sys.exit(1)

def main():
    """Main application runner."""
    print("Starting MLB Prediction Application...")
    
    # Set up environment
    setup_environment()
    
    # Train model if needed
    train_model_if_needed()
    
    # Run application
    print("Starting application...")
    run_application()

if __name__ == "__main__":
    main() 