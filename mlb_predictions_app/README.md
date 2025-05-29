# MLB Game Predictions

A comprehensive application that predicts scores for Major League Baseball games using advanced analytics, machine learning, and real-time data integration.

## Features

- Real-time MLB game predictions
- Team performance analysis
- Weather impact analysis
- Injury report integration
- Betting odds analysis
- Historical performance tracking
- Interactive dashboard

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- API keys for:
  - Weather API (e.g., OpenWeatherMap)
  - Odds API (e.g., The Odds API)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mlb-predictions.git
   cd mlb-predictions
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with the following variables:
   ```
   WEATHER_API_KEY=your_weather_api_key_here
   ODDS_API_KEY=your_odds_api_key_here
   DEBUG=False
   LOG_LEVEL=INFO
   ```

## Usage

1. Start the application:
   ```bash
   python run.py
   ```

2. Access the dashboard at `http://localhost:8501`

## Project Structure

```
mlb_predictions/
├── data/                  # Data storage
├── models/               # ML models
├── src/
│   ├── data/            # Data collection and processing
│   ├── features/        # Feature engineering
│   ├── models/          # Model training and prediction
│   ├── api/             # API endpoints
│   └── utils/           # Utility functions
├── tests/               # Test files
├── notebooks/           # Jupyter notebooks for analysis
├── app.py              # Main application entry point
├── run.py              # Application runner
└── requirements.txt    # Python dependencies
```

## Data Sources

- MLB Stats API
- Weather API
- Odds API
- Baseball Reference
- Team injury reports

## Model Training

The model is trained on historical MLB data from 2018 to the present. To retrain the model:

```bash
python src/models/train_model.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Contact

For questions or feedback, please contact the development team. 