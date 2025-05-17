# EPL Champion Prediction System

This project uses advanced football metrics and machine learning to forecast the next English Premier League champion.

## Project Structure

```
epl_prediction_system/
│
├── data/                      # Data directory
│   ├── raw/                   # Raw data from APIs/web scraping
│   └── processed/             # Cleaned and processed datasets
│
├── src/                       # Source code
│   ├── data_collection.py     # Scripts for collecting data from APIs/web scraping
│   ├── data_processing.py     # Data cleaning and feature engineering
│   ├── modeling.py            # ML model training and evaluation
│   └── visualization.py       # Data visualization functions
│
├── notebooks/                 # Jupyter notebooks for exploration and analysis
│
├── models/                    # Saved trained models
│
├── visualizations/            # Generated visualizations
│
├── main.py                    # Main script to run the prediction system
├── app.py                     # Streamlit dashboard (optional)
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Data Sources

- Football-Data.org API
- Understat (web scraping)
- FBref (web scraping)

## Features Used

- Team statistics (points, wins, draws, losses)
- Expected Goals (xG) and Expected Goals Against (xGA)
- Possession percentage
- Shots per 90 minutes
- Pass completion rate
- Team form over last 5 matches
- xG differential
- Strength of schedule

## Models

- Random Forest Classifier
- XGBoost
- Logistic Regression

## Setup and Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the main script:
   ```
   python main.py
   ```

## Results

The prediction results for the upcoming season will be available in the visualizations directory and through the Streamlit dashboard (if implemented).

## License

This project is licensed under the MIT License - see the LICENSE file for details.