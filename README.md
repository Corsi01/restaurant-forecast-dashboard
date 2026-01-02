# Restaurant Forecasting Dashboard

Interactive dashboard for restaurant total sales and number of bills forecasting using Prophet models.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://restaurant-forecast-dashboard.streamlit.app/)

---

## Overview

Prophet models for **6 restaurants** and **2 dimensions** (bills and total sales) were fitted using **temporal cross-validation** and **grid search** to test different configurations of seasonalities, holidays, regressors, and solutions to handle COVID-19 influence.

The models were deployed in an **interactive Streamlit dashboard** that allows users to:

- **Generate predictions** using fitted models
- **Update models** with new data
- **Compare predicted vs real values**
- **Analyze components** (trend, holidays effect, seasonalities)

---

## Live Demo

View the deployed dashboard: https://restaurant-forecast-dashboard.streamlit.app/

---

## Run Locally

Clone the repository and run:
```bash
git clone https://github.com/Corsi01/restaurant-forecast-dashboard.git
cd restaurant-forecast-dashboard
pip install -r requirements.txt
streamlit run prophet_dashboard.py
```

The dashboard will open automatically at `http://localhost:8501`

---

## Project Structure
```
restaurant-forecast-dashboard/
├── prophet_dashboard.py          # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # Documentation
├── models/                       # Pre-trained Prophet models (12 models × 2 files each)
│   ├── Ristorante_1/
│   │   ├── Scontrini.json
│   │   └── Totale.json
│   ├── Ristorante_2/
│   │   ├── Scontrini.json
│   │   └── Totale.json
│   └── ...                       # Additional restaurants
│
└── data/                         # Dataset files
    ├── original/                 # Raw historical data
    │   └── complete_dataset.csv
    ├── preprocessed/             # Cleaned data (post Sep 1, 2018)
    │   ├── Ristorante_1.csv
    │   ├── Ristorante_2.csv
    │   └── ...                   # Individual restaurant files   -->  # 6 CSV files for testing dashboard features
    └── covid/                    # COVID-19 lockdown timelines
        ├── lombardia_lockdowns.csv
        └── emilia_romagna_lockdowns.csv
        # Generated using GPT-5 Deep Search
```

---

## Data Format

CSV files must include the following columns:
```csv
data,scontrini,totale
2024-01-01,150,5000
2024-01-02,165,5500
```

Where:
- `data` - Date (YYYY-MM-DD format)
- `scontrini` - Number of bills (daily transaction count)
- `totale` - Total sales (daily revenue in €)

---

## Testing the Dashboard

The `data/` folder contains **6 CSV files** with the last **60 days** of each restaurant's data (hold-out test set). These files can be used to test dashboard functionalities:

- **Model Update:** Upload a holdout file to retrain a model with new data
- **Forecast Validation:** Compare model predictions against actual values from the test period

Example usage:
1. Navigate to **"Retrain & Forecast"** tab
2. Upload `Ristorante_1_holdout.csv`
3. Generate updated forecasts with the retrained model

---

## Model Training Details

All models were optimized using:
- **Grid Search** over seasonality configurations (weekly, monthly, yearly)
- **Temporal Cross-Validation** with rolling origin
- **COVID-19 Regressors** based on regional lockdown timelines (Lombardia, Emilia Romagna)
- **Holiday Effects** for Italian national holidays

---
