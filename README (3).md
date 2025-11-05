# 🍽️ Restaurant Forecasting Dashboard

Interactive dashboard for restaurant sales and bills forecasting using Prophet models.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](INSERISCI_QUI_IL_TUO_LINK)

---

## 📊 Overview

Prophet models for **6 restaurants** and **2 dimensions** (bills and total sales) were fitted using **temporal cross-validation** and **grid search** to test different configurations of seasonalities, holidays, regressors, and solutions to handle COVID-19 influence.

The models were deployed in an **interactive Streamlit dashboard** that allows users to:

- 📈 **Generate predictions** using fitted models
- 🔄 **Update models** with new data
- 📉 **Compare predicted vs real values**
- 🔍 **Analyze components** (trend, holidays effect, seasonalities)

---

## 🚀 Live Demo

**View the deployed dashboard:** [INSERISCI_QUI_IL_TUO_LINK](INSERISCI_QUI_IL_TUO_LINK)

---

## 💻 Run Locally

Clone the repository and run:

```bash
git clone https://github.com/TUO_USERNAME/restaurant-forecast-dashboard.git
cd restaurant-forecast-dashboard
pip install -r requirements.txt
streamlit run prophet_dashboard.py
```

The dashboard will open automatically at `http://localhost:8501`

---

## 🛠️ Tech Stack

- **Prophet** - Time series forecasting
- **Streamlit** - Interactive web interface
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation

---

## 📁 Project Structure

```
restaurant-forecast-dashboard/
├── prophet_dashboard.py          # Main application
├── requirements.txt              # Dependencies
├── README.md                     # Documentation
└── models/                       # Pre-trained models (24 files)
    ├── prophet_model_*.json
    └── *_metadata.json
```

---

## 📊 Data Format

CSV files must include:

```csv
data,scontrini,totale
2024-01-01,150,5000
2024-01-02,165,5500
```

Where:
- `data` - Date (YYYY-MM-DD)
- `scontrini` - Number of bills
- `totale` - Total sales

---

## 👨‍💻 Author

Created for Data Science project - [TUO NOME]

---

⭐ **Star this repo if you find it useful!** ⭐

