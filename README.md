# Car-Prices-Prediction

## Hybrid Ensemble Model for Vehicle Price Prediction

# Features
Data Ingestion & Cleaning: Handles missing values, outliers, and feature engineering (e.g., YearsOfManufacture).\
Log Transformation: Applies natural log to MSRP to stabilize variance.\
Recursive Feature Elimination (RFE): Selects the top N most informative features.\
XGBoost Base Model: Captures nonlinear relationships among vehicle attributes.\
Stacking Ensemble: Blends base model predictions with a linear meta-model for residual correction.\
SHAP Explainability: Generates feature‑importance plots for model transparency.\
Serialization: Saves preprocessing pipeline and models with joblib for reproducibility.

# Structure

```
Car-Prices-Prediction/\
├── Data/                   # CSV datasets
├── Jupyter Notebook        # Jupyter notebooks from Parent Paper
├── src/                    # Source code modules
│   ├── data_loader.py      # Load raw data
│   ├── eda.py              # Exploratory data analysis
│   ├── preprocessing.py    # Data cleaning and feature engineering
│   ├── models.py           # Traditional Model training
│   ├── visualization.py    # SHAP and other visualization tools
│   ├── main.py             # Entry for running the pipeline
│   └── hybrid_model.py     # Hybrid ensemble model
├── requirements.txt        # Project dependencies
└── README.md               # This document
```
# Usage
**1. Clone the repository:**
```
git clone https://github.com/ZelinChen823/Car-Prices-Prediction.git
cd Car-Prices-Prediction
```
**2. Install dependencies**
```
pip install -r requirements.txt
```
**3. Run the pipeline model**\
Run hybrid_model.py in any IDE your prefer.\
Or in Terminal:
```
cd .\src\
python .\hybrid_model.py
```

## Evaluation Metrics

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R-Squared** (Coefficient of Determination)