Car-Prices-Prediction

Hybrid Ensemble Model for Vehicle Price Prediction

# Features
Data Ingestion & Cleaning: Handles missing values, outliers, and feature engineering (e.g., YearsOfManufacture).
Log Transformation: Applies natural log to MSRP to stabilize variance.
Recursive Feature Elimination (RFE): Selects the top N most informative features.
XGBoost Base Model: Captures nonlinear relationships among vehicle attributes.
Stacking Ensemble: Blends base model predictions with a linear meta-model for residual correction.
SHAP Explainability: Generates feature‑importance plots for model transparency.
Serialization: Saves preprocessing pipeline and models with joblib for reproducibility.

# Structure
Car-Prices-Prediction/
├── Data/                   # CSV datasets
├── notebooks/              # Jupyter notebooks from Parent Paper
├── src/                    # Source code modules
│   ├── preprocess.py       # Data cleaning and feature engineering
│   ├── feature_selection.py# RFE implementation
│   ├── model.py            # XGBoost training and stacking logic
│   ├── explain.py          # SHAP analysis scripts
│   └── utils.py            # Helper functions
├── hybrid_model.py         # Main pipeline runner
├── requirements.txt        # Project dependencies
├── README.md               # This documentation file
└── outputs/                # Model outputs, plots, and serialized artifacts

# Usage
1. Clone the repository:
`git clone https://github.com/ZelinChen823/Car-Prices-Prediction.git`
`cd Car-Prices-Prediction`