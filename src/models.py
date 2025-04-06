import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results = pd.DataFrame({'MSRP': y_test.reset_index(drop=True), 'Predicted Output': y_pred})
    return results, mae, rmse, r2

def run_all_models(X_train, y_train, X_test, y_test):
    error_mae = []
    error_rmse = []
    error_r2 = []
    model_names = []
    results_dict = {}

    models = [
        ("Linear Regression", LinearRegression()),
        ("Support Vector Regressor", SVR()),
        ("K Nearest Regressor", KNeighborsRegressor(n_neighbors=2)),
        ("PLS Regression", PLSRegression(n_components=20)),
        ("Decision Tree Regressor", DecisionTreeRegressor(splitter='random')),
        ("Gradient Boosting Regressor", GradientBoostingRegressor())
    ]

    for name, model in models:
        results, mae, rmse, r2 = train_and_evaluate(model, X_train, y_train, X_test, y_test)
        model_names.append(name)
        error_mae.append(int(mae))
        error_rmse.append(int(rmse))
        error_r2.append(r2)
        results_dict[name] = results
        print(f"{name}: RMSE = {rmse:.2f}, MAE = {mae:.2f}, R² = {r2:.2f}")

    summary_df = pd.DataFrame({
        'Models': model_names,
        'Root Mean Squared Error': error_rmse,
        'Mean Absolute Error': error_mae,
        'R^2 Score': error_r2
    })
    return summary_df, results_dict
