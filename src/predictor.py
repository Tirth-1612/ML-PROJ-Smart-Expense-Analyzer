"""
src/predictor.py
Expense Prediction engine utilizing Linear Regression baseline and Random Forest Regressor.
Performs chronological train/test splits, computes MAE and RMSE, and forecasts next month's expenditure.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def aggregate_monthly_expenses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transactions into a monthly time-series dataframe with
    lagged and seasonal features.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df_work = df.copy()
    df_work["date"] = pd.to_datetime(df_work["date"], errors="coerce")
    df_work = df_work.dropna(subset=["date", "amount"])

    if df_work.empty:
        return pd.DataFrame()

    df_work["year_month"] = df_work["date"].dt.to_period("M")
    monthly = df_work.groupby("year_month").agg(
        total_amount=("amount", "sum"),
        txn_count=("amount", "count"),
        avg_txn_size=("amount", "mean"),
        max_txn_size=("amount", "max")
    ).reset_index()

    monthly["year_month_str"] = monthly["year_month"].astype(str)
    monthly["year"] = monthly["year_month"].apply(lambda p: p.year)
    monthly["month"] = monthly["year_month"].apply(lambda p: p.month)
    monthly = monthly.sort_values("year_month").reset_index(drop=True)

    monthly["month_idx"] = np.arange(len(monthly))

    # Lags & rolling features
    monthly["lag_1"] = monthly["total_amount"].shift(1)
    monthly["lag_2"] = monthly["total_amount"].shift(2)
    monthly["rolling_avg_3"] = monthly["total_amount"].rolling(window=3, min_periods=1).mean().shift(1)

    return monthly


def predict_next_month(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Builds baseline Linear Regression and Random Forest models on chronological data.
    Evaluates historical accuracy on a holdout test period and predicts next month's expenditure.
    """
    monthly = aggregate_monthly_expenses(df)
    n_months = len(monthly)

    if n_months < 3:
        return {
            "success": False,
            "message": f"Historical data contains only {n_months} month(s). At least 3 distinct calendar months are required for time-series forecasting.",
            "monthly_history": monthly.to_dict(orient="records") if not monthly.empty else []
        }

    # If data has between 3 and 5 months, we can build a baseline Linear Regression with simple trend
    use_rf = n_months >= 6
    test_size = 1 if n_months <= 4 else (2 if n_months <= 8 else 3)
    train_size = n_months - test_size

    # Prepare features
    # For small datasets, use simple trend + month seasonal; for larger, include lags
    if n_months >= 5:
        # Fill first lag NA with backfill
        monthly_clean = monthly.copy()
        monthly_clean["lag_1"] = monthly_clean["lag_1"].bfill()
        monthly_clean["lag_2"] = monthly_clean["lag_2"].bfill()
        monthly_clean["rolling_avg_3"] = monthly_clean["rolling_avg_3"].bfill()
        feature_cols = ["month_idx", "month", "lag_1", "rolling_avg_3"]
    else:
        monthly_clean = monthly.copy()
        feature_cols = ["month_idx", "month"]

    X = monthly_clean[feature_cols]
    y = monthly_clean["total_amount"]

    # Chronological Split (prevents temporal data leakage)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    test_periods = monthly_clean["year_month_str"].iloc[train_size:].tolist()

    # Model 1: Linear Regression (Baseline)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_pred_lr = np.clip(y_pred_lr, a_min=0, a_max=None)  # No negative expenses

    mae_lr = float(mean_absolute_error(y_test, y_pred_lr))
    rmse_lr = float(root_mean_squared_error(y_test, y_pred_lr))

    # Model 2: Random Forest (if sufficient data)
    rf_results = None
    if use_rf:
        rf = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        y_pred_rf = np.clip(y_pred_rf, a_min=0, a_max=None)
        mae_rf = float(mean_absolute_error(y_test, y_pred_rf))
        rmse_rf = float(root_mean_squared_error(y_test, y_pred_rf))
        rf_results = {
            "name": "Random Forest Regressor",
            "y_pred": [round(float(v), 2) for v in y_pred_rf],
            "mae": round(mae_lr if np.isnan(mae_rf) else mae_rf, 2),
            "rmse": round(rmse_lr if np.isnan(rmse_rf) else rmse_rf, 2)
        }

    # Now fit models on ALL data to forecast the upcoming Month (t+1)
    lr_full = LinearRegression()
    lr_full.fit(X, y)

    last_row = monthly_clean.iloc[-1]
    last_period = last_row["year_month"]
    next_period = last_period + 1
    next_month_idx = last_row["month_idx"] + 1
    next_month_num = next_period.month

    if "lag_1" in feature_cols:
        next_features = pd.DataFrame([{
            "month_idx": next_month_idx,
            "month": next_month_num,
            "lag_1": float(last_row["total_amount"]),
            "rolling_avg_3": float(monthly_clean["total_amount"].tail(3).mean())
        }])
    else:
        next_features = pd.DataFrame([{
            "month_idx": next_month_idx,
            "month": next_month_num
        }])

    forecast_lr = float(np.clip(lr_full.predict(next_features)[0], 0, None))

    forecast_rf = None
    if use_rf:
        rf_full = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
        rf_full.fit(X, y)
        forecast_rf = float(np.clip(rf_full.predict(next_features)[0], 0, None))

    # Determine recommended prediction
    if use_rf and rf_results and rf_results["mae"] <= mae_lr:
        best_model_name = "Random Forest Regressor"
        primary_forecast = forecast_rf
    else:
        best_model_name = "Linear Regression (Baseline)"
        primary_forecast = forecast_lr

    # Test evaluation comparison table
    eval_comparison = []
    for idx, period in enumerate(test_periods):
        actual_val = float(y_test.iloc[idx])
        lr_val = float(y_pred_lr[idx])
        item = {
            "month": period,
            "actual": round(actual_val, 2),
            "linear_regression_pred": round(lr_val, 2),
            "lr_error": round(abs(actual_val - lr_val), 2)
        }
        if rf_results:
            rf_val = float(y_pred_rf[idx])
            item["random_forest_pred"] = round(rf_val, 2)
            item["rf_error"] = round(abs(actual_val - rf_val), 2)
        eval_comparison.append(item)

    return {
        "success": True,
        "n_months": n_months,
        "train_size": train_size,
        "test_size": test_size,
        "next_month_label": str(next_period),
        "primary_forecast": round(primary_forecast, 2),
        "best_model_name": best_model_name,
        "lr_forecast": round(forecast_lr, 2),
        "lr_mae": round(mae_lr, 2),
        "lr_rmse": round(rmse_lr, 2),
        "rf_forecast": round(forecast_rf, 2) if forecast_rf is not None else None,
        "rf_results": rf_results,
        "eval_comparison": eval_comparison,
        "monthly_history": monthly.to_dict(orient="records"),
        "features_used": feature_cols
    }
