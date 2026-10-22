"""
src/anomaly.py
Isolation Forest anomaly detection engine for identifying unusual, fraudulent, or outlier expenses.
Calculates anomaly scores, contextual risk factors, and human-readable explanations.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.03) -> Dict[str, Any]:
    """
    Fits an Isolation Forest model to detect unusual transactions based on:
    - Transaction amount
    - Ratio compared to category median
    - Timing features (day of month, day of week)
    """
    if df is None or len(df) < 15:
        return {
            "success": False,
            "message": "At least 15 transactions are required to train the Isolation Forest anomaly detector.",
            "df_anomalies": pd.DataFrame(),
            "anomaly_count": 0,
            "anomaly_percentage": 0.0
        }

    df_work = df.copy()
    df_work["date"] = pd.to_datetime(df_work["date"], errors="coerce")
    df_work = df_work.dropna(subset=["date", "amount"])

    if len(df_work) < 15:
        return {
            "success": False,
            "message": "Insufficient valid data records.",
            "df_anomalies": pd.DataFrame(),
            "anomaly_count": 0,
            "anomaly_percentage": 0.0
        }

    # Compute category-level median and std for contextual comparison
    cat_medians = df_work.groupby("category")["amount"].transform("median")
    cat_means = df_work.groupby("category")["amount"].transform("mean")
    cat_stds = df_work.groupby("category")["amount"].transform("std").fillna(1.0)
    cat_stds = cat_stds.replace(0, 1.0)

    df_work["cat_median"] = cat_medians
    df_work["ratio_to_cat_median"] = df_work["amount"] / (cat_medians + 1.0)
    df_work["z_score_cat"] = (df_work["amount"] - cat_means) / cat_stds
    df_work["day_of_month"] = df_work["date"].dt.day
    df_work["day_of_week"] = df_work["date"].dt.dayofweek

    # Feature matrix for Isolation Forest
    feature_cols = ["amount", "ratio_to_cat_median", "day_of_month", "day_of_week"]
    X = df_work[feature_cols].copy()

    # Isolation Forest setup
    # Contamination bounded between 0.01 and 0.15
    clamped_contam = max(0.01, min(0.15, float(contamination)))
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=clamped_contam,
        random_state=42
    )

    preds = iso_forest.fit_predict(X)
    # decision_function: lower means more abnormal
    scores = iso_forest.decision_function(X)

    df_work["is_anomaly"] = (preds == -1)
    # Normalize anomaly severity score between 0 and 100 (100 = most anomalous)
    min_score, max_score = scores.min(), scores.max()
    if max_score > min_score:
        severity = 100 * (max_score - scores) / (max_score - min_score)
    else:
        severity = np.zeros_like(scores)

    df_work["anomaly_severity"] = np.round(severity, 1)

    # Generate friendly plain-English reason for anomalies
    reasons: List[str] = []
    overall_median = df_work["amount"].median()

    for idx, row in df_work.iterrows():
        if not row["is_anomaly"]:
            reasons.append("Normal spending pattern")
            continue

        parts = []
        amt = row["amount"]
        cat = row["category"]
        c_median = row["cat_median"]

        if c_median > 0 and amt >= 2.5 * c_median:
            ratio = round(amt / c_median, 1)
            parts.append(f"Amount is {ratio}x higher than median for {cat} (₹{c_median:,.0f})")
        elif amt >= 3 * overall_median:
            parts.append(f"Significantly above overall median transaction (₹{overall_median:,.0f})")

        if row["day_of_month"] > 25 and amt > 3000:
            parts.append("High unexpected expenditure near month-end")

        if not parts:
            parts.append(f"Unusual combination of amount (₹{amt:,.0f}) and category timing")

        reasons.append(" • ".join(parts))

    df_work["anomaly_reason"] = reasons

    anomalies_only = df_work[df_work["is_anomaly"]].copy()
    anomalies_only = anomalies_only.sort_values(by="anomaly_severity", ascending=False)

    return {
        "success": True,
        "df_all": df_work,
        "df_anomalies": anomalies_only,
        "anomaly_count": len(anomalies_only),
        "total_count": len(df_work),
        "anomaly_percentage": round((len(anomalies_only) / len(df_work)) * 100, 2),
        "contamination": clamped_contam
    }
