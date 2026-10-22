"""
tests/test_ml_models.py
Unit tests for core Machine Learning algorithms:
1. K-Means clustering (Spending Behaviour)
2. Linear Regression & Random Forest (Expense Prediction)
3. Isolation Forest (Anomaly Detection)
4. Dynamic Saving Recommendations
"""

import os
import pandas as pd
from src.clustering import run_spending_clustering
from src.predictor import predict_next_month, aggregate_monthly_expenses
from src.anomaly import detect_anomalies
from src.recommendations import generate_saving_recommendations

SAMPLE_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_expenses.csv")


def test_kmeans_clustering():
    df = pd.read_csv(SAMPLE_CSV)
    res = run_spending_clustering(df, n_clusters=3)

    assert res["success"] is True
    assert "df_clustered" in res
    assert "silhouette" in res
    assert res["k"] == 3
    assert len(res["cluster_profiles"]) == 3
    # Check that each cluster profile has persona name and stats
    for p in res["cluster_profiles"]:
        assert "persona_name" in p
        assert p["count"] > 0
        assert p["mean_amount"] >= 0


def test_expense_prediction():
    df = pd.read_csv(SAMPLE_CSV)
    res = predict_next_month(df)

    assert res["success"] is True
    assert res["n_months"] >= 6
    assert res["primary_forecast"] > 0
    assert "best_model_name" in res
    assert res["lr_mae"] >= 0
    assert res["lr_rmse"] >= 0
    assert len(res["eval_comparison"]) > 0


def test_anomaly_detection():
    df = pd.read_csv(SAMPLE_CSV)
    res = detect_anomalies(df, contamination=0.03)

    assert res["success"] is True
    assert res["anomaly_count"] > 0
    assert not res["df_anomalies"].empty

    # Top anomalies should include our prominent spike (e.g. gold jewelry or car overhaul)
    top_anom = res["df_anomalies"].iloc[0]
    assert bool(top_anom["is_anomaly"]) is True
    assert len(top_anom["anomaly_reason"]) > 0


def test_saving_recommendations():
    df = pd.read_csv(SAMPLE_CSV)
    res = generate_saving_recommendations(df)

    assert res["success"] is True
    assert res["total_spend"] > 0
    assert "budget_breakdown" in res
    assert "Needs" in res["budget_breakdown"]
    assert "Wants" in res["budget_breakdown"]
    assert "Savings" in res["budget_breakdown"]
    assert len(res["recommendations"]) >= 2
