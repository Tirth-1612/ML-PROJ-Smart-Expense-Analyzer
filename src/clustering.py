"""
src/clustering.py
K-Means clustering module to discover and profile user spending behaviors.
Uses feature scaling (StandardScaler), determines optimal or configured clusters,
computes Silhouette scores, and explains resulting behavioral personas.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def prepare_clustering_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Extracts numerical behavioral features from expense transactions:
    - amount: transaction size
    - day_of_month: timing within monthly salary/bill cycle (1-31)
    - day_of_week: 0 (Mon) - 6 (Sun)
    - is_weekend: flag for weekend leisure spending
    """
    df_work = df.copy()
    df_work["date"] = pd.to_datetime(df_work["date"], errors="coerce")
    df_work = df_work.dropna(subset=["date", "amount"])

    df_work["day_of_month"] = df_work["date"].dt.day
    df_work["day_of_week"] = df_work["date"].dt.dayofweek
    df_work["is_weekend"] = df_work["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    feature_cols = ["amount", "day_of_month", "day_of_week", "is_weekend"]
    X = df_work[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=df_work.index)

    return df_work, X_scaled, scaler


def assign_persona_label(centroid: pd.Series, overall_median_amount: float) -> str:
    """
    Heuristically label a cluster centroid into an intuitive persona based on
    transaction amount, timing, and weekend distribution.
    """
    amt = centroid["amount"]
    dom = centroid["day_of_month"]
    is_wknd = centroid["is_weekend"]

    if amt > overall_median_amount * 4:
        if dom <= 10:
            return "Fixed Commitments & Major Bills (Rent / Investments)"
        return "High-Value Major Purchases & Irregular Spikes"
    elif amt > overall_median_amount * 1.5:
        if is_wknd > 0.4:
            return "Weekend Splurge & Leisure (Dining / Shopping)"
        return "Mid-Tier Household & Regular Expenses"
    else:
        if is_wknd > 0.4:
            return "Weekend Outings & Socializing"
        return "Daily Micro-Transactions (Groceries / Commute / Snacks)"


def run_spending_clustering(df: pd.DataFrame, n_clusters: int = 3) -> Dict[str, Any]:
    """
    Executes K-Means clustering on transactions and generates comprehensive profile metrics.
    """
    if df is None or len(df) < 10:
        return {
            "success": False,
            "message": "At least 10 transactions are required to perform meaningful behavioral clustering.",
            "df_clustered": pd.DataFrame(),
            "silhouette": 0.0,
            "cluster_profiles": [],
            "elbow_data": []
        }

    df_work, X_scaled, scaler = prepare_clustering_features(df)
    n_samples = len(df_work)
    k = min(n_clusters, n_samples - 1)
    if k < 2:
        k = 2

    # Run KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_work["cluster"] = cluster_labels

    # Calculate Silhouette Score
    sil_score = float(silhouette_score(X_scaled, cluster_labels)) if k > 1 and len(df_work) > k else 0.0

    # Calculate Inertia for Elbow curve (k=2 to min(8, n_samples-1))
    elbow_data = []
    max_k = min(7, n_samples - 1)
    for trial_k in range(2, max_k + 1):
        km_test = KMeans(n_clusters=trial_k, random_state=42, n_init=5).fit(X_scaled)
        elbow_data.append({"k": trial_k, "inertia": round(km_test.inertia_, 2)})

    # Profile each cluster
    cluster_profiles = []
    overall_median = float(df_work["amount"].median())

    for c_id in sorted(df_work["cluster"].unique()):
        sub_df = df_work[df_work["cluster"] == c_id]
        count = len(sub_df)
        pct = (count / n_samples) * 100
        mean_amt = float(sub_df["amount"].mean())
        median_amt = float(sub_df["amount"].median())
        top_cats = sub_df["category"].value_counts().head(3).to_dict()
        top_modes = sub_df["payment_method"].value_counts().head(2).to_dict()
        avg_dom = float(sub_df["day_of_month"].mean())
        weekend_pct = float(sub_df["is_weekend"].mean()) * 100

        centroid_raw = pd.Series({
            "amount": mean_amt,
            "day_of_month": avg_dom,
            "is_weekend": sub_df["is_weekend"].mean()
        })
        persona = assign_persona_label(centroid_raw, overall_median)

        cluster_profiles.append({
            "cluster_id": int(c_id),
            "persona_name": persona,
            "count": count,
            "percentage": round(pct, 1),
            "mean_amount": round(mean_amt, 2),
            "median_amount": round(median_amt, 2),
            "avg_day_of_month": round(avg_dom, 1),
            "weekend_percentage": round(weekend_pct, 1),
            "top_categories": top_cats,
            "top_payment_methods": top_modes
        })

    # Sort profiles by mean amount
    cluster_profiles.sort(key=lambda x: x["mean_amount"])
    # Map friendly sorted names
    for i, profile in enumerate(cluster_profiles):
        profile["cluster_label"] = f"Cluster {profile['cluster_id']}: {profile['persona_name']}"

    return {
        "success": True,
        "df_clustered": df_work,
        "silhouette": round(sil_score, 3),
        "inertia": round(kmeans.inertia_, 2),
        "k": k,
        "cluster_profiles": cluster_profiles,
        "elbow_data": elbow_data
    }
