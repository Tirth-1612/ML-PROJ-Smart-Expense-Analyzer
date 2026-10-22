"""
src/visualizer.py
Modern Plotly chart generation module for financial and machine learning dashboards.
Uses Indian Rupee (₹) formatting and a cohesive modern color palette.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Aesthetic color palette
PALETTE = [
    "#4361EE", "#3A0CA3", "#7209B7", "#F72585", "#4CC9F0",
    "#06D6A0", "#FFB703", "#FB8500", "#118AB2", "#EF476F"
]

LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, -apple-system, sans-serif", size=12),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=40, b=20),
    hoverlabel=dict(font_size=12, font_family="Inter, sans-serif")
)


def plot_monthly_trend(df: pd.DataFrame) -> go.Figure:
    """Monthly total expenditure bar and trend chart."""
    df_work = df.copy()
    df_work["date"] = pd.to_datetime(df_work["date"])
    monthly = df_work.groupby(df_work["date"].dt.to_period("M"))["amount"].sum().reset_index()
    monthly["month_str"] = monthly["date"].astype(str)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["month_str"],
        y=monthly["amount"],
        marker_color="#4361EE",
        name="Monthly Spend",
        hovertemplate="<b>%{x}</b><br>Total: ₹%{y:,.2f}<extra></extra>"
    ))

    # Add trend line
    fig.add_trace(go.Scatter(
        x=monthly["month_str"],
        y=monthly["amount"],
        mode="lines+markers",
        line=dict(color="#F72585", width=2.5, shape="spline"),
        marker=dict(size=6, color="#F72585"),
        name="Trend"
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="<b>Monthly Spending Trend</b>",
        xaxis_title="Month",
        yaxis_title="Total Amount (₹)",
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_category_breakdown(df: pd.DataFrame) -> go.Figure:
    """Category breakdown donut chart."""
    cat_df = df.groupby("category")["amount"].sum().reset_index()
    cat_df = cat_df.sort_values(by="amount", ascending=False)

    fig = px.pie(
        cat_df,
        names="category",
        values="amount",
        hole=0.55,
        color_discrete_sequence=PALETTE,
        title="<b>Expenditure by Category</b>"
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Amount: ₹%{value:,.2f}<br>Share: %{percent}<extra></extra>"
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        showlegend=False
    )
    return fig


def plot_category_horizontal_bars(df: pd.DataFrame) -> go.Figure:
    """Horizontal ranked bar chart for category spending."""
    cat_df = df.groupby("category")["amount"].agg(["sum", "count"]).reset_index()
    cat_df.columns = ["category", "total_amount", "txn_count"]
    cat_df = cat_df.sort_values(by="total_amount", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cat_df["category"],
        x=cat_df["total_amount"],
        orientation="h",
        marker=dict(
            color=cat_df["total_amount"],
            colorscale="Viridis",
            showscale=False
        ),
        hovertemplate="<b>%{y}</b><br>Spent: ₹%{x:,.2f}<extra></extra>"
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="<b>Category Spending Ranking</b>",
        xaxis_title="Total Amount (₹)",
        yaxis_title="",
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(showgrid=False)
    )
    return fig


def plot_payment_methods(df: pd.DataFrame) -> go.Figure:
    """Payment method distribution pie chart."""
    pm_df = df.groupby("payment_method")["amount"].sum().reset_index()
    fig = px.pie(
        pm_df,
        names="payment_method",
        values="amount",
        hole=0.45,
        color_discrete_sequence=["#06D6A0", "#4361EE", "#FFB703", "#7209B7", "#EF476F"],
        title="<b>Payment Methods Distribution</b>"
    )
    fig.update_traces(
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Amount: ₹%{value:,.2f}<br>Share: %{percent}<extra></extra>"
    )
    fig.update_layout(**LAYOUT_DEFAULTS, showlegend=False)
    return fig


def plot_clusters_scatter(df_clustered: pd.DataFrame) -> go.Figure:
    """Scatter plot displaying behavioral clusters (Amount vs Day of Month)."""
    fig = px.scatter(
        df_clustered,
        x="day_of_month",
        y="amount",
        color="cluster",
        color_continuous_scale="Turbo",
        hover_data=["description", "category", "payment_method"],
        title="<b>Behavioral Spending Clusters (K-Means)</b>",
        labels={"day_of_month": "Day of Month (1-31)", "amount": "Transaction Amount (₹)", "cluster": "Cluster ID"}
    )
    fig.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=1, color="rgba(255,255,255,0.4)")))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)", dtick=5)
    )
    return fig


def plot_forecast_comparison(
    monthly_history: List[Dict[str, Any]],
    eval_comparison: List[Dict[str, Any]],
    next_month_label: str,
    next_month_pred: float,
    model_name: str
) -> go.Figure:
    """Timeline comparing historical expenses, test holdout actual vs pred, and future forecast."""
    df_hist = pd.DataFrame(monthly_history)
    fig = go.Figure()

    # Historical Actual Line
    fig.add_trace(go.Scatter(
        x=df_hist["year_month_str"],
        y=df_hist["total_amount"],
        mode="lines+markers",
        name="Historical Actual Spend",
        line=dict(color="#4361EE", width=3),
        marker=dict(size=7)
    ))

    # Test Split Predictions (Holdout evaluation)
    if eval_comparison:
        eval_df = pd.DataFrame(eval_comparison)
        fig.add_trace(go.Scatter(
            x=eval_df["month"],
            y=eval_df["linear_regression_pred"],
            mode="lines+markers",
            name="Linear Reg Test Fit",
            line=dict(color="#FFB703", width=2, dash="dash"),
            marker=dict(size=6, symbol="x")
        ))
        if "random_forest_pred" in eval_df.columns:
            fig.add_trace(go.Scatter(
                x=eval_df["month"],
                y=eval_df["random_forest_pred"],
                mode="lines+markers",
                name="Random Forest Test Fit",
                line=dict(color="#06D6A0", width=2, dash="dot"),
                marker=dict(size=6, symbol="diamond")
            ))

    # Next Month Forecast Point
    last_hist_month = df_hist["year_month_str"].iloc[-1]
    last_hist_amt = df_hist["total_amount"].iloc[-1]

    fig.add_trace(go.Scatter(
        x=[last_hist_month, next_month_label],
        y=[last_hist_amt, next_month_pred],
        mode="lines+markers",
        name=f"Forecast ({model_name})",
        line=dict(color="#F72585", width=3, dash="dot"),
        marker=dict(size=11, symbol="star", color="#F72585"),
        hovertemplate="<b>Projected: %{x}</b><br>Estimate: ₹%{y:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="<b>Monthly Expenditure Trajectory & Next-Month Forecast</b>",
        xaxis_title="Month",
        yaxis_title="Total Expenditure (₹)",
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_anomalies_scatter(df_all: pd.DataFrame) -> go.Figure:
    """Highlights normal transactions vs Isolation Forest anomalies."""
    fig = go.Figure()

    normal_df = df_all[~df_all["is_anomaly"]]
    anom_df = df_all[df_all["is_anomaly"]]

    fig.add_trace(go.Scatter(
        x=normal_df["date"],
        y=normal_df["amount"],
        mode="markers",
        name="Normal Transaction",
        marker=dict(color="#4361EE", size=6, opacity=0.6),
        hovertemplate="<b>%{customdata[0]}</b><br>Date: %{x}<br>Amount: ₹%{y:,.2f}<br>Category: %{customdata[1]}<extra></extra>",
        customdata=normal_df[["description", "category"]].values
    ))

    fig.add_trace(go.Scatter(
        x=anom_df["date"],
        y=anom_df["amount"],
        mode="markers",
        name="Anomaly (Flagged)",
        marker=dict(color="#EF476F", size=12, symbol="circle-open-dot", line=dict(width=2, color="#EF476F")),
        hovertemplate="<b>⚠️ ANOMALY: %{customdata[0]}</b><br>Date: %{x}<br>Amount: ₹%{y:,.2f}<br>Category: %{customdata[1]}<br>Reason: %{customdata[2]}<extra></extra>",
        customdata=anom_df[["description", "category", "anomaly_reason"]].values
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="<b>Anomaly Detection (Isolation Forest Outliers)</b>",
        xaxis_title="Date",
        yaxis_title="Transaction Amount (₹)",
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_budget_comparison(budget_breakdown: Dict[str, Any]) -> go.Figure:
    """Compares actual spending vs recommended 50/30/20 budget framework."""
    categories = ["Needs (Essentials)", "Wants (Discretionary)", "Savings / Investments"]
    actuals = [
        budget_breakdown.get("Needs", {}).get("percentage", 0),
        budget_breakdown.get("Wants", {}).get("percentage", 0),
        budget_breakdown.get("Savings", {}).get("percentage", 0)
    ]
    recommended = [50, 30, 20]

    fig = go.Figure(data=[
        go.Bar(name="Your Actual %", x=categories, y=actuals, marker_color="#4361EE", hovertemplate="%{y}%<extra></extra>"),
        go.Bar(name="Recommended % (50/30/20)", x=categories, y=recommended, marker_color="#06D6A0", hovertemplate="%{y}%<extra></extra>")
    ])
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        barmode="group",
        title="<b>50 / 30 / 20 Budget Health Comparison</b>",
        yaxis_title="Percentage of Total Spending (%)",
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)", range=[0, max(max(actuals, default=50) + 10, 60)]),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
