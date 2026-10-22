"""
src/recommendations.py
Generates personalized, data-driven saving opportunities and financial recommendations.
Analyzes 50/30/20 budget framework, recurring subscriptions, dining habits, and discretionary leakage.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

NEEDS_CATEGORIES = {"Rent & Housing", "Groceries", "Utilities & Bills", "Transportation", "Health & Fitness"}
WANTS_CATEGORIES = {"Food & Dining", "Shopping", "Entertainment & Subscriptions", "Miscellaneous"}
SAVINGS_CATEGORIES = {"Investments"}


def generate_saving_recommendations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes financial health metrics and generates targeted, actionable savings tips.
    """
    if df is None or df.empty:
        return {
            "success": False,
            "message": "No expense data available to generate saving recommendations.",
            "recommendations": [],
            "budget_breakdown": {}
        }

    df_work = df.copy()
    df_work["date"] = pd.to_datetime(df_work["date"], errors="coerce")
    df_work = df_work.dropna(subset=["date", "amount"])

    total_spend = df_work["amount"].sum()
    if total_spend <= 0:
        return {
            "success": False,
            "message": "Total expenditure is zero.",
            "recommendations": [],
            "budget_breakdown": {}
        }

    # 1. 50/30/20 Budget Rule Allocation
    def classify_budget_bucket(cat: str) -> str:
        if cat in NEEDS_CATEGORIES:
            return "Needs"
        elif cat in SAVINGS_CATEGORIES:
            return "Savings"
        return "Wants"

    df_work["budget_bucket"] = df_work["category"].apply(classify_budget_bucket)
    bucket_totals = df_work.groupby("budget_bucket")["amount"].sum().to_dict()

    needs_spend = bucket_totals.get("Needs", 0.0)
    wants_spend = bucket_totals.get("Wants", 0.0)
    savings_spend = bucket_totals.get("Savings", 0.0)

    needs_pct = (needs_spend / total_spend) * 100
    wants_pct = (wants_spend / total_spend) * 100
    savings_pct = (savings_spend / total_spend) * 100

    # Monthly average baseline
    n_months = max(1, df_work["date"].dt.to_period("M").nunique())
    avg_monthly_spend = total_spend / n_months

    recommendations: List[Dict[str, Any]] = []
    total_potential_monthly_savings = 0.0

    # 2. Check Wants vs Needs
    if wants_pct > 35:
        excess_wants_monthly = ((wants_pct - 30) / 100 * avg_monthly_spend)
        total_potential_monthly_savings += max(0, excess_wants_monthly * 0.5)
        recommendations.append({
            "title": "Discretionary Spending ('Wants') Alert",
            "type": "warning",
            "category": "Budgeting",
            "impact": f"High (~₹{round(excess_wants_monthly * 0.5, 0):,.0f}/month potential savings)",
            "details": f"Your discretionary expenses (Dining, Shopping, Entertainment) account for {round(wants_pct, 1)}% of total spend (ideal is ≤ 30%). Trimming non-essential impulse orders could free up significant cash."
        })

    # 3. Check Food & Dining / Delivery Leakage
    dining_df = df_work[df_work["category"] == "Food & Dining"]
    dining_total = dining_df["amount"].sum()
    dining_pct = (dining_total / total_spend) * 100
    dining_monthly = dining_total / n_months

    if dining_pct > 18:
        potential_save = dining_monthly * 0.25
        total_potential_monthly_savings += potential_save
        recommendations.append({
            "title": "Food Delivery & Dining Optimization",
            "type": "opportunity",
            "category": "Food & Dining",
            "impact": f"Save ~₹{round(potential_save, 0):,.0f}/month",
            "details": f"Dining out & food delivery averages ₹{round(dining_monthly, 0):,.0f} per month ({round(dining_pct, 1)}% of expenses). Preparing meals at home 2 more days per week can save approximately 25% of this budget."
        })

    # 4. Check Subscriptions / Recurring Charges
    sub_df = df_work[df_work["category"] == "Entertainment & Subscriptions"]
    sub_total = sub_df["amount"].sum()
    sub_monthly = sub_total / n_months
    if sub_monthly > 1500:
        potential_save = sub_monthly * 0.30
        total_potential_monthly_savings += potential_save
        recommendations.append({
            "title": "Subscription & OTT Service Audit",
            "type": "tip",
            "category": "Entertainment",
            "impact": f"Save ~₹{round(potential_save, 0):,.0f}/month",
            "details": f"You spend ~₹{round(sub_monthly, 0):,.0f} monthly on entertainment/subscriptions. Audit unused streaming platforms, gym plans, or consider annual combo discounts."
        })

    # 5. Weekend vs Weekday Spending Analysis
    df_work["is_weekend"] = df_work["date"].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    weekend_df = df_work[df_work["is_weekend"] == 1]
    weekday_df = df_work[df_work["is_weekend"] == 0]

    weekend_daily = weekend_df["amount"].sum() / max(1, (len(weekend_df["date"].dt.date.unique())))
    weekday_daily = weekday_df["amount"].sum() / max(1, (len(weekday_df["date"].dt.date.unique())))

    if weekend_daily > 1.8 * weekday_daily:
        recommendations.append({
            "title": "Weekend Spending Surge",
            "type": "info",
            "category": "Behavioral",
            "impact": "Lifestyle adjustment",
            "details": f"Your average daily spending on weekends (₹{round(weekend_daily, 0):,.0f}/day) is {round(weekend_daily / max(1, weekday_daily), 1)}x higher than weekdays. Setting a strict weekend entertainment budget can curb spontaneous splurging."
        })

    # 6. Savings & Investments Rate Check
    if savings_pct < 20:
        recommendations.append({
            "title": "Boost Long-Term Savings & SIPs",
            "type": "action",
            "category": "Investments",
            "impact": f"Target: Increase to 20% (Currently {round(savings_pct, 1)}%)",
            "details": f"Your recorded investments/savings represent {round(savings_pct, 1)}% of your expenses. Aim to automate a 10% increase in your monthly SIP on payday before discretionary spending occurs ('Pay yourself first')."
        })
    else:
        recommendations.append({
            "title": "Solid Savings Discipline",
            "type": "positive",
            "category": "Investments",
            "impact": f"Great Job! ({round(savings_pct, 1)}% allocated to savings/investments)",
            "details": "You are consistently committing over 20% of your outflow toward investments and wealth building. Maintain this disciplined habit."
        })

    # Default general recommendation if list is short
    if len(recommendations) < 3:
        recommendations.append({
            "title": "Annual Utility & Mobile Plan Audit",
            "type": "tip",
            "category": "Utilities",
            "impact": "Save ~₹300 - ₹600/month",
            "details": "Compare postpaid vs prepaid plans for mobile and fiber broadband. Bundling OTT benefits with existing broadband often eliminates duplicate subscriptions."
        })

    return {
        "success": True,
        "total_spend": round(total_spend, 2),
        "avg_monthly_spend": round(avg_monthly_spend, 2),
        "potential_monthly_savings": round(total_potential_monthly_savings, 2),
        "budget_breakdown": {
            "Needs": {"amount": round(needs_spend, 2), "percentage": round(needs_pct, 1), "recommended_pct": 50},
            "Wants": {"amount": round(wants_spend, 2), "percentage": round(wants_pct, 1), "recommended_pct": 30},
            "Savings": {"amount": round(savings_spend, 2), "percentage": round(savings_pct, 1), "recommended_pct": 20}
        },
        "recommendations": recommendations
    }
