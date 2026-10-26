"""
Smart Expense Analyzer & Spending Behaviour Predictor
A comprehensive Streamlit web application for personal expense tracking,
behavioral clustering, time-series forecasting, and anomaly detection.
"""

import os
import io
from datetime import datetime, date
import pandas as pd
import streamlit as st

# Internal Modules
from src.database import (
    init_db,
    get_all_expenses,
    add_expense,
    update_expense,
    delete_expense,
    clear_all_expenses,
    import_from_csv,
    load_sample_data,
    DEFAULT_DB_PATH
)
from src.categorizer import ExpenseCategorizer, CATEGORIES
from src.clustering import run_spending_clustering
from src.predictor import predict_next_month
from src.anomaly import detect_anomalies
from src.recommendations import generate_saving_recommendations
from src.visualizer import (
    plot_monthly_trend,
    plot_category_breakdown,
    plot_category_horizontal_bars,
    plot_payment_methods,
    plot_clusters_scatter,
    plot_forecast_comparison,
    plot_anomalies_scatter,
    plot_budget_comparison
)

# App Configuration
st.set_page_config(
    page_title="Smart Expense Analyzer & Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern, Premium Financial UI
st.markdown("""
<style>
    /* Global Typography & Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        letter-spacing: -0.01em;
    }

    /* Streamlit Main Container Tweaks */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
        max-width: 95% !important;
    }

    /* Header Gradient Hero Banner */
    .hero-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #312E81 100%);
        border-radius: 16px;
        padding: 26px 32px;
        color: #FFFFFF;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3), 0 8px 10px -6px rgba(15, 23, 42, 0.2);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, rgba(99, 102, 241, 0) 70%);
        pointer-events: none;
    }
    .hero-header h1 {
        color: #F8FAFC !important;
        font-size: 26px !important;
        font-weight: 800 !important;
        margin: 0 0 6px 0 !important;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: -0.02em;
    }
    .hero-header p {
        color: #94A3B8 !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        margin: 0 !important;
        max-width: 800px;
        line-height: 1.5;
    }

    /* Metric Cards - Modern Glassmorphic Look */
    .metric-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 110px;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #4361EE, #4CC9F0);
    }
    .metric-title {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748B;
        font-weight: 700;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .metric-val {
        font-size: 23px;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
        font-feature-settings: "tnum" 1;
    }
    .metric-sub {
        font-size: 12px;
        color: #64748B;
        font-weight: 500;
    }

    /* Recommendation & Persona Cards */
    .rec-card {
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 16px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        position: relative;
        border-left: 5px solid #4361EE;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .rec-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.06);
    }
    .rec-card.warning {
        border-left-color: #F43F5E;
        background: linear-gradient(135deg, #FFF1F2 0%, #FFFFFF 100%);
    }
    .rec-card.opportunity {
        border-left-color: #0284C7;
        background: linear-gradient(135deg, #F0F9FF 0%, #FFFFFF 100%);
    }
    .rec-card.positive {
        border-left-color: #10B981;
        background: linear-gradient(135deg, #ECFDF5 0%, #FFFFFF 100%);
    }
    .rec-title {
        font-size: 16px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .rec-impact {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        background: #EEF2FF;
        color: #4361EE;
        margin-bottom: 8px;
    }
    .rec-card.warning .rec-impact {
        background: #FFE4E6;
        color: #E11D48;
    }
    .rec-card.positive .rec-impact {
        background: #D1FAE5;
        color: #059669;
    }

    /* Badges */
    .sample-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #DCFCE7;
        color: #166534;
        padding: 3px 8px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
    }
    .live-dot {
        width: 7px;
        height: 7px;
        background-color: #22C55E;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.2);
    }

    /* Sidebar Modern Premium Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%) !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 1.5rem 1rem !important;
    }
    /* Modern Radio Navigation Menu */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        gap: 6px;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 9px 14px !important;
        margin-bottom: 2px;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        display: flex;
        align-items: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background: #F8FAFC;
        border-color: #CBD5E1;
        transform: translateX(3px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.04);
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%) !important;
        border-color: #6366F1 !important;
        box-shadow: 0 2px 6px rgba(99, 102, 241, 0.15) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p {
        color: #3730A3 !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label p {
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: #334155 !important;
    }
    section[data-testid="stSidebar"] hr {
        margin: 1.2rem 0 !important;
        border-color: #E2E8F0 !important;
    }

    /* Modern Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F1F5F9;
        padding: 4px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        color: #64748B !important;
        border: none !important;
        background-color: transparent !important;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #4361EE !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08) !important;
    }

    /* Buttons Modernization */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database and Ensure Sample Data exists if empty
init_db()
sample_csv_path = os.path.join(os.path.dirname(__file__), "data", "sample_expenses.csv")
df_initial = get_all_expenses()
if df_initial.empty and os.path.exists(sample_csv_path):
    load_sample_data(sample_csv_path, reset=True)
    df_initial = get_all_expenses()

# Initialize Categorizer instance in Session State
if "categorizer" not in st.session_state:
    st.session_state.categorizer = ExpenseCategorizer()
    if not df_initial.empty:
        st.session_state.categorizer.train_ml_fallback(df_initial)


# Sidebar Navigation & Global Controls
st.sidebar.markdown("""
<div style="padding: 12px 6px 8px 6px;">
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
        <div style="width: 38px; height: 38px; border-radius: 10px; background: linear-gradient(135deg, #4361EE 0%, #3F37C9 100%); display: flex; align-items: center; justify-content: center; font-size: 20px; box-shadow: 0 4px 10px rgba(67, 97, 238, 0.35);">
            💰
        </div>
        <div>
            <div style="font-size: 18px; font-weight: 800; color: #0F172A; letter-spacing: -0.02em; line-height: 1.2;">Smart Expense <span style="background: linear-gradient(135deg, #4361EE, #7209B7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AI</span></div>
            <div style="display: flex; align-items: center; gap: 5px; margin-top: 2px;">
                <span class="live-dot"></span>
                <span style="font-size: 11px; font-weight: 600; color: #166534; text-transform: uppercase; letter-spacing: 0.5px;">Engine Active</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="font-size: 10.5px; font-weight: 800; text-transform: uppercase; color: #94A3B8; letter-spacing: 1px; margin: 12px 6px 6px 6px;">
    MAIN NAVIGATION
</div>
""", unsafe_allow_html=True)

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Dashboard",
        "💳 Expense Management",
        "🧠 Spending Behaviour (K-Means)",
        "📈 Expense Prediction (Forecasting)",
        "🚨 Anomaly Detection (Outliers)",
        "💡 Saving Recommendations",
        "🎓 Model Evaluation & ML Guide"
    ],
    index=0,
    label_visibility="collapsed"
)

st.sidebar.markdown("""
<div style="font-size: 10.5px; font-weight: 800; text-transform: uppercase; color: #94A3B8; letter-spacing: 1px; margin: 20px 6px 6px 6px;">
    DATASET & DATABASE
</div>
""", unsafe_allow_html=True)

# Dataset Status in Sidebar
df_current = get_all_expenses()
sample_count = int(df_current["is_sample"].sum()) if not df_current.empty and "is_sample" in df_current.columns else 0
user_count = len(df_current) - sample_count
total_records = len(df_current)

st.sidebar.markdown(f"""
<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px;">
        <span style="font-size: 12px; font-weight: 600; color: #64748B;">Total Transactions</span>
        <span style="font-size: 18px; font-weight: 800; color: #0F172A; font-feature-settings: 'tnum' 1;">{total_records:,}</span>
    </div>
    <div style="display: flex; gap: 8px;">
        <div style="flex: 1; background: #EEF2FF; border-radius: 8px; padding: 6px 8px; text-align: center;">
            <div style="font-size: 10px; font-weight: 700; color: #4361EE; text-transform: uppercase; letter-spacing: 0.5px;">Sample</div>
            <div style="font-size: 13px; font-weight: 700; color: #1E1B4B;">{sample_count}</div>
        </div>
        <div style="flex: 1; background: #ECFDF5; border-radius: 8px; padding: 6px 8px; text-align: center;">
            <div style="font-size: 10px; font-weight: 700; color: #059669; text-transform: uppercase; letter-spacing: 0.5px;">Custom</div>
            <div style="font-size: 13px; font-weight: 700; color: #064E3B;">{user_count}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="font-size: 10.5px; font-weight: 800; text-transform: uppercase; color: #94A3B8; letter-spacing: 1px; margin: 14px 6px 6px 6px;">
    QUICK ACTIONS
</div>
""", unsafe_allow_html=True)

# Quick Reset / Reload Sample Data
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    if st.button("🔄 Reset Data", use_container_width=True, help="Reload default 12-month sample dataset"):
        if os.path.exists(sample_csv_path):
            clear_all_expenses()
            load_sample_data(sample_csv_path, reset=True)
            st.session_state.categorizer.train_ml_fallback(get_all_expenses())
            st.sidebar.success("Reset successfully!")
            st.rerun()

with col_sb2:
    if st.button("🗑️ Clear All", use_container_width=True, help="Wipe database to start completely fresh"):
        clear_all_expenses()
        st.sidebar.warning("Cleared!")
        st.rerun()

st.sidebar.markdown("""
<div style="margin-top: 24px; padding: 10px 6px; text-align: center; border-top: 1px solid #E2E8F0;">
    <div style="font-size: 11px; color: #94A3B8; font-weight: 500;">
        Powered by Scikit-Learn & Streamlit
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. DASHBOARD
# ==============================================================================
if nav_choice == "📊 Dashboard":
    st.markdown("""
    <div class="hero-header">
        <h1>Smart Expense Analyzer & Dashboard</h1>
        <p>Comprehensive overview of your spending patterns, dynamic KPI metrics, and category distributions.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_current.empty:
        st.info("👋 No transactions found in the database. Use the button in the sidebar to load realistic sample data or add transactions in **Expense Management**.")
    else:
        # Date Filter Row
        df_current["parsed_date"] = pd.to_datetime(df_current["date"])
        min_date = df_current["parsed_date"].min().date()
        max_date = df_current["parsed_date"].max().date()

        col_f1, col_f2 = st.columns([1, 3])
        with col_f1:
            st.markdown("##### 📅 Timeframe Filter")
        with col_f2:
            date_range = st.date_input(
                "Filter Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                label_visibility="collapsed"
            )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_d, end_d = date_range
            filtered_df = df_current[
                (df_current["parsed_date"].dt.date >= start_d) &
                (df_current["parsed_date"].dt.date <= end_d)
            ].copy()
        else:
            filtered_df = df_current.copy()

        # Dynamic KPIs
        total_exp = filtered_df["amount"].sum()
        total_txns = len(filtered_df)
        avg_txn = total_exp / max(1, total_txns)
        top_cat = filtered_df.groupby("category")["amount"].sum().idxmax() if not filtered_df.empty else "N/A"
        top_cat_amt = filtered_df.groupby("category")["amount"].sum().max() if not filtered_df.empty else 0.0

        # Current calendar month spending
        latest_month = filtered_df["parsed_date"].dt.to_period("M").max()
        this_month_df = filtered_df[filtered_df["parsed_date"].dt.to_period("M") == latest_month]
        this_month_exp = this_month_df["amount"].sum()

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title"><span>💳</span> Total Spending</div>
                <div class="metric-val">₹{total_exp:,.2f}</div>
                <div class="metric-sub">{total_txns} total transactions</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title"><span>📅</span> Recent Month ({latest_month})</div>
                <div class="metric-val" style="color: #4361EE;">₹{this_month_exp:,.2f}</div>
                <div class="metric-sub">{len(this_month_df)} transactions</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title"><span>📊</span> Avg Ticket Size</div>
                <div class="metric-val">₹{avg_txn:,.2f}</div>
                <div class="metric-sub">Mean spend / txn</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title"><span>🏷️</span> Top Category</div>
                <div class="metric-val" style="font-size: 19px; word-break: break-word; color: #7209B7;">{top_cat}</div>
                <div class="metric-sub">₹{top_cat_amt:,.2f} cumulative</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi5:
            distinct_months = max(1, filtered_df["parsed_date"].dt.to_period("M").nunique())
            avg_monthly = total_exp / distinct_months
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title"><span>⚡</span> Monthly Average</div>
                <div class="metric-val">₹{avg_monthly:,.2f}</div>
                <div class="metric-sub">Across {distinct_months} month(s)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Main Charts
        chart_c1, chart_c2 = st.columns([3, 2])
        with chart_c1:
            st.plotly_chart(plot_monthly_trend(filtered_df), use_container_width=True)
        with chart_c2:
            st.plotly_chart(plot_category_breakdown(filtered_df), use_container_width=True)

        chart_c3, chart_c4 = st.columns([3, 2])
        with chart_c3:
            st.plotly_chart(plot_category_horizontal_bars(filtered_df), use_container_width=True)
        with chart_c4:
            st.plotly_chart(plot_payment_methods(filtered_df), use_container_width=True)

        # Recent Transactions Mini-Table
        st.markdown("### 🕒 Recent Transactions")
        display_cols = ["date", "description", "category", "amount", "payment_method", "notes"]
        recent_df = filtered_df[display_cols].head(10).copy()
        recent_df["amount"] = recent_df["amount"].apply(lambda v: f"₹{v:,.2f}")
        st.dataframe(recent_df, use_container_width=True)


# ==============================================================================
# 2. EXPENSE MANAGEMENT (CRUD, CSV IMPORT/EXPORT)
# ==============================================================================
elif nav_choice == "💳 Expense Management":
    st.markdown("""
    <div class="hero-header">
        <h1>Expense Management</h1>
        <p>Add, edit, delete, filter, and import/export expense transactions with intelligent auto-categorization.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_add, tab_view, tab_csv = st.tabs(["➕ Add / Edit Expense", "📋 View & Filter Expenses", "📂 CSV Import & Export"])

    # --- TAB 1: Add / Edit Transaction ---
    with tab_add:
        st.markdown("### Record a New Expense")
        with st.form("add_expense_form", clear_on_submit=False):
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                new_date = st.date_input("Transaction Date", value=date.today())
                new_desc = st.text_input("Expense Description", placeholder="e.g. Swiggy biryani, Uber ride, House rent")
                new_amount = st.number_input("Amount (₹)", min_value=0.0, step=10.0, format="%.2f")

            with col_a2:
                # Real-time auto-categorization preview
                suggested_cat, method_used = st.session_state.categorizer.categorize(new_desc)
                default_idx = CATEGORIES.index(suggested_cat) if suggested_cat in CATEGORIES else 0

                new_cat = st.selectbox(
                    f"Category (Suggested: {suggested_cat} via {method_used})",
                    options=CATEGORIES,
                    index=default_idx
                )
                new_method = st.selectbox("Payment Method", ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash"])
                new_notes = st.text_input("Notes (Optional)", placeholder="e.g. Shared with roommates")

            submitted = st.form_submit_button("💾 Save Expense", use_container_width=True)
            if submitted:
                if not new_desc.strip():
                    st.error("Please enter a valid expense description.")
                elif new_amount <= 0:
                    st.error("Amount must be greater than zero.")
                else:
                    new_id = add_expense(
                        date=str(new_date),
                        description=new_desc,
                        category=new_cat,
                        amount=new_amount,
                        payment_method=new_method,
                        notes=new_notes,
                        is_sample=0
                    )
                    # Retrain categorizer on new user data
                    st.session_state.categorizer.train_ml_fallback(get_all_expenses())
                    st.success(f"Expense successfully recorded! (ID #{new_id})")
                    st.rerun()

    # --- TAB 2: View, Search, Filter & Edit/Delete ---
    with tab_view:
        st.markdown("### Search & Filter Records")
        df_all = get_all_expenses()

        if df_all.empty:
            st.info("No expense records found.")
        else:
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                search_term = st.text_input("🔍 Search Description / Notes", placeholder="e.g. Swiggy, rent")
            with fc2:
                cat_filter = st.multiselect("Category Filter", options=CATEGORIES, default=[])
            with fc3:
                pm_options = sorted(df_all["payment_method"].dropna().unique().tolist())
                pm_filter = st.multiselect("Payment Method", options=pm_options, default=[])

            # Apply filters
            filtered = df_all.copy()
            if search_term.strip():
                term = search_term.lower()
                filtered = filtered[
                    filtered["description"].str.lower().str.contains(term, na=False) |
                    filtered["notes"].str.lower().str.contains(term, na=False)
                ]
            if cat_filter:
                filtered = filtered[filtered["category"].isin(cat_filter)]
            if pm_filter:
                filtered = filtered[filtered["payment_method"].isin(pm_filter)]

            st.write(f"Showing **{len(filtered)}** of **{len(df_all)}** transactions")

            # Table display
            display_df = filtered.copy()
            display_df["amount_formatted"] = display_df["amount"].apply(lambda x: f"₹{x:,.2f}")
            st.dataframe(
                display_df[["id", "date", "description", "category", "amount_formatted", "payment_method", "notes"]],
                use_container_width=True
            )

            # Edit / Delete Section
            st.markdown("---")
            st.markdown("#### ✏️ Modify or Delete Record")
            col_act1, col_act2 = st.columns([2, 3])
            with col_act1:
                record_id = st.number_input("Enter Transaction ID to Modify / Delete", min_value=1, step=1)
                action = st.radio("Action", ["Edit", "Delete"], horizontal=True)

            with col_act2:
                selected_row = df_all[df_all["id"] == record_id]
                if selected_row.empty:
                    st.caption("No transaction matching the entered ID.")
                else:
                    curr = selected_row.iloc[0]
                    if action == "Delete":
                        st.warning(f"Are you sure you want to delete **'{curr['description']}'** (₹{curr['amount']:,.2f}) on {curr['date']}?")
                        if st.button("🚨 Confirm Delete", type="primary"):
                            delete_expense(record_id)
                            st.success(f"Transaction #{record_id} deleted.")
                            st.rerun()
                    else:
                        with st.form(f"edit_form_{record_id}"):
                            e_date = st.date_input("Date", value=datetime.strptime(curr["date"], "%Y-%m-%d").date())
                            e_desc = st.text_input("Description", value=curr["description"])
                            e_amount = st.number_input("Amount (₹)", min_value=0.0, value=float(curr["amount"]), format="%.2f")
                            e_cat = st.selectbox("Category", options=CATEGORIES, index=CATEGORIES.index(curr["category"]) if curr["category"] in CATEGORIES else 0)
                            e_pm = st.selectbox("Payment Method", ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash"],
                                               index=["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash"].index(curr["payment_method"]) if curr["payment_method"] in ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash"] else 0)
                            e_notes = st.text_input("Notes", value=curr["notes"] or "")

                            if st.form_submit_button("Update Transaction"):
                                update_expense(record_id, str(e_date), e_desc, e_cat, e_amount, e_pm, e_notes)
                                st.success(f"Transaction #{record_id} updated.")
                                st.rerun()

    # --- TAB 3: CSV Import & Export ---
    with tab_csv:
        st.markdown("### Import Transactions from CSV")
        st.markdown("Upload any standard CSV statement or export. The parser automatically detects column names (`date`, `description`, `category`, `amount`, `payment_method`, `notes`).")

        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            if st.button("📥 Process & Import CSV", type="primary"):
                count, warnings = import_from_csv(uploaded_file, is_sample=0)
                if count > 0:
                    st.success(f"Successfully imported {count} transactions!")
                    st.session_state.categorizer.train_ml_fallback(get_all_expenses())
                if warnings:
                    for w in warnings:
                        st.warning(w)
                if count > 0:
                    st.rerun()

        st.markdown("---")
        st.markdown("### Export Current Database")
        df_export = get_all_expenses()
        if not df_export.empty:
            csv_buffer = io.StringIO()
            df_export.to_csv(csv_buffer, index=False)
            st.download_button(
                label="⬇️ Download All Transactions as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"smart_expenses_export_{date.today()}.csv",
                mime="text/csv",
                use_container_width=True
            )


# ==============================================================================
# 3. SPENDING BEHAVIOUR (CLUSTERING)
# ==============================================================================
elif nav_choice == "🧠 Spending Behaviour (K-Means)":
    st.markdown("""
    <div class="hero-header">
        <h1>Spending Behaviour & Pattern Clustering</h1>
        <p>Unsupervised Machine Learning (K-Means) groups your financial transactions into behavioral personas based on amount, frequency, and timing.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_current.empty or len(df_current) < 10:
        st.warning("⚠️ At least 10 transactions are required to perform meaningful behavioral clustering.")
    else:
        col_k1, col_k2 = st.columns([1, 3])
        with col_k1:
            k_val = st.slider("Select Number of Clusters (k)", min_value=2, max_value=5, value=3)

        cluster_results = run_spending_clustering(df_current, n_clusters=k_val)

        if not cluster_results["success"]:
            st.error(cluster_results["message"])
        else:
            # Score Metrics
            m_c1, m_c2, m_c3 = st.columns(3)
            with m_c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Clusters Formed</div>
                    <div class="metric-val">{cluster_results['k']} Personas</div>
                    <div class="metric-sub">K-Means with StandardScaler</div>
                </div>
                """, unsafe_allow_html=True)
            with m_c2:
                sil = cluster_results['silhouette']
                sil_desc = "Strong separation" if sil > 0.5 else ("Moderate structure" if sil > 0.25 else "Overlapping patterns")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Silhouette Score</div>
                    <div class="metric-val">{sil}</div>
                    <div class="metric-sub">{sil_desc}</div>
                </div>
                """, unsafe_allow_html=True)
            with m_c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Inertia (Sum of Squared Distances)</div>
                    <div class="metric-val">{cluster_results['inertia']}</div>
                    <div class="metric-sub">Compactness metric</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Scatter Plot Visualization
            st.plotly_chart(plot_clusters_scatter(cluster_results["df_clustered"]), use_container_width=True)

            # Cluster Personas Explanation Cards
            st.markdown("### 🏷️ Discovered Behavioral Personas")
            for profile in cluster_results["cluster_profiles"]:
                top_cats_str = ", ".join([f"{k} ({v})" for k, v in profile["top_categories"].items()])
                st.markdown(f"""
                <div class="rec-card opportunity">
                    <div class="rec-title">Cluster {profile['cluster_id']}: {profile['persona_name']}</div>
                    <div class="rec-impact">{profile['count']} Transactions ({profile['percentage']}% of total)</div>
                    <div style="font-size: 13px; color: #374151; margin-top: 6px;">
                        • <b>Average Transaction:</b> ₹{profile['mean_amount']:,.2f} (Median: ₹{profile['median_amount']:,.2f})<br>
                        • <b>Timing:</b> Typical Day of Month: ~{profile['avg_day_of_month']} | Weekend Activity: {profile['weekend_percentage']}%<br>
                        • <b>Frequent Categories:</b> {top_cats_str}
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# 4. EXPENSE PREDICTION (FORECASTING)
# ==============================================================================
elif nav_choice == "📈 Expense Prediction (Forecasting)":
    st.markdown("""
    <div class="hero-header">
        <h1>Monthly Expenditure Forecasting</h1>
        <p>Predict next month's total expenses using Linear Regression (baseline) and Random Forest Regressor with chronological train/test validation.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_current.empty:
        st.warning("⚠️ No data available for expense prediction.")
    else:
        pred_results = predict_next_month(df_current)

        if not pred_results["success"]:
            st.info(f"ℹ️ {pred_results['message']}")
            st.caption("Hint: Reset to the included sample dataset from the sidebar to experience 12 months of chronological predictive modeling.")
        else:
            # Prediction Hero Banner
            p_c1, p_c2, p_c3 = st.columns(3)
            with p_c1:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid #F72585;">
                    <div class="metric-title">Forecasted Next Month ({pred_results['next_month_label']})</div>
                    <div class="metric-val" style="color: #F72585;">₹{pred_results['primary_forecast']:,.2f}</div>
                    <div class="metric-sub">Recommended: {pred_results['best_model_name']}</div>
                </div>
                """, unsafe_allow_html=True)
            with p_c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Linear Regression (Baseline)</div>
                    <div class="metric-val">₹{pred_results['lr_forecast']:,.2f}</div>
                    <div class="metric-sub">MAE: ₹{pred_results['lr_mae']:,.2f} | RMSE: ₹{pred_results['lr_rmse']:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with p_c3:
                rf_str = f"₹{pred_results['rf_forecast']:,.2f}" if pred_results['rf_forecast'] is not None else "N/A"
                rf_sub = f"MAE: ₹{pred_results['rf_results']['mae']:,.2f} | RMSE: ₹{pred_results['rf_results']['rmse']:,.2f}" if pred_results['rf_results'] else "Needs ≥6 months"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Random Forest Regressor</div>
                    <div class="metric-val">{rf_str}</div>
                    <div class="metric-sub">{rf_sub}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Trajectory & Forecast Chart
            st.plotly_chart(
                plot_forecast_comparison(
                    pred_results["monthly_history"],
                    pred_results["eval_comparison"],
                    pred_results["next_month_label"],
                    pred_results["primary_forecast"],
                    pred_results["best_model_name"]
                ),
                use_container_width=True
            )

            # Chronological Evaluation Holdout Table
            st.markdown("### 📊 Chronological Holdout Evaluation")
            st.caption(f"Trained on earliest {pred_results['train_size']} months, evaluated on most recent {pred_results['test_size']} holdout month(s) to strictly prevent lookahead bias.")

            eval_df = pd.DataFrame(pred_results["eval_comparison"])
            rename_map = {
                "month": "Test Period",
                "actual": "Actual Expenditure (₹)",
                "linear_regression_pred": "Linear Reg Pred (₹)",
                "lr_error": "LR Abs Error (₹)",
                "random_forest_pred": "Random Forest Pred (₹)",
                "rf_error": "RF Abs Error (₹)"
            }
            eval_df = eval_df.rename(columns={k: v for k, v in rename_map.items() if k in eval_df.columns})
            st.dataframe(eval_df, use_container_width=True)


# ==============================================================================
# 5. ANOMALY DETECTION (ISOLATION FOREST)
# ==============================================================================
elif nav_choice == "🚨 Anomaly Detection (Outliers)":
    st.markdown("""
    <div class="hero-header">
        <h1>Unusual Expense & Anomaly Detection</h1>
        <p>Uses Isolation Forest unsupervised machine learning to isolate and flag irregular transactions, unexpected surges, and high-risk outflows.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_current.empty or len(df_current) < 15:
        st.warning("⚠️ At least 15 transactions are required to train the Isolation Forest model.")
    else:
        col_an1, col_an2 = st.columns([1, 3])
        with col_an1:
            contam = st.slider(
                "Anomaly Sensitivity (Contamination Rate)",
                min_value=0.01,
                max_value=0.10,
                value=0.03,
                step=0.01,
                help="Estimated fraction of outliers expected in the dataset."
            )

        anom_results = detect_anomalies(df_current, contamination=contam)

        if not anom_results["success"]:
            st.error(anom_results["message"])
        else:
            # Metric row
            a_c1, a_c2, a_c3 = st.columns(3)
            with a_c1:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid #EF476F;">
                    <div class="metric-title">Anomalies Detected</div>
                    <div class="metric-val" style="color: #EF476F;">{anom_results['anomaly_count']} Transactions</div>
                    <div class="metric-sub">{anom_results['anomaly_percentage']}% of total records</div>
                </div>
                """, unsafe_allow_html=True)
            with a_c2:
                total_anom_spend = anom_results["df_anomalies"]["amount"].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Anomaly Outflow Total</div>
                    <div class="metric-val">₹{total_anom_spend:,.2f}</div>
                    <div class="metric-sub">Sum of flagged purchases</div>
                </div>
                """, unsafe_allow_html=True)
            with a_c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Algorithm</div>
                    <div class="metric-val" style="font-size: 18px;">Isolation Forest</div>
                    <div class="metric-sub">100 Trees | Multidimensional</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Scatter Plot
            st.plotly_chart(plot_anomalies_scatter(anom_results["df_all"]), use_container_width=True)

            # Flagged Transactions Detail Table
            st.markdown("### ⚠️ Flagged Unusual Transactions")
            display_cols = ["date", "description", "category", "amount", "payment_method", "anomaly_severity", "anomaly_reason"]
            show_anom = anom_results["df_anomalies"][display_cols].copy()
            show_anom["amount"] = show_anom["amount"].apply(lambda v: f"₹{v:,.2f}")
            show_anom["anomaly_severity"] = show_anom["anomaly_severity"].apply(lambda s: f"{s} / 100")
            st.dataframe(show_anom, use_container_width=True)


# ==============================================================================
# 6. SAVING RECOMMENDATIONS
# ==============================================================================
elif nav_choice == "💡 Saving Recommendations":
    st.markdown("""
    <div class="hero-header">
        <h1>Personalized Saving Opportunities & Financial Health</h1>
        <p>Actionable, data-driven suggestions derived from your real spending habits, 50/30/20 budget adherence, and recurring leaks.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_current.empty:
        st.warning("⚠️ No transactions found to generate saving opportunities.")
    else:
        rec_results = generate_saving_recommendations(df_current)

        if not rec_results["success"]:
            st.error(rec_results["message"])
        else:
            # Summary Metrics
            s_c1, s_c2, s_c3 = st.columns(3)
            with s_c1:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid #06D6A0;">
                    <div class="metric-title">Estimated Monthly Savings Potential</div>
                    <div class="metric-val" style="color: #06D6A0;">₹{rec_results['potential_monthly_savings']:,.2f}</div>
                    <div class="metric-sub">Achievable through targeted optimizations</div>
                </div>
                """, unsafe_allow_html=True)
            with s_c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Average Monthly Outflow</div>
                    <div class="metric-val">₹{rec_results['avg_monthly_spend']:,.2f}</div>
                    <div class="metric-sub">Total: ₹{rec_results['total_spend']:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with s_c3:
                wants_p = rec_results["budget_breakdown"]["Wants"]["percentage"]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Discretionary Ratio ('Wants')</div>
                    <div class="metric-val">{wants_p}%</div>
                    <div class="metric-sub">Ideal benchmark: ≤ 30%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # 50/30/20 Chart
            st.plotly_chart(plot_budget_comparison(rec_results["budget_breakdown"]), use_container_width=True)

            # Recommendations List
            st.markdown("### 🎯 Tailored Recommendations")
            for rec in rec_results["recommendations"]:
                rec_type = rec.get("type", "opportunity")
                st.markdown(f"""
                <div class="rec-card {rec_type}">
                    <div class="rec-title">{rec['title']}</div>
                    <div class="rec-impact">{rec['impact']}</div>
                    <p style="margin: 6px 0 0 0; font-size: 14px; color: #374151;">{rec['details']}</p>
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# 7. MODEL EVALUATION & ML GUIDE
# ==============================================================================
elif nav_choice == "🎓 Model Evaluation & ML Guide":
    st.markdown("""
    <div class="hero-header">
        <h1>Machine Learning Evaluation & Algorithmic Foundation</h1>
        <p>In-depth explanations of the ML algorithms, performance metrics, and preprocessing pipelines powering this application.</p>
    </div>
    """, unsafe_allow_html=True)

    exp1, exp2, exp3, exp4 = st.tabs([
        "1. K-Means Clustering",
        "2. Time-Series Forecasting",
        "3. Isolation Forest Outliers",
        "4. Categorization Pipeline"
    ])

    with exp1:
        st.markdown("""
        ### 🧠 K-Means Clustering (Spending Behaviour)
        - **Objective:** Group transactions into distinct behavioral patterns without manual labels.
        - **Preprocessing:** All numerical features (`amount`, `day_of_month`, `day_of_week`, `is_weekend`) undergo **StandardScaler** z-score normalization ($z = \frac{x - \mu}{\sigma}$). Without scaling, high-magnitude transaction amounts would overpower day-of-month and day-of-week timing features.
        - **Evaluation Metrics:**
          - **Inertia:** Sum of squared distances of samples to their closest cluster center (lower = more compact).
          - **Silhouette Score:** Measures cohesion within clusters vs separation from other clusters (ranges from -1 to +1, where values > 0.35 indicate clear cluster separation).
        - **Persona Assignment:** Centroids are mapped into human-interpretable personas (e.g. *Daily Micro-Transactions*, *Weekend Splurges*, *Fixed Monthly Commitments*).
        """)

    with exp2:
        st.markdown("""
        ### 📈 Linear Regression & Random Forest (Forecasting)
        - **Objective:** Predict total expenditure for the upcoming calendar month ($t+1$).
        - **Feature Engineering:** Aggregates transactions into monthly time series and engineers:
          - Linear trend index ($t$)
          - Seasonal month index
          - Auto-regressive lag features ($y_{t-1}, y_{t-2}$)
          - 3-month rolling moving average
        - **Why Chronological Splits Matter:**
          Standard random k-fold cross validation causes **lookahead bias** in time series (training on future data to predict the past). We enforce strict chronological splits (e.g., train on months $1 \dots t-1$, evaluate on month $t$).
        - **Evaluation Metrics:**
          - **MAE (Mean Absolute Error):** $\\frac{1}{n} \\sum |y - \\hat{y}|$ (direct interpretation in ₹ rupees).
          - **RMSE (Root Mean Squared Error):** $\\sqrt{\\frac{1}{n} \\sum (y - \\hat{y})^2}$ (heavily penalizes large forecasting mistakes).
        """)

    with exp3:
        st.markdown("""
        ### 🚨 Isolation Forest (Anomaly Detection)
        - **Objective:** Flag unusual purchases, fraudulent charges, or extreme expenditure surges.
        - **How it Works:**
          Instead of modeling normal instances, Isolation Forest isolates anomalies by randomly partitioning feature space with decision trees.
          Outliers require significantly fewer splits to isolate than clustered normal points, resulting in shorter path lengths in the tree ensemble.
        - **Contextual Features:**
          In addition to absolute transaction amount, the model examines the ratio of the transaction amount relative to its category median, preventing routine high-value rent payments from masking high-value dining outliers.
        """)

    with exp4:
        st.markdown("""
        ### 🏷️ Intelligent Expense Categorizer
        - **Two-Tier Architecture:**
          1. **Tier 1 (Rule-Based Regex Engine):** Uses high-precision keyword patterns matching hundreds of common Indian merchants (Swiggy, Zomato, Blinkit, Bescom, Uber, Netflix, etc.).
          2. **Tier 2 (ML Classifier Fallback):** When user accounts accumulate labeled transactions, a **TF-IDF Vectorizer + Multinomial Naive Bayes** pipeline is dynamically trained on existing descriptions to infer categories for unlisted merchants.
        """)
