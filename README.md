# Smart Expense Analyzer & Spending Behaviour Predictor 💰

A college-level Machine Learning and Data Science web application built with Python, Streamlit, and Scikit-Learn to track personal expenses, categorize spending, cluster spending behavior into behavioral personas, forecast future monthly expenditure, detect anomalous transactions, and deliver personalized saving recommendations.

---

## 📌 Project Overview
Many personal budgeting apps only display static sums and past charts without intelligent predictive insights. **Smart Expense Analyzer & Spending Behaviour Predictor** incorporates core machine learning techniques to provide actionable, future-looking personal financial management:
- **Auto-Categorization:** Rule-based keyword engine with an ML Naive Bayes classifier fallback.
- **Spending Behaviour Clustering:** K-Means clustering with feature scaling (`StandardScaler`) to profile spending habits into clear personas.
- **Expenditure Prediction:** Baseline Linear Regression and Random Forest Regressor using strict chronological train/test validation to forecast next month's total spending.
- **Anomaly & Fraud Detection:** Isolation Forest algorithm to detect unusual outliers and unexpected spending surges.
- **Saving Opportunities:** Data-driven 50/30/20 budget framework analysis and quantified monthly rupee savings.

All amounts and visualizations default to **Indian Rupees (₹)**.

---

## 🚀 Key Features

### 1. Interactive Dashboard
- Summary KPI metric cards: Total expenditure, recent calendar month spending, average ticket size, and highest spending category.
- Plotly interactive charts: Monthly spending trends, category breakdown donut chart, ranked horizontal spending bars, and payment method distribution.
- Date range filtering and sample data indicator.

### 2. Expense Management & Persistence
- SQLite database (`data/expenses.db`) with persistent storage across application restarts.
- Full CRUD operations: Add new expenses (with real-time auto-categorization preview), edit existing records, and delete records.
- Robust CSV import with flexible column mapping, type coercion, and validation of malformed rows.
- Full CSV export functionality.
- One-click reset to pre-seeded realistic sample dataset (spanning 12 months with 370+ transactions).

### 3. Spending Behaviour Analysis (K-Means Clustering)
- Feature engineering: Transaction amount, day of month, day of week, and weekend indicator.
- Normalization with `StandardScaler` to prevent high amounts from skewing temporal dimensions.
- Evaluation metrics: **Silhouette Score** and **Inertia** (with dynamic cluster slider $k=2 \dots 5$).
- Automatic persona labeling: e.g., *Daily Micro-Transactions*, *Weekend Splurges*, *Fixed Monthly Commitments*.

### 4. Expense Prediction (Time-Series Forecasting)
- Monthly time-series feature engineering: linear trend index, month seasonality, auto-regressive lags ($t-1, t-2$), and 3-month rolling averages.
- **Chronological Train/Test Split:** Evaluated on the most recent holdout month(s) without lookahead bias.
- Baseline **Linear Regression** vs **Random Forest Regressor**.
- Evaluated using **MAE (Mean Absolute Error)** and **RMSE (Root Mean Squared Error)**.
- Visual trajectory plot connecting historical spend with projected next-month outlay.

### 5. Anomaly Detection (Isolation Forest)
- Unsupervised isolation trees isolating rare and extreme data points.
- Multi-dimensional features including transaction amount and ratio relative to category median (distinguishing routine high rent from abnormal luxury purchases).
- Contamination sensitivity slider ($1\% \dots 10\%$).
- Anomaly severity scores ($0 \dots 100$) and plain-English diagnostic explanations.

### 6. Saving Recommendations
- Evaluates actual spending against the classic **50/30/20 budget framework** (Needs, Wants, Savings).
- Detects discretionary expenditure overages, high food delivery/dining frequencies, and subscription creep.
- Computes estimated monthly potential savings in ₹.

### 7. Educational & ML Evaluation Hub
- Beginner-friendly explanations of all algorithms, preprocessing methods, and evaluation metrics for academic evaluators and students.

---

## 🛠️ Technology Stack
- **Language:** Python 3.10+
- **Web UI:** Streamlit
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualizations:** Plotly Express & Plotly Graph Objects
- **Storage:** SQLite3 & CSV

---

## 📂 Project Architecture

```
NBA-Accreditation-Improvement/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies
├── data/
│   ├── sample_expenses.csv     # 12-month sample dataset (~370+ transactions)
│   └── expenses.db             # Persistent SQLite database
├── src/
│   ├── __init__.py
│   ├── database.py             # SQLite CRUD, validation, and CSV import/export
│   ├── categorizer.py          # Rule-based regex & Naive Bayes categorization
│   ├── clustering.py           # K-Means clustering, scaling & persona labeling
│   ├── predictor.py            # Linear Regression & Random Forest forecasting
│   ├── anomaly.py              # Isolation Forest anomaly detection engine
│   ├── recommendations.py     # 50/30/20 budget analysis & savings generator
│   └── visualizer.py           # Reusable Plotly chart builders
├── tests/
│   ├── __init__.py
│   ├── test_categorizer.py     # Categorizer tests
│   ├── test_database.py        # Database & CSV import tests
│   └── test_ml_models.py       # ML clustering, forecasting, anomaly tests
└── README.md
```

---

## 💻 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/dhruvlad1/NBA-Accreditation-Improvement.git
cd NBA-Accreditation-Improvement
```

### 2. Set up virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install required packages
```bash
python3 -m pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Unit Tests
Execute the automated test suite using `pytest`:
```bash
python3 -m pytest tests/ -v
```
All 9 unit tests validate database CRUD, CSV parsing, categorization rules, K-Means clustering, forecasting, and Isolation Forest models.

---

## ⚠️ Limitations & Future Work
- **Time-Series Length:** Meaningful chronological forecasting requires at least 3-6 months of historical transactions. For new accounts, the baseline Linear Regression gracefully activates until sufficient data accumulates.
- **Categorization Scope:** Keyword rules cover common Indian and international merchants; uncommon names default to *Miscellaneous* until trained by the user's transaction history.
- **Future Enhancements:** Bank statement PDF parsing, receipt OCR, and deep learning sequence models (LSTM / Prophet) for multi-year seasonal analysis.
