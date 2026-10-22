"""
src/database.py
SQLite database manager for persistent expense records.
Supports CRUD operations, CSV import/export, and sample dataset seeding.
"""

import os
import sqlite3
from typing import List, Optional, Tuple, Union
import pandas as pd

DEFAULT_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "expenses.db")


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Get SQLite database connection with row factory enabled."""
    path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database tables and indexes."""
    path = db_path or DEFAULT_DB_PATH
    conn = get_connection(path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL CHECK(amount >= 0),
                payment_method TEXT NOT NULL DEFAULT 'UPI',
                notes TEXT DEFAULT '',
                is_sample INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_expenses_date ON expenses(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_expenses_category ON expenses(category)")
    conn.close()


def add_expense(
    date: str,
    description: str,
    category: str,
    amount: float,
    payment_method: str = "UPI",
    notes: str = "",
    is_sample: int = 0,
    db_path: Optional[str] = None
) -> int:
    """Insert a new expense record and return its new ID."""
    conn = get_connection(db_path)
    with conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO expenses (date, description, category, amount, payment_method, notes, is_sample)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (str(date).strip(), str(description).strip(), str(category).strip(),
             float(amount), str(payment_method).strip(), str(notes or "").strip(), int(is_sample))
        )
        new_id = cursor.lastrowid
    conn.close()
    return new_id


def update_expense(
    expense_id: int,
    date: str,
    description: str,
    category: str,
    amount: float,
    payment_method: str,
    notes: str = "",
    db_path: Optional[str] = None
) -> bool:
    """Update an existing expense by ID."""
    conn = get_connection(db_path)
    with conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE expenses
            SET date = ?, description = ?, category = ?, amount = ?, payment_method = ?, notes = ?
            WHERE id = ?
            """,
            (str(date).strip(), str(description).strip(), str(category).strip(),
             float(amount), str(payment_method).strip(), str(notes or "").strip(), int(expense_id))
        )
        updated = cursor.rowcount > 0
    conn.close()
    return updated


def delete_expense(expense_id: int, db_path: Optional[str] = None) -> bool:
    """Delete an expense record by ID."""
    conn = get_connection(db_path)
    with conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM expenses WHERE id = ?", (int(expense_id),))
        deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def clear_all_expenses(db_path: Optional[str] = None) -> int:
    """Delete all expense records from the database."""
    conn = get_connection(db_path)
    with conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM expenses")
        deleted_count = cursor.rowcount
    conn.close()
    return deleted_count


def get_all_expenses(db_path: Optional[str] = None, as_df: bool = True) -> Union[pd.DataFrame, List[dict]]:
    """Retrieve all expenses ordered chronologically."""
    init_db(db_path)
    conn = get_connection(db_path)
    query = """
        SELECT id, date, description, category, amount, payment_method, notes, is_sample, created_at
        FROM expenses
        ORDER BY date DESC, id DESC
    """
    if as_df:
        df = pd.read_sql_query(query, conn)
        conn.close()
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        return df
    else:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows


def import_from_csv(
    file_or_path: Union[str, any],
    db_path: Optional[str] = None,
    is_sample: int = 0
) -> Tuple[int, List[str]]:
    """
    Import transactions from CSV with flexible column mapping and robust validation.
    Returns (imported_count, list_of_warning_messages).
    """
    init_db(db_path)
    warnings = []
    try:
        df = pd.read_csv(file_or_path)
    except Exception as e:
        return 0, [f"Failed to read CSV: {str(e)}"]

    if df.empty:
        return 0, ["The CSV file is empty."]

    # Normalize column names for flexible matching
    col_map = {}
    for col in df.columns:
        norm = col.strip().lower().replace("_", " ")
        if norm in ["date", "txn date", "transaction date", "time"]:
            col_map[col] = "date"
        elif norm in ["description", "desc", "title", "particulars", "merchant", "item", "details"]:
            col_map[col] = "description"
        elif norm in ["category", "cat", "expense type", "type"]:
            col_map[col] = "category"
        elif norm in ["amount", "amt", "cost", "price", "spent", "expenditure", "debit"]:
            col_map[col] = "amount"
        elif norm in ["payment method", "payment mode", "mode", "method"]:
            col_map[col] = "payment_method"
        elif norm in ["notes", "note", "remark", "remarks", "comment"]:
            col_map[col] = "notes"

    df = df.rename(columns=col_map)
    required = ["date", "description", "amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return 0, [f"Missing required columns in CSV: {', '.join(missing)}. Available columns: {', '.join(df.columns)}"]

    # Provide defaults for optional columns
    if "category" not in df.columns:
        df["category"] = "Miscellaneous"
    if "payment_method" not in df.columns:
        df["payment_method"] = "UPI"
    if "notes" not in df.columns:
        df["notes"] = ""

    # Parse and validate dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    invalid_dates = df["date"].isna().sum()
    if invalid_dates > 0:
        warnings.append(f"Dropped {invalid_dates} rows with invalid date formats.")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Clean description and amount
    df["description"] = df["description"].fillna("Unknown Expense").astype(str).str.strip()
    df["category"] = df["category"].fillna("Miscellaneous").astype(str).str.strip()
    df["category"] = df["category"].replace("", "Miscellaneous")
    df["payment_method"] = df["payment_method"].fillna("UPI").astype(str).str.strip()
    df["payment_method"] = df["payment_method"].replace("", "UPI")
    df["notes"] = df["notes"].fillna("").astype(str).str.strip()

    # Clean amount (remove currency signs, commas)
    if df["amount"].dtype == object:
        df["amount"] = df["amount"].astype(str).str.replace(r"[^\d.]", "", regex=True)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    invalid_amounts = df["amount"].isna().sum()
    if invalid_amounts > 0:
        warnings.append(f"Dropped {invalid_amounts} rows with non-numeric amounts.")
    df = df.dropna(subset=["amount"])
    df = df[df["amount"] >= 0]

    if df.empty:
        return 0, warnings + ["No valid records remained after data sanitization."]

    # Bulk insert
    conn = get_connection(db_path)
    records = []
    for _, row in df.iterrows():
        records.append((
            row["date"],
            row["description"],
            row["category"],
            float(row["amount"]),
            row["payment_method"],
            row["notes"],
            int(is_sample)
        ))

    with conn:
        conn.executemany(
            """
            INSERT INTO expenses (date, description, category, amount, payment_method, notes, is_sample)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            records
        )
    conn.close()

    return len(records), warnings


def load_sample_data(sample_csv_path: str, db_path: Optional[str] = None, reset: bool = True) -> Tuple[int, List[str]]:
    """Loads default sample CSV into SQLite database."""
    if reset:
        clear_all_expenses(db_path)
    return import_from_csv(sample_csv_path, db_path=db_path, is_sample=1)
