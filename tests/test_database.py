"""
tests/test_database.py
Unit tests for database CRUD, schema handling, and CSV importing.
"""

import os
import tempfile
import pandas as pd
from src.database import (
    init_db,
    add_expense,
    get_all_expenses,
    update_expense,
    delete_expense,
    clear_all_expenses,
    import_from_csv
)


def test_database_crud():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        init_db(db_path)

        # 1. Add
        id1 = add_expense(
            date="2026-05-10",
            description="Chai Point Snacks",
            category="Food & Dining",
            amount=180.50,
            payment_method="UPI",
            notes="Evening tea",
            is_sample=0,
            db_path=db_path
        )
        assert id1 > 0

        # 2. Retrieve
        df = get_all_expenses(db_path=db_path, as_df=True)
        assert len(df) == 1
        assert df.iloc[0]["description"] == "Chai Point Snacks"
        assert df.iloc[0]["amount"] == 180.50

        # 3. Update
        updated = update_expense(
            expense_id=id1,
            date="2026-05-10",
            description="Chai Point Snacks with Cookies",
            category="Food & Dining",
            amount=220.00,
            payment_method="UPI",
            notes="Updated note",
            db_path=db_path
        )
        assert updated is True

        df_updated = get_all_expenses(db_path=db_path, as_df=True)
        assert df_updated.iloc[0]["description"] == "Chai Point Snacks with Cookies"
        assert df_updated.iloc[0]["amount"] == 220.00

        # 4. Delete
        deleted = delete_expense(expense_id=id1, db_path=db_path)
        assert deleted is True

        df_empty = get_all_expenses(db_path=db_path, as_df=True)
        assert len(df_empty) == 0

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_import_from_csv():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db_path = tmp_db.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as tmp_csv:
        tmp_csv.write("Date,Description,Category,Amount,Payment Method,Notes\n")
        tmp_csv.write("2026-04-01,House Rent,Rent & Housing,22000,Net Banking,Monthly rent\n")
        tmp_csv.write("2026-04-05,Swiggy Lunch,Food & Dining,350.50,UPI,Weekend treat\n")
        tmp_csv.write("invalid-date,Bad Row,Food & Dining,100,UPI,\n")
        csv_path = tmp_csv.name

    try:
        init_db(db_path)
        count, warnings = import_from_csv(csv_path, db_path=db_path, is_sample=0)
        assert count == 2
        assert len(warnings) > 0  # Warned about dropped invalid date

        df = get_all_expenses(db_path=db_path)
        assert len(df) == 2
        assert set(df["category"].tolist()) == {"Rent & Housing", "Food & Dining"}

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
        if os.path.exists(csv_path):
            os.remove(csv_path)
