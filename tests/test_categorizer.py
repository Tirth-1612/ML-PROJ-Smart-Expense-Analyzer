"""
tests/test_categorizer.py
Unit tests for rule-based and ML-assisted expense categorization.
"""

import pandas as pd
from src.categorizer import ExpenseCategorizer, CATEGORIES


def test_rule_based_categorization():
    categorizer = ExpenseCategorizer()

    test_cases = [
        ("Swiggy biryani delivery", "Food & Dining"),
        ("Zomato dinner bowl", "Food & Dining"),
        ("Uber cab ride", "Transportation"),
        ("HPCL petrol refill", "Transportation"),
        ("Blinkit grocery essentials", "Groceries"),
        ("Bescom Electricity bill", "Utilities & Bills"),
        ("Flat rent payment to landlord", "Rent & Housing"),
        ("Netflix monthly subscription", "Entertainment & Subscriptions"),
        ("Amazon summer sale purchase", "Shopping"),
        ("Apollo pharmacy fever tablets", "Health & Fitness"),
        ("SIP investment mutual fund", "Investments")
    ]

    for desc, expected_cat in test_cases:
        cat, method = categorizer.categorize(desc)
        assert cat == expected_cat, f"Failed for '{desc}': expected '{expected_cat}', got '{cat}'"
        assert method == "Rule-Based"


def test_unrecognized_defaults_to_miscellaneous():
    categorizer = ExpenseCategorizer()
    cat, method = categorizer.categorize("XYZ Random Nonexistent Token 12345")
    assert cat == "Miscellaneous"
    assert method == "Default"


def test_ml_fallback_categorization():
    categorizer = ExpenseCategorizer()

    # Create dummy training data
    data = [
        {"description": f"tasty roll meal {i}", "category": "Food & Dining"} for i in range(10)
    ] + [
        {"description": f"highway toll express {i}", "category": "Transportation"} for i in range(10)
    ]
    train_df = pd.DataFrame(data)

    trained = categorizer.train_ml_fallback(train_df)
    assert trained is True
    assert categorizer.is_trained is True

    # Check that a novel phrase with similar tokens gets categorized
    pred_cat, method = categorizer.categorize("delightful roll meal")
    assert method in ["Rule-Based", "ML-Inferred"]
    assert pred_cat in CATEGORIES
