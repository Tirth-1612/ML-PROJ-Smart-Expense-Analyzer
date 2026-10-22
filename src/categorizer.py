"""
src/categorizer.py
Provides rule-based keyword categorization and ML-assisted fallback for expense descriptions.
"""

import re
from typing import Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd

# Standard predefined categories
CATEGORIES = [
    "Food & Dining",
    "Groceries",
    "Transportation",
    "Utilities & Bills",
    "Rent & Housing",
    "Entertainment & Subscriptions",
    "Shopping",
    "Health & Fitness",
    "Investments",
    "Miscellaneous",
]

# Keyword dictionary with priority rules
RULE_MAPPINGS = {
    "Food & Dining": [
        "swiggy", "zomato", "restaurant", "cafe", "coffee", "starbucks",
        "mcdonalds", "burger king", "dominos", "pizza", "biryani", "dhaba",
        "chai", "tea stall", "food court", "lunch", "dinner", "breakfast",
        "haldiram", "barbeque", "subway", "kfc"
    ],
    "Groceries": [
        "blinkit", "zepto", "instamart", "bigbasket", "dmart", "d-mart",
        "supermarket", "grocery", "vegetables", "fruits", "milk", "dairy",
        "ration", "kirana", "nature basket", "spencer"
    ],
    "Transportation": [
        "uber", "ola", "rapido", "metro", "auto", "petrol", "diesel",
        "hpcl", "bpcl", "indian oil", "fuel", "parking", "toll", "fastag",
        "flight", "irctc", "train ticket", "bus ticket", "redbus", "indigo"
    ],
    "Utilities & Bills": [
        "electricity", "bescom", "tneb", "mseb", "water bill", "gas cylinder",
        "piped gas", "airtel", "jio", "vi recharge", "broadband", "wifi",
        "tatasky", "dth", "recharge", "utility", "maintenance fee", "society"
    ],
    "Rent & Housing": [
        "rent", "house rent", "flat rent", "brokerage", "deposit", "landlord",
        "society maintenance", "home repair", "plumber", "electrician"
    ],
    "Entertainment & Subscriptions": [
        "netflix", "spotify", "prime video", "hotstar", "disney", "youtube",
        "movie", "pvr", "inox", "bookmyshow", "gaming", "steam", "playstation",
        "audible", "kindle", "concert", "amusement"
    ],
    "Shopping": [
        "amazon", "flipkart", "myntra", "ajio", "zara", "h&m", "uniqlo",
        "meesho", "clothing", "shoes", "electronics", "gadgets", "croma",
        "reliance digital", "tata cliq", "jewellery", "tanishq"
    ],
    "Health & Fitness": [
        "pharmacy", "apollo", "medplus", "1mg", "pharmeasy", "doctor",
        "hospital", "clinic", "lab test", "cult.fit", "gym", "fitness",
        "dental", "opticals", "lenskart", "vitamins", "protein"
    ],
    "Investments": [
        "sip", "mutual fund", "zerodha", "groww", "upstox", "ppf", "nps",
        "fixed deposit", "fd transfer", "stocks", "crypto", "lic premium", "insurance"
    ]
}


class ExpenseCategorizer:
    """
    Categorizes transactions using rules first, and falls back to a TF-IDF + Naive Bayes
    classifier trained on existing transaction records if available.
    """

    def __init__(self):
        self.ml_pipeline: Optional[Pipeline] = None
        self.is_trained: bool = False

    def predict_by_rules(self, text: str) -> Optional[str]:
        """Check description against keyword dictionaries."""
        if not text or not isinstance(text, str):
            return None
        clean_text = text.lower().strip()

        # Check exact and word-boundary matches
        for category, keywords in RULE_MAPPINGS.items():
            for kw in keywords:
                # Word boundary or substring if keyword is multi-word
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, clean_text) or kw in clean_text:
                    return category
        return None

    def train_ml_fallback(self, df: pd.DataFrame) -> bool:
        """
        Train ML classifier if enough labeled transactions exist (at least 15 rows
        and at least 2 distinct categories).
        """
        if df is None or len(df) < 15:
            self.is_trained = False
            return False

        valid_df = df.dropna(subset=["description", "category"]).copy()
        valid_df = valid_df[valid_df["description"].str.strip() != ""]
        valid_df = valid_df[valid_df["category"].isin(CATEGORIES)]

        if len(valid_df) < 15 or valid_df["category"].nunique() < 2:
            self.is_trained = False
            return False

        try:
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("clf", MultinomialNB(alpha=0.5))
            ])
            pipeline.fit(valid_df["description"], valid_df["category"])
            self.ml_pipeline = pipeline
            self.is_trained = True
            return True
        except Exception:
            self.is_trained = False
            return False

    def categorize(self, description: str) -> Tuple[str, str]:
        """
        Categorizes an expense description.
        Returns:
            (category_name, detection_method: 'Rule-Based' | 'ML-Inferred' | 'Default')
        """
        if not description or not description.strip():
            return "Miscellaneous", "Default"

        # 1. First attempt: Rule-based keyword matching
        rule_cat = self.predict_by_rules(description)
        if rule_cat:
            return rule_cat, "Rule-Based"

        # 2. Second attempt: ML classifier fallback if trained
        if self.is_trained and self.ml_pipeline is not None:
            try:
                pred = self.ml_pipeline.predict([description])[0]
                return pred, "ML-Inferred"
            except Exception:
                pass

        # 3. Fallback
        return "Miscellaneous", "Default"
