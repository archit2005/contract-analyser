# Day 1 - Baseline model: TF-IDF + Logistic Regression
# Predicts two things for each clause: its TYPE (Rent, Deposit...) and its RISK (Normal / Risky)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# 1. Load the labeled clauses
df = pd.read_csv("data/clauses.csv")
print("Total clauses:", len(df))
print(df["clause_type"].value_counts(), "\n")

# 2. Train one model per label column
for label in ["clause_type", "risk"]:
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df[label],
        test_size=0.25,          # keep 25% aside to test on unseen clauses
        random_state=42,         # same split every run, so results are comparable
        stratify=df[label],      # keep the same mix of labels in train and test
    )

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2)),        # text -> numbers (words + word pairs)
        LogisticRegression(max_iter=1000),          # learns which words point to which label
    )
    model.fit(X_train, y_train)

    print(f"===== Results for: {label} =====")
    print(classification_report(y_test, model.predict(X_test), zero_division=0))

    # 3. Try it on a brand-new clause
    sample = "The landlord can keep the full deposit if the tenant leaves early."
    print("Sample clause:", sample)
    print("Prediction:", model.predict([sample])[0], "\n")
