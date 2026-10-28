# app/classifier.py - label clauses with the fine-tuned DistilBERT models

from transformers import pipeline

type_model = pipeline("text-classification", model="app/model_clause_type")
risk_model = pipeline("text-classification", model="app/model_risk")


def label_clauses(clauses):
    """Take a list of clause texts, return one label dict per clause (same order)."""
    types = type_model(clauses, truncation=True)
    risks = risk_model(clauses, truncation=True)
    return [
        {"type": t["label"], "risk": r["label"], "confidence": round(r["score"], 2)}
        for t, r in zip(types, risks)
    ]