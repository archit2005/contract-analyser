# app/pipeline.py - runs the full process for one uploaded contract

from app.parser import split_clauses
from app.classifier import label_clauses
from app.rag import index_contract


def process_contract(text, doc_id):
    """Split -> label -> store. Returns one entry per clause for the risk report."""
    clauses = split_clauses(text)                    # 1. split into clauses
    labels = label_clauses(clauses)                  # 2. DistilBERT labels each clause
    index_contract(clauses, doc_id, labels)          # 3. store clauses + labels in ChromaDB

    report = []
    for i, (clause, lab) in enumerate(zip(clauses, labels)):
        report.append({"clause_no": i + 1, "text": clause, **lab})
    return report