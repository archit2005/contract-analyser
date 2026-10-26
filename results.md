# Model Results (188 clauses, 47 test clauses)

| Metric | Baseline (TF-IDF + LR) | DistilBERT |
|---|---|---|
| Clause type: accuracy | 0.83 | 0.91 |
| Clause type: macro F1 | 0.83 | 0.91 |
| Risk: accuracy | 0.85 | 0.83 |
| Risk: macro F1 | 0.84 | 0.82 |
| Risky recall | 0.70 | 0.75 |
| Risky precision | 0.93 | 0.83 |

## Notes
- DistilBERT clearly better on clause type.
- Risk results within one test clause of each other; DistilBERT chosen for higher risky recall and better handling of new wording.
- Example: "The landlord can keep the full deposit if the tenant leaves early" — baseline: Maintenance (wrong), DistilBERT: Deposit (correct).
- Known issue found: "The flat owner will fix any water leakage at his own cost" was flagged Risky; added owner-perspective clauses to fix.