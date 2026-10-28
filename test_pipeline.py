from app.pipeline import process_contract
from app.rag import build_law_library, answer

build_law_library()


def run_test(title, file, doc_id, questions):
    """Process one document with the full pipeline, print its risk report, then ask questions."""
    print(f"\n========== {title} ==========\n")
    text = open(file).read()
    report = process_contract(text, doc_id=doc_id)

    print("===== RISK REPORT =====")
    for r in report:
        flag = "⚠️ " if r["risk"] == "Risky" else "✅ "
        print(f"{flag}Clause {r['clause_no']} | {r['type']} | {r['risk']} ({r['confidence']:.0%})")

    risky = [r for r in report if r["risk"] == "Risky"]
    print(f"\n{len(risky)} risky clause(s) out of {len(report)}\n")

    print("===== QUESTIONS =====")
    for q in questions:
        reply, c_hits, l_hits = answer(q, doc_id=doc_id)
        print("Q:", q)
        print(reply)
        print("-" * 60)


run_test(
    "RENT AGREEMENT",
    "data/sample_agreement.txt",
    "sample",
    ["Can I leave the flat after 6 months?"],
)

run_test(
    "OFFER LETTER",
    "data/sample_offer_letter.txt",
    "offer",
    [
        "Do I have to pay anything if I leave after 1 year?",
        "Can I join a competitor after I resign?",
    ],
)
