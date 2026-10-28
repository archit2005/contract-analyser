from app.rag import build_law_library, index_contract, answer
from app.parser import split_clauses

build_law_library()                                   # runs once; reuses it after that

text = open("data/sample_agreement.txt").read()
clauses = index_contract(split_clauses(text), doc_id="sample")
print(f"Indexed {len(clauses)} clauses\n")

questions = [
    "Can I leave the flat after 6 months?",
    "Is a six month security deposit allowed?",
    "Who pays if the roof leaks?",
    "Can the landlord ask me to leave with one week notice?",
    "What is the capital of France?",                 # should be refused: not in any document
]

for q in questions:
    reply, contract_hits, law_hits = answer(q, doc_id="sample")
    print("Q:", q)
    print("Matches:", [h["score"] for h in contract_hits], [h["score"] for h in law_hits])
    print(reply)
    print("-" * 60)

print("\n========== OFFER LETTER TEST ==========\n")
offer = open("data/sample_offer_letter.txt").read()
offer_clauses = index_contract(split_clauses(offer), doc_id="offer")
print(f"Indexed {len(offer_clauses)} clauses\n")

offer_questions = [
    "Do I have to pay anything if I leave after 1 year?",
    "Can I join a competitor after I resign?",
    "Is my notice period fair?",
]

for q in offer_questions:
    reply, c_hits, l_hits = answer(q, doc_id="offer")
    print("Q:", q)
    print("Matches:", [h["score"] for h in c_hits], [h["score"] for h in l_hits])
    print(reply)
    print("-" * 60)