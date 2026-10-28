# ui/streamlit_app.py - the website: upload, risk report, chat

import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="Contract Analyzer", page_icon="📄", layout="wide")
st.title("📄 Contract Analyzer")
st.caption("Upload a rent agreement or offer letter (text PDF). Risk is judged from the "
           "tenant's / employee's side. Labels are automatic and can miss risky clauses. "
           "This is general information, not legal advice.")

# Remember things between clicks
if "doc" not in st.session_state:
    st.session_state.doc = None
    st.session_state.chat = []

# ---------- 1. Upload ----------
uploaded = st.file_uploader("Upload your agreement", type=["pdf", "txt"])
if uploaded and st.button("Analyze", type="primary"):
    with st.spinner("Reading and checking every clause..."):
        res = requests.post(f"{API}/upload",
                            files={"file": (uploaded.name, uploaded.getvalue())},
                            timeout=300)
    if res.ok:
        st.session_state.doc = res.json()
        st.session_state.chat = []
    else:
        st.error(res.json().get("detail", "Something went wrong."))

doc = st.session_state.doc

# ---------- 2. Risk report ----------
if doc:
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Clauses found", doc["total"])
    c2.metric("Risky clauses", doc["risky"])
    c3.metric("Normal clauses", doc["total"] - doc["risky"])

    st.subheader("⚠️ Risky clauses")
    risky = [c for c in doc["clauses"] if c["risk"] == "Risky"]
    if not risky:
        st.success("No risky clauses were flagged.")
    for c in risky:
        st.error(f"**Clause {c['clause_no']} · {c['type']}** ({c['confidence']:.0%} confident)\n\n{c['text']}")

    with st.expander("See all clauses"):
        for c in doc["clauses"]:
            icon = "⚠️" if c["risk"] == "Risky" else "✅"
            unsure = " · *low confidence, please review*" if c["confidence"] < 0.7 else ""
            st.markdown(f"{icon} **Clause {c['clause_no']} · {c['type']} · {c['risk']}**{unsure}\n\n{c['text']}")
            st.markdown("---")

    # ---------- 3. Chat ----------
    st.divider()
    st.subheader("💬 Ask about your document")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['label']}** (match {s['score']:.2f})\n\n{s['text']}")

    question = st.chat_input("e.g. Can I leave before the lock-in ends?")
    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.spinner("Looking through your document and the law..."):
            res = requests.post(f"{API}/ask",
                                json={"doc_id": doc["doc_id"], "question": question},
                                timeout=300)
        if res.ok:
            data = res.json()
            st.session_state.chat.append({"role": "assistant",
                                          "content": data["answer"],
                                          "sources": data["sources"]})
        else:
            st.session_state.chat.append({"role": "assistant",
                                          "content": "Sorry, something went wrong."})
        st.rerun()