import uuid

import streamlit as st
from langsmith import Client

from bot import PROMPT_VERSION, answer

ls = Client()

st.set_page_config(page_title="Acme Support Bot", page_icon="💬")
st.title("Acme Support Bot")
st.caption(f"Prompt version: {PROMPT_VERSION} · every answer is traced in LangSmith")

with st.sidebar:
    st.link_button("Open LangSmith", "https://smith.langchain.com")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


def send_feedback(run_id, key):
    # Attaches a thumbs up/down score to the exact trace in LangSmith
    value = st.session_state.get(key)  # 1 = thumbs up, 0 = thumbs down
    if value is not None:
        ls.create_feedback(run_id, key="user_score", score=value)


prompt = st.chat_input("Ask a support question")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    run_id = uuid.uuid4()
    with st.spinner("Thinking..."):
        result = answer(prompt, langsmith_extra={"run_id": run_id})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "category": result["category"],
            "doc_ids": result["doc_ids"],
            "run_id": run_id,
        }
    )

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            sources = ", ".join(m["doc_ids"]) or "none"
            st.caption(f"category: {m['category']} · sources: {sources}")
            key = f"fb_{m['run_id']}"
            st.feedback("thumbs", key=key, on_change=send_feedback, args=(m["run_id"], key))
