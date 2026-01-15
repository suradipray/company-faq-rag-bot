import streamlit as st
import requests

API_URL = "http://localhost:8000/ask"

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Company FAQ Chatbot",
    page_icon="📘",
    layout="centered"
)

st.title("📘 Company FAQ Chatbot")
st.caption("Streamlit UI → FastAPI → Agentic RAG")

# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

def format_chat_history(messages):
    lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)

# -----------------------------
# Display history
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# Chat input
# -----------------------------
user_input = st.chat_input("Ask a question about the company...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    chat_history = format_chat_history(st.session_state.messages)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                API_URL,
                json={
                    "question": user_input,
                    "chat_history": chat_history
                },
                timeout=120
            )

            if response.status_code != 200:
                st.error("Backend error")
                st.stop()

            data = response.json()
            answer = data["answer"]
            sources = data["sources"]

            st.markdown(answer)

            if sources:
                st.markdown("**Sources:**")
                for src in sources:
                    page = src["page"]
                    if isinstance(page, int):
                        page += 1
                    st.markdown(f"- {src['source']} (page {page})")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
