import streamlit as st
from rag_chain import get_rag_chain

st.set_page_config(
    page_title="Company FAQ Chatbot",
    page_icon="📘",
    layout="centered"
)

st.title("📘 Company FAQ Chatbot")
st.caption("Answers are grounded in company FAQ documents")

@st.cache_resource
def load_chain():
    return get_rag_chain()

chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question about the company...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chain.invoke(user_input)

            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)

            if sources:
                st.markdown("**Sources:**")
                for doc in sources:
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "N/A")
                    st.markdown(f"- {source} (page {page + 1})")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
