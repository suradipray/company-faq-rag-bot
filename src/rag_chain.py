from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

VECTOR_DB_PATH = "vectorstore"

def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    from src.query_agent import rewrite_query
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = OllamaLLM(
        model="llama3.2:3b",
        temperature=0
    )

    prompt = PromptTemplate.from_template("""
You are a company FAQ assistant.

Use the conversation history and the provided context to answer.
Answer ONLY using the context.
If the answer is not present, say: "I don't have that information."

Conversation history:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = RunnableParallel(
        {
            # ✅ retriever gets ONLY the question string
            "docs": lambda x: retriever.invoke(
                rewrite_query(x["question"], x["chat_history"])
            ),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        }
    ) | {
        "answer": (
            {
                "context": lambda x: format_docs(x["docs"]),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"]
            }
            | prompt
            | llm
        ),
        "sources": lambda x: x["docs"]
    }

    return chain
