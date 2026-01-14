from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

VECTOR_DB_PATH = "vectorstore"

def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = OllamaLLM(
        model="llama3.1:8b",
        temperature=0
    )

    prompt = PromptTemplate.from_template("""
You are a company FAQ assistant.
Answer ONLY using the provided context.
If the answer is not present, say: "I don't have that information."

Context:
{context}

Question:
{question}

Answer:
""")

    # Runnable that formats docs into context
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = RunnableParallel(
        {
            "docs": retriever,
            "question": RunnablePassthrough()
        }
    ) | {
        "answer": (
            {
                "context": lambda x: format_docs(x["docs"]),
                "question": lambda x: x["question"]
            }
            | prompt
            | llm
        ),
        "sources": lambda x: x["docs"]
    }

    return chain
