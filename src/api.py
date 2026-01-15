from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_chain import get_rag_chain

app = FastAPI(title="Company FAQ RAG API")

chain = get_rag_chain()

class QueryRequest(BaseModel):
    question: str
    chat_history: str = ""

@app.post("/ask")
def ask_question(req: QueryRequest):
    result = chain.invoke(
        {
            "question": req.question,
            "chat_history": req.chat_history
        }
    )

    return {
        "answer": result["answer"],
        "sources": [
            {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page")
            }
            for doc in result["sources"]
        ]
    }
