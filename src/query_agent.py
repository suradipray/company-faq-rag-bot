from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

rewrite_llm = OllamaLLM(
    model="llama3.2:3b",
    temperature=0
)

rewrite_prompt = PromptTemplate.from_template("""
Rewrite the user's question into a clear, standalone question.
Use conversation context if needed.

Conversation:
{chat_history}

User question:
{question}

Rewritten question:
""")

def rewrite_query(question: str, chat_history: str) -> str:
    return rewrite_llm.invoke(
        rewrite_prompt.format(
            question=question,
            chat_history=chat_history
        )
    )
