from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

summary_llm = OllamaLLM(
    model="llama3.2:3b",  # fast & cheap
    temperature=0
)

summary_prompt = PromptTemplate.from_template("""
Summarize the following conversation briefly.
Keep only important facts and user intent.

Conversation:
{conversation}

Summary:
""")

def summarize_conversation(conversation_text: str) -> str:
    return summary_llm.invoke(
        summary_prompt.format(conversation=conversation_text)
    )
