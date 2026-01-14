from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

PDF_PATH = "data/company_faq.pdf"
VECTOR_DB_PATH = "vectorstore"

def ingest():
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    print("Storing vectors...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    print("Ingestion completed successfully.")

if __name__ == "__main__":
    ingest()
