# src/modules/ingestion.py
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def verify_documents_exist(pdf_paths):
    for path in pdf_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"❌ Fichier manquant : {path}")

def load_documents_from_pdfs(pdf_paths):
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        documents.extend(docs)
    return documents