# src/modules/embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings

def create_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

def generate_embeddings_for_chunks(embedding_model, chunks):
    # LangChain gère l'embedding via embed_documents
    contents = [doc.page_content for doc in chunks]
    return embedding_model.embed_documents(contents)