# src/modules/vector_store.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

def get_pinecone_client():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("❌ PINECONE_API_KEY manquante dans .env")
    return Pinecone(api_key=api_key)

def ensure_index_exists(pc, index_name="mental-health-raft", dimension=384):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-1")
        )
        print(f"🆕 Index '{index_name}' créé en {dimension}D.")
    else:
        print(f"📥 Index '{index_name}' existe déjà.")

def upsert_chunks_to_pinecone(index, chunks, embeddings, batch_size=100):
    print("📤 Insertion dans Pinecone...")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        vectors = [
            (str(i + j), emb, {"text": batch[j].page_content})
            for j, emb in enumerate(batch_embeddings)
        ]
        index.upsert(vectors=vectors)
    print(f"✅ {len(chunks)} chunks indexés.")