# scripts/setup_documents.py
from src.modules.ingestion import verify_documents_exist, load_documents_from_pdfs
from src.modules.chunking import split_documents_into_chunks
from src.modules.embeddings import create_embedding_model, generate_embeddings_for_chunks
from src.modules.vector_store import get_pinecone_client, ensure_index_exists, upsert_chunks_to_pinecone

# Configuration
INDEX_NAME = "mental-health-raft"
PDF_PATHS = [
    "docs/EmotionalIntelligence.pdf",
    "docs/Managing-Stress-Principles-and-Strategies-for-Health-and-Wellbeing.pdf",
    "docs/the-social-skills-guidebook-fhc-dr-notes.pdf"
]

if __name__ == "__main__":
    # 1. Vérifier les fichiers
    verify_documents_exist(PDF_PATHS)
    
    # 2. Charger les documents
    print("📄 Chargement des PDFs...")
    documents = load_documents_from_pdfs(PDF_PATHS)
    
    # 3. Découper en chunks
    texts = split_documents_into_chunks(documents)
    print(f"✅ {len(texts)} chunks créés.")
    
    # 4. Préparer les embeddings
    print("🧠 Génération des embeddings...")
    embedding_model = create_embedding_model()
    embeddings = generate_embeddings_for_chunks(embedding_model, texts)
    
    # 5. Indexer dans Pinecone
    pc = get_pinecone_client()
    ensure_index_exists(pc, INDEX_NAME)
    index = pc.Index(INDEX_NAME)
    upsert_chunks_to_pinecone(index, texts, embeddings)