# setup_pinecone.py
import os
from pathlib import Path
from dotenv import load_dotenv 
import pinecone 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# Config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY manquante dans .env")

INDEX_NAME = "mental-health-raft"

# PDFs
pdf_paths = [
    "docs/EmotionalIntelligence.pdf",
    "docs/Managing-Stress-Principles-and-Strategies-for-Health-and-Wellbeing.pdf",
    "docs/the-social-skills-guidebook-fhc-dr-notes.pdf"
]

# Vérifier les fichiers
for path in pdf_paths:
    if not Path(path).exists():
        raise FileNotFoundError(f"❌ Fichier manquant : {path}")

# Charger et découper
print("📄 Chargement des PDFs...")
documents = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    docs = loader.load()
    documents.extend(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
print(f"✅ {len(texts)} chunks créés.")

# Embeddings 384D
print("🧠 Génération des embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Pinecone
print("📡 Connexion à Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Créer l'index (384D)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # ✅ Doit être 384
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-1")  # ✅ pas de 'environment'
    )
    print(f"🆕 Index '{INDEX_NAME}' créé en 384D.")
else:
    print(f"📥 Index '{INDEX_NAME}' existe déjà.")

# Upsert en lots
index = pc.Index(INDEX_NAME)
batch_size = 100
print("📤 Insertion dans Pinecone...")
for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    contents = [doc.page_content for doc in batch]
    embeds = embeddings.embed_documents(contents)
    vectors = [
        (str(i + j), emb, {"text": contents[j]})
        for j, emb in enumerate(embeds)
    ]
    index.upsert(vectors=vectors)

print(f"✅ {len(texts)} chunks indexés.")