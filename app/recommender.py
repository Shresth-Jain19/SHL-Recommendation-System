import os
import chromadb
from sentence_transformers import SentenceTransformer

# Set path to existing ChromaDB - use environment variable or default
CHROMA_PATH = os.environ.get("CHROMA_PATH", "data/chroma_db")
print(f"[INFO] Using ChromaDB from: {CHROMA_PATH}")

# Verify database exists
if not os.path.exists(CHROMA_PATH):
    raise FileNotFoundError(f"ChromaDB not found at {CHROMA_PATH}. Run build_chroma_db.py first.")

# Global variables
_client = None
_collection = None
_model = None

def get_client():
    global _client
    if _client is None:
        # Only create client, don't create database
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client

def get_collection():
    global _collection
    if _collection is None:
        client = get_client()
        try:
            # Only get collection, don't create
            _collection = client.get_collection("shl_assessments")
        except Exception as e:
            raise RuntimeError(f"Failed to get ChromaDB collection: {e}")
    return _collection

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-mpnet-base-v2")
    return _model

def chroma_search(query_text, top_k=10):
    model = get_model()
    collection = get_collection()
    
    # Perform search
    query_emb = model.encode([query_text])[0].tolist()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["metadatas"]
    )
    
    return results["metadatas"][0]