# === ChromaDB + MPNet search logic ===
import chromadb
from chromadb.config import Settings
import os
from sentence_transformers import SentenceTransformer

# Check if ChromaDB exists, handle error properly
chroma_path = os.path.join("data", "chroma_db")
if not os.path.exists(chroma_path):
    raise FileNotFoundError(f"ChromaDB not found at {chroma_path}. Run build_chroma_db.py first.")

# Only retrieval: do not create or delete collections here!
client = chromadb.PersistentClient(path=chroma_path)
try:
    collection = client.get_collection("shl_assessments")
    model = SentenceTransformer("all-mpnet-base-v2")
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    raise

def chroma_search(query_text, top_k=10):
    print(f"[DEBUG] Searching ChromaDB for: {query_text}")
    query_emb = model.encode([query_text])[0].tolist()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["metadatas", "documents"]
    )
    print(f"[DEBUG] ChromaDB returned {len(results['metadatas'][0])} results")
    if len(results['metadatas'][0]) > 0:
        print(f"[DEBUG] First result metadata: {results['metadatas'][0][0]}")
        print(f"[DEBUG] First result document: {results['documents'][0][0]}")
    else:
        print("[DEBUG] No results found in ChromaDB.")
    return results["metadatas"][0]
    


# === FAISS Indexing and Searching ===
# import faiss
# import numpy as np
# from typing import List, Tuple

# def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
#     """
#     Builds a FAISS index from the given embeddings.
#     """
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings.astype(np.float32))
#     return index

# def save_faiss_index(index: faiss.IndexFlatL2, path: str):
#     """
#     Saves the FAISS index to disk.
#     """
#     faiss.write_index(index, path)

# def load_faiss_index(path: str) -> faiss.IndexFlatL2:
#     """
#     Loads a FAISS index from disk.
#     """
#     return faiss.read_index(path)

# def search_faiss_index(
#     index: faiss.IndexFlatL2, 
#     query_embedding: np.ndarray, 
#     top_k: int = 10
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Searches the FAISS index for the top_k most similar embeddings.
#     Returns (distances, indices).
#     """
#     if query_embedding.ndim == 1:
#         query_embedding = query_embedding.reshape(1, -1)
#     distances, indices = index.search(query_embedding.astype(np.float32), top_k)
#     return distances[0], indices[0]