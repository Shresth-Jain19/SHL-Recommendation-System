"""
SHL Assessment Recommendation System - Recommender Module
--------------------------------------------------------
Core recommendation engine that powers the semantic search functionality.
Handles ChromaDB interactions and vector similarity search to find the
most relevant SHL assessments based on natural language queries.

This module implements lazy loading patterns to conserve memory and
efficiently manages the connection to the vector database.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer

# Set path to existing ChromaDB - use environment variable or default
CHROMA_PATH = os.environ.get("CHROMA_PATH", "data/chroma_db")
print(f"[INFO] Using ChromaDB from: {CHROMA_PATH}")

# Verify database exists
if not os.path.exists(CHROMA_PATH):
    raise FileNotFoundError(f"ChromaDB not found at {CHROMA_PATH}. Run build_chroma_db.py first.")

# Global variables for lazy loading
_client = None
_collection = None
_model = None

def get_client():
    """
    Lazily initializes and returns the ChromaDB client.
    Uses a persistent client to access the pre-built database.
    
    Returns:
        PersistentClient: Initialized ChromaDB client
    """
    global _client
    if _client is None:
        # Only create client, don't create database
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client

def get_collection():
    """
    Lazily loads the assessment collection from ChromaDB.
    
    Returns:
        Collection: ChromaDB collection containing assessment embeddings
        
    Raises:
        RuntimeError: If the collection cannot be accessed
    """
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
    """
    Lazily loads the sentence transformer model for query embedding.
    Uses the smaller MiniLM model for better performance on constrained resources.
    
    Returns:
        SentenceTransformer: Loaded model for text embedding
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def chroma_search(query_text, top_k=10):
    """
    Performs vector similarity search to find relevant assessments.
    
    Args:
        query_text (str): The job description or processed URL content
        top_k (int): Number of results to return (default: 10)
        
    Returns:
        list: The top-k most relevant assessment metadata objects
        
    Notes:
        - Embeddings are generated on-the-fly for the query
        - Results are sorted by decreasing similarity score
    """
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