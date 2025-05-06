import os
import chromadb
from sentence_transformers import SentenceTransformer
import logging
import gc
import time

# Get logger for this module
logger = logging.getLogger(__name__)

# Set path to existing ChromaDB - use environment variable or default
CHROMA_PATH = os.environ.get("CHROMA_PATH", "data/chroma_db")
logger.info(f"Using ChromaDB from: {CHROMA_PATH}")

# Verify database exists
if not os.path.exists(CHROMA_PATH):
    logger.error(f"ChromaDB not found at {CHROMA_PATH}")
    raise FileNotFoundError(f"ChromaDB not found at {CHROMA_PATH}. Run build_chroma_db.py first.")

# Direct initialization of database only - not model
logger.info("Initializing ChromaDB client")
client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    logger.info("Loading SHL assessments collection")
    collection = client.get_collection("shl_assessments")
    logger.info("ChromaDB collection loaded successfully")
except Exception as e:
    logger.error(f"Failed to get ChromaDB collection: {e}")
    raise RuntimeError(f"Failed to get ChromaDB collection: {e}")

# The model is not pre-loaded to save memory
_model = None

def get_model():
    """Load model on demand and release after use to minimize memory footprint"""
    global _model
    if _model is None:
        logger.info("Loading sentence transformer model on demand")
        _model = SentenceTransformer("all-mpnet-base-v2")
        logger.info("Model loaded successfully")
    return _model

def release_model():
    """Release the model to free memory"""
    global _model
    _model = None
    gc.collect()
    logger.info("Model released from memory")

def chroma_search(query_text, top_k=10):
    """
    Perform vector search with extreme memory optimization.
    Loads model only when needed and releases it after use.
    """
    # Get model for this search only
    model = get_model()
    
    # Perform search
    logger.info(f"Generating embedding for query: {query_text[:50]}...")
    start_time = time.time()
    
    try:
        # Generate embedding with minimal options
        query_emb = model.encode(
            [query_text], 
            show_progress_bar=False,  # Disable progress bar
            convert_to_tensor=False   # Keep as numpy, don't convert to torch tensor
        )[0].tolist()
        
        # Release model immediately after use
        logger.info(f"Embedding generated in {time.time() - start_time:.2f}s")
        
        logger.info(f"Performing vector search with top_k={top_k}")
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["metadatas"]
        )
        
        num_results = len(results["metadatas"][0])
        logger.info(f"Search returned {num_results} results")
        
        # Free up memory after search
        release_model()
        
        return results["metadatas"][0]
    except Exception as e:
        logger.error(f"Search failed: {e}")
        release_model()  # Ensure model is released even on error
        raise