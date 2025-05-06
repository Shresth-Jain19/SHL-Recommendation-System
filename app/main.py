from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import chromadb
from app.gemini_utils import get_query_from_url
from app.recommender import chroma_search, get_model, release_model
import logging
import traceback
import time
import psutil
from functools import lru_cache
import gc
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable parallelism to reduce memory usage
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def force_gc():
    """Force garbage collection"""
    gc.collect()
    logger.info("Memory cleanup performed")

class RecommendRequest(BaseModel):
    query: str
    is_url: Optional[bool] = False

class AssessmentResponse(BaseModel):
    name: str
    url: str
    remote_testing: str
    adaptive_irt_support: str
    duration: str
    test_type: str

app = FastAPI()

# Model warmup tracker
model_warmed_up = False

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "SHL Recommendation API is running"}

@app.get("/health")
async def health():
    logger.info("Health check endpoint accessed")
    # Force cleanup on health check
    force_gc()
    return {"status": "healthy"}

@app.get("/warmup")
async def warmup():
    """Endpoint to warm up the model by running a simple embedding."""
    global model_warmed_up
    
    if model_warmed_up:
        return {"status": "Model already warmed up"}
    
    try:
        start_time = time.time()
        logger.info("Running model warmup")
        
        # Get model (loads it if not loaded)
        model = get_model()
        
        # Generate a simple embedding
        _ = model.encode(["This is a warmup query"])
        
        # Mark as warmed up but release the model
        model_warmed_up = True
        duration = time.time() - start_time
        logger.info(f"Model warmup completed in {duration:.2f} seconds")
        
        # Release model after warmup to free memory
        release_model()
        force_gc()
        
        return {"status": "success", "duration_seconds": duration}
    except Exception as e:
        logger.error(f"Error during model warmup: {str(e)}")
        # Make sure model is released even on error
        release_model()
        return {"status": "error", "message": str(e)}

@app.get("/memory")
async def memory_usage():
    """Endpoint to monitor memory usage of the application."""
    # Force cleanup before checking memory
    force_gc()
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    # Get overall system memory
    system_memory = psutil.virtual_memory()
    
    return {
        "process_memory_mb": memory_info.rss / 1024 / 1024,
        "system_memory": {
            "total_mb": system_memory.total / 1024 / 1024,
            "available_mb": system_memory.available / 1024 / 1024,
            "used_percent": system_memory.percent
        }
    }

@app.get("/test-db")
async def test_db():
    """Simple endpoint to test ChromaDB access without using the model."""
    try:
        start_time = time.time()
        # Just get a single item to verify database access
        info = collection.get(limit=1)
        duration = time.time() - start_time
        return {
            "status": "ok", 
            "db_access_time_ms": duration * 1000,
            "item_count": len(info["ids"])
        }
    except Exception as e:
        logger.error(f"Error testing DB: {str(e)}")
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

# Use a smaller LRU cache to reduce memory overhead
@lru_cache(maxsize=5)
def cached_search(query_text, top_k):
    """Cache recent search results to improve performance on repeated queries."""
    return chroma_search(query_text, top_k)

@app.post("/recommend")
async def recommend_assessments(
    request: RecommendRequest, 
    top_k: int = Query(5, ge=1, le=10),  # Default to fewer results
    background_tasks: BackgroundTasks = None
):
    logger.info(f"Processing recommendation request: query_length={len(request.query)}, is_url={request.is_url}, top_k={top_k}")
    start_time = time.time()
    
    # Force cleanup before processing
    force_gc()
    
    try:
        # Process URL if provided
        if request.query.startswith("http"):
            logger.info(f"Processing URL: {request.query[:50]}...")
            prompt = get_query_from_url(request.query)
            query_text = prompt
        else:
            query_text = request.query
            logger.info(f"Processing text query: {query_text[:50]}...")

        # Set a timeout for the search to avoid 502 errors
        max_process_time = 20  # seconds (Render timeout is 30s)
        remaining_time = max_process_time - (time.time() - start_time)
        
        if remaining_time <= 0:
            logger.warning("Request processing taking too long, returning early failure")
            raise HTTPException(status_code=503, detail="Request is taking too long to process")
        
        # Check memory before search
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage before search: {memory_before:.2f} MB")
        
        # Get recommendations using vector search with caching
        results = cached_search(query_text, top_k)
        
        # Check memory after search
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage after search: {memory_after:.2f} MB, delta: {memory_after-memory_before:.2f} MB")
        
        # Force cleanup after search
        force_gc()
        
        logger.info(f"Found {len(results)} matching assessments")

        # Format response
        response = [
            AssessmentResponse(
                name=item.get("name", ""),
                url=item.get("url", ""),
                remote_testing=item.get("remote_testing", ""),
                adaptive_irt_support=item.get("adaptive/irt_support", ""),
                duration=item.get("duration", ""),
                test_type=item.get("test_type", "")
            )
            for item in results
        ]
        
        # Log timing information
        duration = time.time() - start_time
        logger.info(f"Request completed in {duration:.2f} seconds")
        
        # If this took a long time, log a warning
        if duration > 10:
            logger.warning(f"Request took {duration:.2f} seconds - approaching timeout limit")
        
        logger.info(f"Returning {len(response)} formatted assessment responses")
        return response
    except Exception as e:
        # Force cleanup on error
        force_gc()
        
        duration = time.time() - start_time
        error_msg = f"Internal server error: {str(e)}"
        logger.error(f"Error in recommend_assessments after {duration:.2f}s: {error_msg}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Only initialize chromadb client at startup, not the model
logger.info("Starting application, initializing database connection only")
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection("shl_assessments")
logger.info("Database initialization complete. Model will load on demand.")

# Force garbage collection at startup
force_gc()