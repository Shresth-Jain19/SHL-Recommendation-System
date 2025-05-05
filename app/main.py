"""
SHL Assessment Recommendation System - FastAPI Backend
-----------------------------------------------------
Main application entry point that defines API endpoints for recommending
SHL assessments based on job descriptions or URLs using vector similarity search.

This module handles HTTP requests, CORS, and error management while delegating
the core recommendation logic to other modules.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import chromadb
from app.gemini_utils import get_query_from_url
from app.recommender import chroma_search


class RecommendRequest(BaseModel):
    """
    Request model for recommendation endpoint.
    
    Attributes:
        query (str): Job description text or URL
        is_url (bool, optional): Flag indicating if the query is a URL. Defaults to False.
    """
    query: str
    is_url: Optional[bool] = False

class AssessmentResponse(BaseModel):
    """
    Response model for assessment recommendations.
    
    Attributes:
        name (str): Assessment name
        url (str): URL to the assessment details page
        remote_testing (str): Whether remote testing is supported
        adaptive_irt_support (str): Whether adaptive/IRT is supported
        duration (str): Expected duration of the assessment
        test_type (str): Type of assessment (e.g., cognitive, personality)
    """
    name: str
    url: str
    remote_testing: str
    adaptive_irt_support: str
    duration: str
    test_type: str

app = FastAPI()

# BASE URL
@app.get("/")
def root():
    """Root endpoint to verify API is running."""
    return {"message": "SHL Recommendation API is running"}

# HEALTH CHECK
@app.get("/health")
def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return {"status": "ok"}

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= CHROMA + GEMINI RECOMMENDER =================

# Global variables for lazy loading
_model = None
_collection = None

def get_model():
    """
    Lazily loads the sentence transformer model.
    Uses the smaller MiniLM model for better performance on constrained resources.
    
    Returns:
        SentenceTransformer: Loaded model for text embedding
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def get_collection():
    """
    Lazily loads the ChromaDB collection.
    
    Returns:
        Collection: ChromaDB collection containing assessment embeddings
    """
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path="data/chroma_db")
        _collection = client.get_collection("shl_assessments")
    return _collection

@app.post("/recommend")
def recommend_assessments(request: RecommendRequest, top_k: int = Query(10, ge=1, le=10)):
    """
    Endpoint to get SHL assessment recommendations based on job description or URL.
    
    Args:
        request (RecommendRequest): Request containing query text or URL
        top_k (int): Number of recommendations to return (1-10)
        
    Returns:
        List[AssessmentResponse]: List of recommended assessments
        
    Raises:
        HTTPException: If an error occurs during processing
    """
    try:
        # Process URL if provided
        if request.query.startswith("http"):
            prompt = get_query_from_url(request.query)
            query_text = prompt
        else:
            query_text = request.query
        
        # Get recommendations using vector search
        results = chroma_search(query_text, top_k=top_k)
        
        # Format response
        return [
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
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")