from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import chromadb
from app.gemini_utils import get_query_from_url
from app.recommender import chroma_search


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

# BASE URL
@app.get("/")
def root():
    return {"message": "SHL Recommendation API is running"}

# HEALTH CHECK
@app.get("/health")
def health_check():
    return {"status": "ok"}

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://shl-assessment-recommendation-system-shresth-jn.streamlit.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= CHROMA + GEMINI RECOMMENDER =================

# Global variables for lazy loading
_model = None
_collection = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-mpnet-base-v2")
    return _model

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path="data/chroma_db")
        _collection = client.get_collection("shl_assessments")
    return _collection

@app.post("/recommend")
def recommend_assessments(request: RecommendRequest, top_k: int = Query(10, ge=1, le=10)):
    try:
        if request.query.startswith("http"):
            prompt = get_query_from_url(request.query)
            query_text = prompt
        else:
            query_text = request.query
        
        results = chroma_search(query_text, top_k=top_k)
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