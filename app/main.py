from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import os
import numpy as np

from app.data_loader import load_shl_data, prepare_text_for_embedding

from app.recommender import chroma_search
from app.gemini_utils import get_query_from_url  # If you have this utility



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

# Add this near your other endpoints
@app.get("/")
def root():
    return {"message": "SHL Recommendation API", "status": "online"}

@app.get("/health")
def health_check():
    return {"status": "ok"}


# ================= CHROMA + GEMINI RECOMMENDER =================

from app.gemini_utils import get_query_from_url
from app.recommender import chroma_search

@app.post("/recommend", response_model=List[AssessmentResponse])
def recommend_assessments(request: RecommendRequest, top_k: int = Query(10, ge=1, le=10)):
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


# ============== FAISS RECOMMENDER ==============

# from app.embeddings import get_gemini_embeddings
# from app.recommender import load_faiss_index, search_faiss_index

# Load data and FAISS index at startup
# DATA_PATH = "data/SHL_RAW.json"
# FAISS_INDEX_PATH = "data/faiss_index.bin"

# shl_data = load_shl_data(DATA_PATH)
# faiss_index = load_faiss_index(FAISS_INDEX_PATH)

# @app.post("/recommend", response_model=List[AssessmentResponse])
# def recommend_assessments(request: RecommendRequest, top_k: int = Query(10, ge=1, le=10)):
#     # If input is a URL and Gemini supports it, just pass the URL as the query
#     query_input = request.query
#     query_embedding = get_gemini_embeddings([query_input])[0]

#     # Search FAISS index
#     _, indices = search_faiss_index(faiss_index, np.array(query_embedding), top_k=top_k)

#     # Prepare response
#     results = []
#     for idx in indices:
#         assessment = shl_data[idx]
#         results.append(AssessmentResponse(
#             name=assessment.get("name", ""),
#             url=assessment.get("url", ""),
#             remote_testing=assessment.get("remote_testing", ""),
#             adaptive_irt_support=assessment.get("adaptive/irt_support", ""),
#             duration=assessment.get("duration", ""),
#             test_type=assessment.get("test_type", "")
#         ))
#     return results
