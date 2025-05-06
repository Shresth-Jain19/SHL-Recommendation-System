from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")

def get_mpnet_embedding(text: str) -> np.ndarray:
    return model.encode([text])[0]

def get_mpnet_embeddings(texts: List[str]) -> np.ndarray:
    return model.encode(texts, show_progress_bar=True)