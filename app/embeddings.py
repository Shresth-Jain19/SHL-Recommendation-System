import os
from dotenv import load_dotenv
from google import genai
from typing import List
import numpy as np
from google.genai import types
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# === FETCH API KEY ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# === CONFIGURE GEMINI ===
client = genai.Client(api_key=GEMINI_API_KEY)

# === Gemini embedding code (commented out) ===
# def get_gemini_embedding(text: str) -> list:
#     response = client.models.embed_content(
#         model="text-embedding-004",
#         contents=text,
#         config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
#     )
#     return response.embeddings[0].values
#     # return np.array(response.embeddings[0].values).astype(np.float32)

# def get_gemini_embeddings(texts: List[str]) -> np.ndarray:
#     embeddings = []
#     for idx, text in enumerate(texts):
#         emb = get_gemini_embedding(text)
#         embeddings.append(emb)
#         if (idx + 1) % 10 == 0 or idx == len(texts) - 1:
#             print(f"Embedded {idx + 1}/{len(texts)}")
#     return np.array(embeddings)

# === New embedding code using sentence-transformers (MPNet) ===
model = SentenceTransformer("all-mpnet-base-v2")

def get_mpnet_embedding(text: str) -> np.ndarray:
    return model.encode([text])[0]

def get_mpnet_embeddings(texts: List[str]) -> np.ndarray:
    return model.encode(texts, show_progress_bar=True)