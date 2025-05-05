"""
SHL Assessment Recommendation System - ChromaDB Builder
------------------------------------------------------
This script creates and populates a ChromaDB vector database with SHL assessment data.
The assessments are embedded using Sentence Transformers and stored for efficient
similarity search. This is a one-time setup process that must be run before
using the recommendation system.

Usage:
  python build_chroma_db.py

Requirements:
  - data/SHL_RAW.json must exist and contain a list of assessment objects
  - Sufficient storage space for the vector database (typically 100MB-1GB)
"""

import chromadb
from sentence_transformers import SentenceTransformer
import json
import os
from pathlib import Path
from typing import List
import chromadb.errors

class ChromaEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using Sentence Transformers.
    
    Uses the smaller MiniLM model for better performance on constrained resources
    while maintaining good semantic search quality.
    """
    def __init__(self):
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Convert text inputs to embeddings.
        
        Args:
            input: List of strings to embed
            
        Returns:
            List of embedding vectors (as lists of floats)
        """
        embeddings = self._model.encode(input)
        return [embedding.tolist() for embedding in embeddings]

def stringify(value):
    """
    Convert a value to a string representation.
    Handles lists by joining elements with commas.
    
    Args:
        value: The value to stringify (list or scalar)
        
    Returns:
        str: String representation of the value
    """
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return value

def create_vector_db():
    """
    Main function to create and populate the ChromaDB vector database.
    
    Steps:
    1. Initialize ChromaDB persistent client
    2. Load assessment data from JSON
    3. Process and validate assessment records
    4. Create or replace collection
    5. Add documents with embeddings in batches
    
    Raises:
        FileNotFoundError: If the JSON data file is not found
        ValueError: If JSON data is invalid or no valid assessments are found
    """
    # Setup ChromaDB path and ensure directory exists
    chroma_path = os.path.join("data", "chroma_db")
    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chroma_path)

    # Load and validate JSON data
    json_path = os.path.join("data", "SHL_RAW.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find JSON file at {json_path}")

    print(f"✅ Found JSON data at: {json_path}")

    with open(json_path, "r") as f:
        assessment_catalog = json.load(f)
        if not isinstance(assessment_catalog, list):
            raise ValueError("JSON data should be a list of assessments")

    # Prepare documents and metadata
    documents = []
    metadatas = []

    for i, item in enumerate(assessment_catalog):
        # Validate record structure
        if not isinstance(item, dict):
            print(f"⚠️ Skipping invalid item at index {i}")
            continue

        # Check for required fields
        required_fields = [
            "name",
            "url",
            "description",
            "duration",
            "languages",
            "job_level",
            "remote_testing",
            "adaptive/irt_support",
            "test_type",
        ]
        if not all(field in item for field in required_fields):
            print(f"⚠️ Skipping incomplete item at index {i}")
            continue

        # Create document text by concatenating fields for rich semantic search
        documents.append(
            f"{item['name']}: {item['description']}: {item['url']}: {item['duration']}: {stringify(item['languages'])}: {item['job_level']}: {item['remote_testing']}: {item['adaptive/irt_support']}: {item['test_type']}"
        )
        
        # Store structured metadata for each document
        metadatas.append(
            {
                "name": item["name"],
                "url": item["url"],
                "description": item["description"],
                "duration": item["duration"],
                "languages": stringify(item["languages"]),
                "job_level": item["job_level"],
                "remote_testing": item["remote_testing"],
                "adaptive/irt_support": item["adaptive/irt_support"],
                "test_type": item["test_type"],
            }
        )

    if not documents:
        raise ValueError("No valid assessments found in JSON data")

    # Create or replace collection
    try:
        chroma_client.delete_collection("shl_assessments")
        print("♻️ Replaced existing collection")
    except chromadb.errors.NotFoundError:
        print("ℹ️ Collection did not exist, skipping delete.")

    # Create collection with custom embedding function
    collection = chroma_client.get_or_create_collection(
        "shl_assessments", embedding_function=ChromaEmbeddingFunction()
    )

    # Add documents in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=[str(j) for j in range(i, batch_end)],
        )

    print(f"🚀 Success! Created vector DB with {len(documents)} assessments")
    print(f"📁 ChromaDB stored at: {chroma_path}")

if __name__ == "__main__":
    create_vector_db()