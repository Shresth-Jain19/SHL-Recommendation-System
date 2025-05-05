import chromadb
from sentence_transformers import SentenceTransformer
import json
import os
from pathlib import Path
from typing import List
import chromadb.errors

class ChromaEmbeddingFunction:
    def __init__(self):
        self._model = SentenceTransformer("all-mpnet-base-v2")

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(input)
        return [embedding.tolist() for embedding in embeddings]

def stringify(value):
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return value

def create_vector_db():
    chroma_path = os.path.join("data", "chroma_db")
    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chroma_path)

    json_path = os.path.join("data", "SHL_RAW.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find JSON file at {json_path}")

    print(f"✅ Found JSON data at: {json_path}")

    with open(json_path, "r") as f:
        assessment_catalog = json.load(f)
        if not isinstance(assessment_catalog, list):
            raise ValueError("JSON data should be a list of assessments")

    documents = []
    metadatas = []

    for i, item in enumerate(assessment_catalog):
        if not isinstance(item, dict):
            print(f"⚠️ Skipping invalid item at index {i}")
            continue

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

        documents.append(
            f"{item['name']}: {item['description']}: {item['url']}: {item['duration']}: {stringify(item['languages'])}: {item['job_level']}: {item['remote_testing']}: {item['adaptive/irt_support']}: {item['test_type']}"
        )
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

    try:
        chroma_client.delete_collection("shl_assessments")
        print("♻️ Replaced existing collection")
    except chromadb.errors.NotFoundError:
        print("ℹ️ Collection did not exist, skipping delete.")

    collection = chroma_client.get_or_create_collection(
        "shl_assessments", embedding_function=ChromaEmbeddingFunction()
    )

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