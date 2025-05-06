import chromadb
from sentence_transformers import SentenceTransformer
import json
import os
from pathlib import Path
from typing import List
import chromadb.errors
import logging
import gc
import time
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Force garbage collection
def force_gc():
    gc.collect()
    logger.info("Memory cleanup performed")

class LowMemoryEmbeddingFunction:
    def __init__(self):
        logger.info("Initializing sentence transformer model")
        self._model = None  # Don't load model in constructor to save memory
        logger.info("Model will be loaded on demand")

    def __call__(self, input: List[str]) -> List[List[float]]:
        # Load model only when needed and with minimal settings
        if self._model is None:
            logger.info("Loading model on first use")
            self._model = SentenceTransformer("all-mpnet-base-v2")
            # L3 model has fewer parameters than L6
            logger.info("Model initialized")

        # Process in very small batches to minimize memory footprint
        logger.info(f"Encoding {len(input)} texts in small batches")
        batch_size = 16  # Small batches to avoid OOM
        all_embeddings = []
        
        for i in range(0, len(input), batch_size):
            force_gc()  # Clean memory before each batch
            batch_end = min(i + batch_size, len(input))
            batch = input[i:batch_end]
            logger.info(f"Encoding batch {i//batch_size + 1}/{(len(input)-1)//batch_size + 1}")
            
            # Process one text at a time for extreme memory savings
            batch_embeddings = []
            for text in batch:
                emb = self._model.encode([text])[0].tolist()
                batch_embeddings.append(emb)
                time.sleep(0.01)  # Small pause to allow memory cleanup
            
            all_embeddings.extend(batch_embeddings)
            logger.info(f"Completed batch {i//batch_size + 1}")
            force_gc()  # Clean memory after each batch
            
        # Clear model after processing to free memory
        self._model = None
        force_gc()
        logger.info("Released model from memory")
        return all_embeddings

def stringify(value):
    if isinstance(value, list):
        if not value:  # Handle empty lists
            return "Not specified"
        return ", ".join(map(str, value))
    return value

def create_vector_db():
    # Setup ChromaDB path and ensure directory exists
    chroma_path = os.path.join("data", "chroma_db")
    logger.info(f"Creating vector database at: {chroma_path}")
    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    logger.info("Initializing ChromaDB client")
    chroma_client = chromadb.PersistentClient(path=chroma_path)

    # Load and validate JSON data
    json_path = os.path.join("data", "SHL_RAW.json")
    if not os.path.exists(json_path):
        logger.error(f"Could not find JSON file at {json_path}")
        raise FileNotFoundError(f"Could not find JSON file at {json_path}")

    logger.info(f"Found JSON data at: {json_path}")

    try:
        # Load JSON in streaming fashion to avoid loading entire file into memory
        assessment_catalog = []
        with open(json_path, "r") as f:
            logger.info("Loading assessment catalog from JSON")
            assessment_catalog = json.load(f)
            if not isinstance(assessment_catalog, list):
                logger.error("JSON data is not a list of assessments")
                raise ValueError("JSON data should be a list of assessments")
            logger.info(f"Loaded {len(assessment_catalog)} assessments from JSON")
    except Exception as e:
        logger.error(f"Failed to load JSON: {e}")
        sys.exit(1)

    # Process in chunks to avoid memory issues
    max_chunk_size = 100  # Process 100 assessments at a time
    chunks = [assessment_catalog[i:i+max_chunk_size] for i in range(0, len(assessment_catalog), max_chunk_size)]
    
    # Create collection with custom embedding function
    logger.info("Creating new collection with low-memory embedding function")
    try:
        # Try to delete existing collection first
        try:
            chroma_client.delete_collection("shl_assessments")
            logger.info("Deleted existing collection")
        except:
            logger.info("No existing collection to delete")
        
        # Create fresh collection
        collection = chroma_client.create_collection(
            name="shl_assessments",
            embedding_function=LowMemoryEmbeddingFunction()
        )
        logger.info("Created collection successfully")
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        sys.exit(1)
    
    # Process and add documents in chunks
    for chunk_idx, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {chunk_idx+1}/{len(chunks)} with {len(chunk)} assessments")
        force_gc()  # Clean memory before processing
        
        # Prepare documents and metadata for this chunk
        documents = []
        metadatas = []
        ids = []
        
        for i, item in enumerate(chunk):
            # Only process valid items
            if not isinstance(item, dict):
                continue
                
            # Check for required fields
            required_fields = [
                "name", "url", "description", "duration", "languages", 
                "job_level", "remote_testing", "adaptive/irt_support", "test_type"
            ]
            if not all(field in item for field in required_fields):
                continue
                
            # Severely truncate description text to minimize memory
            description = item['description']
            if len(description) > 200:
                description = description[:200]
                
            doc_text = f"{item['name']}: {description}: {item['url']}"
            documents.append(doc_text)
            
            # Store the essential metadata
            metadatas.append({
                "name": item["name"],
                "url": item["url"],
                "description": item["description"],
                "duration": item["duration"],
                "languages": stringify(item["languages"]),
                "job_level": item["job_level"],
                "remote_testing": item["remote_testing"],
                "adaptive/irt_support": item["adaptive/irt_support"],
                "test_type": item["test_type"]
            })
            
            ids.append(f"{chunk_idx * max_chunk_size + i}")
        
        if not documents:
            continue
            
        # Process in mini-batches to minimize memory pressure
        mini_batch_size = 25  # Very small batches
        for j in range(0, len(documents), mini_batch_size):
            end_idx = min(j + mini_batch_size, len(documents))
            logger.info(f"Adding mini-batch {j//mini_batch_size + 1}/{(len(documents)-1)//mini_batch_size + 1}")
            
            try:
                collection.add(
                    documents=documents[j:end_idx],
                    metadatas=metadatas[j:end_idx],
                    ids=ids[j:end_idx]
                )
                logger.info(f"Successfully added batch {j}-{end_idx-1}")
            except Exception as e:
                logger.error(f"Failed to add documents: {e}")
                continue
                
            # Force cleanup after each mini-batch
            force_gc()
            
        # Clear chunk data to free memory
        documents = None
        metadatas = None
        ids = None
        force_gc()
    
    # Verify the database was populated
    try:
        count = collection.count()
        logger.info(f"Final collection contains {count} documents")
        if count == 0:
            logger.error("Failed to add any documents to collection")
            return False
    except Exception as e:
        logger.error(f"Failed to count documents: {e}")
        return False
        
    logger.info(f"Successfully created vector DB at {chroma_path}")
    return True

if __name__ == "__main__":
    success = False
    try:
        # Run with memory cleanup
        logger.info("Starting ChromaDB build with extreme memory optimization")
        force_gc()  # Clean before starting
        success = create_vector_db()
        force_gc()  # Clean after finishing
    except Exception as e:
        logger.error(f"Build failed with error: {e}")
        sys.exit(1)
        
    if success:
        logger.info("Build completed successfully")
        sys.exit(0)
    else:
        logger.error("Build failed")
        sys.exit(1)