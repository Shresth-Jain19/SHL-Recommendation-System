import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_chroma_db():
    db_path = "data/chroma_db"
    
    logger.info(f"Verifying ChromaDB at path: {db_path}")
    
    if not os.path.exists(db_path):
        logger.error(f"ChromaDB directory not found at: {db_path}")
        return False
        
    if not os.path.isdir(db_path):
        logger.error(f"ChromaDB path exists but is not a directory: {db_path}")
        return False
    
    contents = os.listdir(db_path)
    if not contents:
        logger.error(f"ChromaDB directory is empty: {db_path}")
        return False
    
    logger.info(f"ChromaDB verified successfully with {len(contents)} items")
    return True

if __name__ == "__main__":
    if verify_chroma_db():
        logger.info("Verification successful. Database is ready.")
        sys.exit(0)
    else:
        logger.error("Verification failed. Database is not ready.")
    sys.exit(1)