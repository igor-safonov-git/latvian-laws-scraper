#!/usr/bin/env python3
"""
RAG (Retrieval Augmented Generation) module for retrieving relevant context
from PostgreSQL+pgvector database based on semantic similarity.
"""
import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import tiktoken
import asyncpg
import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum detail
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag")

# Ensure logs directory exists
os.makedirs("./logs", exist_ok=True)

# Add file handler for RAG logs
file_handler = logging.FileHandler("./logs/rag.log")
file_handler.setLevel(logging.DEBUG)  # Set to DEBUG for maximum detail
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Add console handler to ensure logs go to the console too
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Set to DEBUG for maximum detail
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Direct output to log file
logger.propagate = False  # Prevent duplication of log messages

# Get configuration from environment
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "8000"))

# OpenAI API endpoint
OPENAI_EMBEDDING_ENDPOINT = "https://api.openai.com/v1/embeddings"

# Initialize tokenizer for text truncation
encoder = tiktoken.get_encoding("cl100k_base")


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """Truncate text to the specified token limit."""
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate to max tokens
    truncated_tokens = tokens[:max_tokens]
    return encoder.decode(truncated_tokens)


@retry(
    retry=retry_if_exception_type((Exception,)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def generate_embedding(question: str) -> List[float]:
    """
    Generate an embedding for the question using OpenAI API.
    Retries 3 times with exponential backoff on failure.
    
    Uses direct API call with requests to avoid client library issues.
    """
    # Truncate to token limit
    truncated_question = truncate_to_token_limit(question, MAX_CONTEXT_TOKENS)
    
    # Prepare API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    payload = {
        "input": truncated_question,
        "model": "text-embedding-3-small", 
        "dimensions": 1536  # Must match the dimension count in our database (1536)
    }
    
    # Make API request (synchronously, then await to maintain async function signature)
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: requests.post(
            OPENAI_EMBEDDING_ENDPOINT,
            headers=headers,
            json=payload
        )
    )
    
    # Check response
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code}, {response.text}")
    
    # Parse response
    result = response.json()
    
    # Extract embedding
    embedding = result["data"][0]["embedding"]
    
    return embedding


@retry(
    retry=retry_if_exception_type((asyncpg.PostgresError, Exception)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def query_similar_docs(embedding: List[float], limit: int) -> List[Dict[str, Any]]:
    """
    Query for similar document chunks from PostgreSQL+pgvector using vector similarity.
    Retrieves chunk-level context for RAG.
    Retries 3 times with exponential backoff on failure.
    """
    # Connect to database
    conn = await asyncpg.connect(DATABASE_URL)
    
    try:
        # Convert embedding to correct format for pgvector
        embedding_str = str(embedding).replace("'", "").replace(", ", ",")
        
        # Query for similar documents from the docs table
        # This is where the embeddings currently exist
        rows = await conn.fetch(
            """
            SELECT
                id,
                metadata,
                embedding <-> $1::vector AS score
            FROM docs
            WHERE embedding IS NOT NULL
            ORDER BY score
            LIMIT $2
            """,
            embedding_str,
            limit
        )
        
        # Log raw results for debugging
        try:
            # Convert rows to a more loggable format
            log_rows = []
            for row in rows:
                log_rows.append({
                    "id": row["id"],
                    "score": float(row["score"]),
                    "metadata_keys": list(row["metadata"].keys()) if row["metadata"] else []
                })
            logger.info(f"Raw query results: {json.dumps(log_rows)}")
        except Exception as e:
            logger.error(f"Error logging query results: {str(e)}")
        
        # Convert rows to dictionaries
        results = []
        for i, row in enumerate(rows):
            # Log the full metadata for debugging
            try:
                logger.info(f"Document {i} full metadata: {row['metadata']}")
                logger.info(f"Document {i} metadata type: {type(row['metadata']).__name__}")
                
                # Parse metadata if it's a string (JSON)
                if isinstance(row['metadata'], str):
                    try:
                        metadata = json.loads(row['metadata'])
                        logger.info(f"Successfully parsed metadata JSON to dict")
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse metadata JSON: {row['metadata']}")
                        metadata = {}
                else:
                    metadata = row["metadata"] if row["metadata"] else {}
            except Exception as e:
                logger.error(f"Document {i} metadata error: {str(e)}")
                metadata = {}
            
            # Check if text_preview exists and log it with more details
            text_preview = metadata.get("text_preview")
            logger.info(f"Document {i} text_preview: {text_preview}")
            logger.info(f"Document {i} metadata keys: {list(metadata.keys()) if metadata else []}")
            
            # Create result with debug info and fallback text
            # Handle the case where text_preview might be missing
            if not text_preview:
                logger.warning(f"Document {i} missing text_preview field, generating fallback")
                # Create a fallback preview from raw metadata string if available
                try:
                    metadata_str = str(row.get('metadata', '{}'))
                    if len(metadata_str) > 20:
                        text_preview = f"Metadata-based preview: {metadata_str[:100]}..."
                    else:
                        text_preview = "No preview available (metadata issue)"
                except:
                    text_preview = "No preview available"
            
            result = {
                "text": text_preview if text_preview is not None else "No preview available",
                "url": metadata.get("url", "Unknown URL"),
                "fetched_at": metadata.get("fetched_at", "Unknown date"),
                "score": row["score"],
                "metadata_keys": list(metadata.keys()) if metadata else []
            }
            results.append(result)
            logger.info(f"Document {i} result: {json.dumps(result)}")
        
        return results
    
    finally:
        # Close connection
        await conn.close()


async def retrieve(question: str) -> List[Dict[str, Any]]:
    """
    Retrieve relevant context snippets based on the question.
    Returns up to TOP_K context snippets sorted by relevance.
    
    Each item is formatted as:
    { source: "db", text: str, url: str, fetched_at: str, metadata_keys: list }
    """
    # Force the log level to DEBUG for this run
    logging.getLogger('rag').setLevel(logging.DEBUG)
    logger.info("==================== RETRIEVAL STARTED ====================")
    logger.info(f"Starting retrieval for query: '{question}'")
    logger.info(f"Using similarity threshold: {SIMILARITY_THRESHOLD}")
    try:
        # Generate embedding for the question
        embedding = await generate_embedding(question)
        
        # Log embedding dimensions for debugging
        logger.info(f"Generated embedding with {len(embedding)} dimensions")
        
        # Query similar documents
        similar_docs = await query_similar_docs(embedding, TOP_K)
        
        # Log raw results before filtering
        scores_log = []
        for i, doc in enumerate(similar_docs):
            scores_log.append((i, f"{doc['score']:.4f}"))
        logger.info(f"Raw similarity scores: {scores_log}")
        
        # Filter results based on similarity threshold
        db_hits = []
        logger.info(f"Similarity threshold is set to: {SIMILARITY_THRESHOLD}")
        
        # Extract raw results first for better logging
        logger.info(f"Processing {len(similar_docs)} candidate documents")
        
        for i, doc in enumerate(similar_docs):
            try:
                logger.info(f"Document {i}: {doc.keys()}")
                score = float(doc["score"]) if "score" in doc else 999.0
                logger.info(f"Document {i} score: {score:.4f}")
                
                if score <= SIMILARITY_THRESHOLD:
                    # Add this document to results
                    db_hits.append({
                        "source": "db",
                        "text": doc.get("text", "No text available"),
                        "url": doc.get("url", "Unknown URL"),
                        "fetched_at": doc.get("fetched_at", "Unknown date"),
                        "score": score
                    })
                    logger.info(f"Added document {i} with score {score:.4f} below threshold {SIMILARITY_THRESHOLD}")
                else:
                    logger.info(f"Skipped document {i} with score {score:.4f} above threshold {SIMILARITY_THRESHOLD}")
            except Exception as e:
                logger.error(f"Error processing document {i}: {str(e)}")
                logger.error(f"Document content: {str(doc)}")
        
        # Log retrieval stats with more detail
        log_entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "db_hits_count": len(db_hits),
            "all_results_count": len(similar_docs),
            "threshold": SIMILARITY_THRESHOLD,
            "scores": [f"{doc['score']:.4f}" for doc in similar_docs[:3]] if similar_docs else []
        }
        logger.info(json.dumps(log_entry))
        
        # Add a final log message for easy identification in logs
        logger.info(f"==================== RETRIEVAL COMPLETE ====================")
        logger.info(f"Final results count: {len(db_hits)}")
        
        # Print the final db_hits for debugging
        for i, hit in enumerate(db_hits):
            logger.info(f"Result {i}: {hit}")
        
        return db_hits
    
    except Exception as e:
        # Log error and return empty list on final failure
        logger.error(f"Error retrieving context for question '{question}': {str(e)}")
        return []


async def test_retrieve():
    """Test the retrieve function with a sample question."""
    question = "What are the regulations for digital signatures in Latvia?"
    results = await retrieve(question)
    
    print(f"Question: {question}")
    print(f"Found {len(results)} relevant documents")
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"URL: {result['url']}")
        print(f"Fetched: {result['fetched_at']}")
        if result.get('text'):
            print(f"Text snippet: {result['text'][:200]}...")
        else:
            print("Text snippet: No text available")


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_retrieve())