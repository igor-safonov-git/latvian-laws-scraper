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
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag")

# Ensure logs directory exists
os.makedirs("./logs", exist_ok=True)

# Add file handler for RAG logs
file_handler = logging.FileHandler("./logs/rag.log")
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

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
        "dimensions": 1536  # Using typical dimension count
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
    Query for similar documents from PostgreSQL database using vector similarity.
    Retries 3 times with exponential backoff on failure.
    """
    # Connect to database
    conn = await asyncpg.connect(DATABASE_URL)
    
    try:
        # Convert embedding to correct format for pgvector
        embedding_str = str(embedding).replace("'", "").replace(", ", ",")
        
        # Query for similar documents
        rows = await conn.fetch("""
            SELECT
                metadata->>'translated_text' AS text,
                metadata->>'url' AS url,
                metadata->>'fetched_at' AS fetched_at,
                embedding <-> $1::vector AS score
            FROM docs
            ORDER BY score
            LIMIT $2
        """, embedding_str, limit)
        
        # Convert rows to dictionaries
        results = []
        for row in rows:
            results.append({
                "text": row["text"],
                "url": row["url"],
                "fetched_at": row["fetched_at"],
                "score": row["score"]
            })
        
        return results
    
    finally:
        # Close connection
        await conn.close()


async def retrieve(question: str) -> List[Dict[str, Any]]:
    """
    Retrieve relevant context snippets based on the question.
    Returns up to TOP_K context snippets sorted by relevance.
    
    Each item is formatted as:
    { source: "db", text: str, url: str, fetched_at: str }
    """
    try:
        # Generate embedding for the question
        embedding = await generate_embedding(question)
        
        # Query similar documents
        similar_docs = await query_similar_docs(embedding, TOP_K)
        
        # Filter results based on similarity threshold
        db_hits = [
            {
                "source": "db",
                "text": doc["text"],
                "url": doc["url"],
                "fetched_at": doc["fetched_at"]
            }
            for doc in similar_docs
            if doc["score"] <= SIMILARITY_THRESHOLD
        ]
        
        # Log retrieval stats
        log_entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "db_hits_count": len(db_hits)
        }
        logger.info(json.dumps(log_entry))
        
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
        print(f"Text snippet: {result['text'][:200]}...")


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_retrieve())