#!/usr/bin/env python3
"""
Embedder service that processes translated documents into vector embeddings
using OpenAI's text-embedding-3-small model with streaming for memory efficiency.

This service:
1. Runs as a scheduled nightly job at 00:30 UTC
2. Clears previous vectors from the database
3. Processes all translated documents into chunks
4. Computes embeddings for each chunk
5. Stores embeddings with metadata in PostgreSQL with pgvector extension
6. Updates document status when processed
7. Logs all operations
"""
import os
import json
import hashlib
import logging
import asyncio
import aiohttp
import psycopg2
import psycopg2.extras
from typing import Dict, List, Any, Optional, AsyncIterator, Iterator, Tuple
from datetime import datetime, timezone
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import tiktoken
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("embedder")

# Ensure logs directory exists
if not os.path.exists("./logs"):
    os.makedirs("./logs")

# Setup file handler for JSON logs
file_handler = logging.FileHandler("./logs/embedder.log")
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Initialize OpenAI client with API key
openai.api_key = os.getenv("OPENAI_API_KEY")

encoder = tiktoken.get_encoding("cl100k_base")  # Used by text-embedding models

# Configuration from environment
DATABASE_URL = os.getenv("DATABASE_URL")
CHUNK_TOKEN_SIZE = int(os.getenv("CHUNK_TOKEN_SIZE", "1024"))
BATCH_DELAY_MS = int(os.getenv("BATCH_DELAY_MS", "0"))
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 512  # Specific to text-embedding-3-small

class DatabaseConnector:
    """Manages database connections and operations."""
    
    def __init__(self, database_url: str):
        """Initialize database connector with connection string."""
        self.database_url = database_url
        self.pool = None
    
    async def setup(self) -> bool:
        """Set up database tables and extensions."""
        try:
            # Connect directly for schema setup (no async pool yet)
            conn = psycopg2.connect(self.database_url)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Enable vector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create documents table if it doesn't exist
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS docs (
                    id TEXT PRIMARY KEY,
                    metadata JSONB,
                    embedding VECTOR({EMBEDDING_DIMENSIONS})
                );
            """)
            
            # Optional: create HNSW index for faster similarity search
            try:
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS docs_embedding_idx 
                    ON docs USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """ )
            except Exception as e:
                logger.warning(f"Could not create HNSW index: {e}")
            
            cursor.close()
            conn.close()
            logger.info("Database setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Database setup failed: {str(e)}")
            return False
    
    async def connect(self) -> bool:
        """Create a connection pool."""
        try:
            # For simplicity, we're using synchronous psycopg2
            # In a production environment, consider using asyncpg
            self.conn = psycopg2.connect(self.database_url)
            self.conn.autocommit = True
            logger.info("Database connection established")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            return False
    
    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    async def clear_vectors(self) -> bool:
        """Clear all vectors from the database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM docs;")
            affected = cursor.rowcount
            cursor.close()
            logger.info(f"Cleared {affected} vectors from database")
            return True
        except Exception as e:
            logger.error(f"Failed to clear vectors: {str(e)}")
            return False
    
    async def get_translations(self) -> List[Dict[str, Any]]:
        """Get all translated documents that need processing."""
        try:
            # Fetch untranslated documents from raw_docs
            cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("""
                SELECT id AS trans_id, url, fetched_at, translated_text
                FROM raw_docs
                WHERE translated_text IS NOT NULL
                  AND (processed = FALSE OR processed IS NULL)
            """ )
            results = cursor.fetchall()
            cursor.close()
            logger.info(f"Found {len(results)} translated documents to process")
            return results
        except Exception as e:
            logger.error(f"Failed to get translations: {str(e)}")
            return []
    
    async def upsert(self, chunk_id: str, embedding: List[float], metadata: Dict[str, Any]) -> bool:
        """Insert or update a document chunk with its embedding."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO docs (id, metadata, embedding) 
                VALUES (%s, %s, %s::vector)
                ON CONFLICT (id) DO UPDATE 
                SET metadata = %s, embedding = %s::vector
                """,
                (
                    chunk_id,
                    json.dumps(metadata),
                    embedding,
                    json.dumps(metadata),
                    embedding
                )
            )
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Failed to upsert document {chunk_id}: {str(e)}")
            return False
    
    async def mark_processed(self, trans_id: str) -> bool:
        """Mark a translation as processed."""
        try:
            # Mark the raw document as processed
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE raw_docs SET processed = TRUE WHERE id = %s",
                (trans_id,)
            )
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Failed to mark document {trans_id} as processed: {str(e)}")
            return False

def chunker(text: str, max_tokens: int) -> Iterator[str]:
    """
    Stream-chunk text into slices with a maximum token size.
    Uses a generator to yield chunks without loading all at once.
    
    Args:
        text: The text to chunk
        max_tokens: Maximum tokens per chunk
        
    Yields:
        Chunks of text, each with no more than max_tokens
    """
    if not text:
        return
    
    # Get token IDs from the text
    tokens = encoder.encode(text)
    
    # Create chunks with maximum token size
    start_idx = 0
    while start_idx < len(tokens):
        # Get chunk of appropriate token size
        end_idx = min(start_idx + max_tokens, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        
        # Convert tokens back to text
        chunk = encoder.decode(chunk_tokens)
        
        # Yield the chunk
        yield chunk
        
        # Update start index for next chunk
        start_idx = end_idx

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
async def get_embedding(text: str, session: aiohttp.ClientSession) -> List[float]:
    """
    Get embedding from OpenAI with retry and exponential backoff.
    Uses the global client with API key.
    
    Args:
        text: Text to embed
        session: aiohttp session for HTTP requests
        
    Returns:
        List of embedding values
        
    Raises:
        Exception: If embedding fails after retries
    """
    try:
        # Use the standard client (not async) in a thread pool to avoid issues
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

async def log_chunk_status(chunk_id: str, status: str, error: Optional[str] = None) -> None:
    """
    Log chunk processing status to JSON log file.
    
    Args:
        chunk_id: ID of the processed chunk
        status: "ok" or "error"
        error: Error message if status is "error"
    """
    log_entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "chunk_id": chunk_id,
        "status": status
    }
    
    if error:
        log_entry["error"] = error
    
    logger.info(json.dumps(log_entry))

class EmbedderService:
    """Main service for processing documents and generating embeddings."""
    
    def __init__(self):
        """Initialize the embedder service."""
        self.db = DatabaseConnector(DATABASE_URL)
        self.scheduler = AsyncIOScheduler()
    
    async def setup(self) -> bool:
        """Set up the service and database."""
        db_setup = await self.db.setup()
        if db_setup:
            logger.info("Embedder service setup complete")
            return True
        return False
    
    async def process_document(self, doc: Dict[str, Any], session: aiohttp.ClientSession) -> Tuple[int, int]:
        """
        Process a single document: chunk it, generate embeddings, and store in database.
        
        Args:
            doc: Document with trans_id, url, fetched_at, and translated_text
            session: aiohttp session for HTTP requests
            
        Returns:
            Tuple of (successful_chunks, failed_chunks)
        """
        trans_id = doc["trans_id"]
        url = doc["url"]
        fetched_at = doc["fetched_at"]
        text = doc["translated_text"]
        
        logger.info(f"Processing document {trans_id} ({len(text)} chars)")
        
        successful_chunks = 0
        failed_chunks = 0
        
        # Stream-chunk the document
        for i, chunk_text in enumerate(chunker(text, CHUNK_TOKEN_SIZE)):
            # Compute chunk ID
            chunk_id = hashlib.sha256(f"{trans_id}:{i}".encode()).hexdigest()
            
            try:
                # Get embedding for this chunk
                embedding = await get_embedding(chunk_text, session)
                
                # Store in database with metadata including a text preview
                metadata = {
                    "url": url,
                    "fetched_at": fetched_at.isoformat() if isinstance(fetched_at, datetime) else fetched_at,
                    "chunk_idx": i,
                    "text_preview": chunk_text[:200] + ("..." if len(chunk_text) > 200 else "")
                }
                
                success = await self.db.upsert(chunk_id, embedding, metadata)
                
                if success:
                    await log_chunk_status(chunk_id, "ok")
                    successful_chunks += 1
                else:
                    await log_chunk_status(chunk_id, "error", "Database insertion failed")
                    failed_chunks += 1
                
                # Apply delay if configured
                if BATCH_DELAY_MS > 0:
                    await asyncio.sleep(BATCH_DELAY_MS/1000)
                    
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
                await log_chunk_status(chunk_id, "error", str(e))
                failed_chunks += 1
                
            # Explicitly help garbage collection
            chunk_text = None
        
        # Mark document as processed
        await self.db.mark_processed(trans_id)
        
        logger.info(f"Completed document {trans_id}: {successful_chunks} chunks succeeded, {failed_chunks} failed")
        return successful_chunks, failed_chunks
    
    async def process_batch(self) -> None:
        """Process all pending translated documents."""
        logger.info("Starting nightly batch processing")
        
        # Connect to database
        if not await self.db.connect():
            logger.error("Failed to connect to database, aborting batch")
            return
        
        try:
            # Clear previous vectors
            await self.db.clear_vectors()
            
            # Get all translations
            docs = await self.db.get_translations()
            
            if not docs:
                logger.info("No documents to process")
                return
            
            # Process each document
            total_chunks = 0
            failed_chunks = 0
            
            async with aiohttp.ClientSession() as session:
                for doc in docs:
                    successes, failures = await self.process_document(doc, session)
                    total_chunks += successes
                    failed_chunks += failures
            
            logger.info(f"Batch complete: {len(docs)} documents, {total_chunks} chunks created, {failed_chunks} chunks failed")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
        
        finally:
            # Close database connection
            await self.db.disconnect()
    
    def schedule_jobs(self) -> None:
        """Schedule the nightly batch job."""
        self.scheduler.add_job(
            self.process_batch,
            'cron',
            hour=0,
            minute=30,
            id='nightly_batch'
        )
        logger.info("Scheduled nightly batch job for 00:30 UTC")
    
    async def run(self) -> None:
        """Run the embedder service."""
        # Set up the service
        setup_success = await self.setup()
        if not setup_success:
            logger.error("Failed to set up embedder service")
            return
        
        # Schedule jobs
        self.schedule_jobs()
        
        # Start scheduler
        self.scheduler.start()
        logger.info("Embedder service started")
        
        try:
            # Keep the service running
            while True:
                await asyncio.sleep(60)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Embedder service shutting down")
            self.scheduler.shutdown()

async def run_once() -> None:
    """Run the embedder service once for manual execution."""
    service = EmbedderService()
    setup_success = await service.setup()
    
    if not setup_success:
        logger.error("Failed to set up embedder service")
        return
    
    await service.process_batch()
    logger.info("Manual execution complete")

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Embedder service for vector embeddings")
    parser.add_argument("--once", action="store_true", help="Run once and exit (no scheduling)")
    
    # Handle both direct arguments and passing after -- in Heroku
    if "--once" in sys.argv:
        args = parser.parse_args()
    else:
        # Try to parse as if passed after -- in Heroku
        try:
            args = parser.parse_args()
        except:
            # Default to running as a service
            args = argparse.Namespace(once=False)
            # Check if passed after -- in Heroku command
            for i, arg in enumerate(sys.argv):
                if arg == "--" and i+1 < len(sys.argv) and sys.argv[i+1] == "--once":
                    args.once = True
                    break
    
    # Run either once or as a scheduled service
    if args.once:
        logger.info("Running embedder service once")
        asyncio.run(run_once())
    else:
        # Run as a service
        logger.info("Starting scheduled embedder service")
        asyncio.run(EmbedderService().run())