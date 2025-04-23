#!/usr/bin/env python3
"""
Memory-optimized embedder service that processes documents into vector embeddings
with minimal RAM footprint (<400MB) to prevent Heroku R14/R15 memory errors.

This service:
1. Runs as a scheduled nightly job at 00:30 UTC (preferably on a worker dyno)
2. Streams data from database with server-side cursors (no fetchall)
3. Processes texts with generator-based chunking
4. Throttles concurrent operations with semaphores
5. Actively monitors and manages memory usage
6. Uses explicit GC hints to release memory
"""
import os
import gc
import json
import psutil
import hashlib
import logging
import asyncio
import aiohttp
import asyncpg
import psycopg2
import psycopg2.extras
from typing import Dict, List, Any, Optional, AsyncIterator, Iterator, Tuple, Callable
from datetime import datetime, timezone
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import tiktoken
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

# Configuration from environment
DATABASE_URL = os.getenv("DATABASE_URL")
CHUNK_TOKEN_SIZE = int(os.getenv("CHUNK_TOKEN_SIZE", "1024"))
BATCH_DELAY_MS = int(os.getenv("BATCH_DELAY_MS", "0"))
MAX_CONCURRENT_EMBEDDINGS = int(os.getenv("MAX_CONCURRENT_EMBEDDINGS", "2"))
MEMORY_LIMIT_MB = int(os.getenv("MEMORY_LIMIT_MB", "350"))
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # Will be updated dynamically if needed

# Encoder for tokenization
encoder = tiktoken.get_encoding("cl100k_base")

class MemoryGuard:
    """Monitors memory usage and prevents exceeding limits."""
    
    @staticmethod
    def get_memory_usage_mb() -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    
    @staticmethod
    async def check_memory(threshold_mb: float = MEMORY_LIMIT_MB) -> bool:
        """
        Check if memory usage is below threshold.
        Returns True if memory usage is acceptable, False if it's too high.
        """
        usage = MemoryGuard.get_memory_usage_mb()
        if usage > threshold_mb:
            logger.warning(f"Memory usage too high: {usage:.2f}MB > {threshold_mb}MB threshold")
            # Trigger garbage collection
            gc.collect()
            await asyncio.sleep(0.1)  # Allow event loop to breathe
            
            # Check again after GC
            usage = MemoryGuard.get_memory_usage_mb()
            if usage > threshold_mb:
                logger.error(f"Memory usage still high after GC: {usage:.2f}MB")
                return False
        return True

class AsyncDatabaseConnector:
    """Manages database connections and operations with asyncpg."""
    
    def __init__(self, database_url: str):
        """Initialize database connector with connection string."""
        self.database_url = database_url
        self.pool = None
        self.conn = None
    
    async def setup_schema(self) -> bool:
        """Set up database tables and extensions."""
        try:
            # Use psycopg2 for initial schema setup since it's more reliable for DDL
            conn = psycopg2.connect(self.database_url)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Enable vector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Check if docs table exists
            cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'docs');")
            table_exists = cursor.fetchone()[0]
            
            if table_exists:
                # Check column dimensions
                try:
                    cursor.execute(f"""
                        SELECT atttypmod FROM pg_attribute 
                        WHERE attrelid = 'docs'::regclass AND attname = 'embedding';
                    """)
                    current_dimensions = cursor.fetchone()[0]
                    
                    if current_dimensions != EMBEDDING_DIMENSIONS:
                        logger.warning(f"Vector dimension mismatch: table has {current_dimensions} dimensions, but {EMBEDDING_DIMENSIONS} expected")
                        logger.info("Recreating docs table with correct dimensions...")
                        
                        # Backup metadata
                        cursor.execute("CREATE TEMP TABLE docs_backup AS SELECT id, metadata FROM docs;")
                        
                        # Drop existing table
                        cursor.execute("DROP TABLE IF EXISTS docs;")
                        
                        # Recreate with correct dimensions
                        cursor.execute(f"""
                            CREATE TABLE docs (
                                id TEXT PRIMARY KEY,
                                metadata JSONB,
                                embedding VECTOR({EMBEDDING_DIMENSIONS})
                            );
                        """)
                        
                        # Restore metadata
                        cursor.execute("""
                            INSERT INTO docs (id, metadata)
                            SELECT id, metadata FROM docs_backup;
                        """)
                        
                        logger.info("Table recreated successfully with correct dimensions")
                except Exception as e:
                    logger.warning(f"Could not check vector dimensions: {e}")
                    # If we can't check dimensions, recreate table to be safe
                    cursor.execute("DROP TABLE IF EXISTS docs;")
                    cursor.execute(f"""
                        CREATE TABLE docs (
                            id TEXT PRIMARY KEY,
                            metadata JSONB,
                            embedding VECTOR({EMBEDDING_DIMENSIONS})
                        );
                    """)
            else:
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
        """Create connection to database using asyncpg."""
        try:
            # Create single connection for operations
            self.conn = await asyncpg.connect(self.database_url)
            logger.info("Database connection established")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            return False
    
    async def disconnect(self) -> None:
        """Close the database connection."""
        if self.conn:
            await self.conn.close()
            self.conn = None
            logger.info("Database connection closed")
    
    async def clear_vectors(self) -> bool:
        """Clear all vectors from the database with memory safety check."""
        if not await MemoryGuard.check_memory():
            logger.error("Memory usage too high to clear vectors")
            return False
        
        try:
            # Use delete with batching for better memory usage
            result = await self.conn.execute("DELETE FROM docs")
            affected = int(result.split(" ")[1]) if "DELETE" in result else 0
            logger.info(f"Cleared {affected} vectors from database")
            
            # Help garbage collector
            gc.collect()
            await asyncio.sleep(0)
            
            return True
        except Exception as e:
            logger.error(f"Failed to clear vectors: {str(e)}")
            return False
    
    async def stream_translations(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream translated documents that need processing using a server-side cursor.
        Yields each document individually to minimize memory usage.
        """
        try:
            # Use asyncpg cursor for streaming results
            async with self.conn.transaction():
                async for record in self.conn.cursor("""
                    SELECT id AS trans_id, url, fetched_at, translated_text
                    FROM raw_docs
                    WHERE translated_text IS NOT NULL
                      AND (processed = FALSE OR processed IS NULL)
                """):
                    # Convert record to dict and yield
                    doc = {
                        "trans_id": record["trans_id"],
                        "url": record["url"],
                        "fetched_at": record["fetched_at"],
                        "translated_text": record["translated_text"]
                    }
                    
                    # Check memory before yielding
                    if await MemoryGuard.check_memory():
                        yield doc
                    else:
                        logger.error(f"Memory limit reached, skipping document {doc['trans_id']}")
                        break
        except Exception as e:
            logger.error(f"Failed to stream translations: {str(e)}")
    
    async def upsert(self, chunk_id: str, embedding: List[float], metadata: Dict[str, Any]) -> bool:
        """Insert or update a document chunk with its embedding."""
        if not await MemoryGuard.check_memory():
            logger.error(f"Memory usage too high to upsert chunk {chunk_id}")
            return False
            
        try:
            # Convert metadata to JSON string
            metadata_json = json.dumps(metadata)
            
            # Prepare embedding for vector type
            embedding_data = embedding
            
            # Perform upsert
            await self.conn.execute("""
                INSERT INTO docs (id, metadata, embedding) 
                VALUES ($1, $2, $3::vector)
                ON CONFLICT (id) DO UPDATE 
                SET metadata = $2, embedding = $3::vector
            """, chunk_id, metadata_json, embedding_data)
            
            return True
        except Exception as e:
            logger.error(f"Failed to upsert document {chunk_id}: {str(e)}")
            return False
    
    async def mark_processed(self, trans_id: str) -> bool:
        """Mark a translation as processed."""
        try:
            # Mark the raw document as processed
            await self.conn.execute(
                "UPDATE raw_docs SET processed = TRUE WHERE id = $1",
                trans_id
            )
            return True
        except Exception as e:
            logger.error(f"Failed to mark document {trans_id} as processed: {str(e)}")
            return False

def stream_chunk_text(text: str, max_tokens: int) -> Iterator[str]:
    """
    Stream-chunk text into slices with maximum token size.
    Uses a generator to yield chunks without loading all at once.
    """
    if not text:
        return
    
    # Process text in segments for large documents to avoid loading all tokens at once
    segment_size = 100000  # Process ~100K chars at a time
    segments = range(0, len(text), segment_size)
    
    for segment_start in segments:
        segment_end = min(segment_start + segment_size, len(text))
        segment = text[segment_start:segment_end]
        
        # Get token IDs from the segment
        tokens = encoder.encode(segment)
        
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
            
            # Help garbage collector between chunks
            chunk_tokens = None

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
async def get_embedding(text: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> List[float]:
    """
    Get embedding from OpenAI with throttling, retry and memory checks.
    Uses semaphore to limit concurrent API calls.
    """
    global EMBEDDING_DIMENSIONS
    
    # Check memory before making API call
    if not await MemoryGuard.check_memory():
        raise Exception("Memory usage too high to generate embedding")
    
    # Use semaphore to limit concurrent API calls
    async with semaphore:
        try:
            # Direct API call using aiohttp session
            api_key = os.getenv("OPENAI_API_KEY")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            payload = {
                "model": EMBEDDING_MODEL,
                "input": text
            }
            
            async with session.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API Error: {response.status}, {error_text}")
                    
                result = await response.json()
                embedding = result["data"][0]["embedding"]
                
                # Verify embedding dimensions
                if len(embedding) != EMBEDDING_DIMENSIONS:
                    logger.warning(f"Unexpected embedding dimensions: got {len(embedding)}, expected {EMBEDDING_DIMENSIONS}")
                    # Update the constant if this is the first run
                    if len(embedding) > 0:
                        EMBEDDING_DIMENSIONS = len(embedding)
                        logger.info(f"Updated EMBEDDING_DIMENSIONS to {EMBEDDING_DIMENSIONS}")
                
                return embedding
                
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
        
        finally:
            # Force garbage collection after embedding
            await asyncio.sleep(0)

async def log_chunk_status(chunk_id: str, status: str, error: Optional[str] = None) -> None:
    """Log chunk processing status to JSON log file."""
    log_entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "chunk_id": chunk_id,
        "status": status,
        "memory_mb": MemoryGuard.get_memory_usage_mb()
    }
    
    if error:
        log_entry["error"] = error
    
    logger.info(json.dumps(log_entry))

class OptimizedEmbedderService:
    """Memory-optimized service for processing documents and generating embeddings."""
    
    def __init__(self):
        """Initialize the embedder service."""
        self.db = AsyncDatabaseConnector(DATABASE_URL)
        self.scheduler = AsyncIOScheduler()
        self.embedding_semaphore = asyncio.Semaphore(MAX_CONCURRENT_EMBEDDINGS)
    
    async def setup(self) -> bool:
        """Set up the service and database."""
        # Test API dimensions first if running as a service
        try:
            async with aiohttp.ClientSession() as session:
                # Use a simple test to check dimensions
                test_embedding = await get_embedding(
                    "Test embedding dimensions", 
                    session, 
                    asyncio.Semaphore(1)
                )
                actual_dimensions = len(test_embedding)
                
                if actual_dimensions != EMBEDDING_DIMENSIONS:
                    logger.warning(f"Embedding dimensions mismatch: got {actual_dimensions}, expected {EMBEDDING_DIMENSIONS}")
                    EMBEDDING_DIMENSIONS = actual_dimensions
                    logger.info(f"Adjusted EMBEDDING_DIMENSIONS to {EMBEDDING_DIMENSIONS}")
                    
                # Force cleanup
                test_embedding = None
                gc.collect()
                await asyncio.sleep(0)
        except Exception as e:
            logger.error(f"Failed to test embedding dimensions: {str(e)}")
            
        # Set up database with potentially updated dimensions
        db_setup = await self.db.setup_schema()
        if db_setup:
            logger.info(f"Embedder service setup complete, using {EMBEDDING_DIMENSIONS} dimensions")
            return True
        return False
    
    async def process_document(self, doc: Dict[str, Any], session: aiohttp.ClientSession) -> Tuple[int, int]:
        """
        Process a single document with memory-efficient streaming.
        
        Args:
            doc: Document with trans_id, url, fetched_at, and translated_text
            session: aiohttp session for HTTP requests
            
        Returns:
            Tuple of (successful_chunks, failed_chunks)
        """
        # Log memory usage at start
        mem_usage = MemoryGuard.get_memory_usage_mb()
        logger.info(f"Starting document processing with {mem_usage:.2f}MB memory usage")
        
        if not await MemoryGuard.check_memory():
            logger.error("Memory usage too high to start document processing")
            return 0, 1
        
        trans_id = doc["trans_id"]
        url = doc["url"]
        fetched_at = doc["fetched_at"]
        text = doc["translated_text"]
        
        logger.info(f"Processing document {trans_id} ({len(text)} chars)")
        
        successful_chunks = 0
        failed_chunks = 0
        
        # Stream-chunk the document with enhanced chunker
        chunk_iter = stream_chunk_text(text, CHUNK_TOKEN_SIZE)
        
        for i, chunk_text in enumerate(chunk_iter):
            # Check memory before processing chunk
            if not await MemoryGuard.check_memory():
                logger.error(f"Memory limit reached, aborting after {i} chunks")
                break
                
            # Compute chunk ID
            chunk_id = hashlib.sha256(f"{trans_id}:{i}".encode()).hexdigest()
            
            try:
                # Get embedding for this chunk
                embedding = await get_embedding(chunk_text, session, self.embedding_semaphore)
                
                # Create metadata with minimal text preview
                metadata = {
                    "url": url,
                    "fetched_at": fetched_at.isoformat() if isinstance(fetched_at, datetime) else fetched_at,
                    "chunk_idx": i,
                    "text_preview": chunk_text[:100] + ("..." if len(chunk_text) > 100 else "")
                }
                
                # Insert into database
                success = await self.db.upsert(chunk_id, embedding, metadata)
                
                if success:
                    await log_chunk_status(chunk_id, "ok")
                    successful_chunks += 1
                else:
                    await log_chunk_status(chunk_id, "error", "Database insertion failed")
                    failed_chunks += 1
                
                # Apply delay if configured or force a small delay for GC
                delay_time = max(BATCH_DELAY_MS / 1000, 0.01)
                await asyncio.sleep(delay_time)
                    
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
                await log_chunk_status(chunk_id, "error", str(e))
                failed_chunks += 1
                
            finally:
                # Explicitly help garbage collection
                chunk_text = None
                embedding = None
                metadata = None
                gc.collect()
        
        # Check if we should mark as processed
        if successful_chunks > 0:
            # Mark document as processed
            await self.db.mark_processed(trans_id)
        
        # Final memory cleanup
        text = None
        gc.collect()
        
        logger.info(f"Completed document {trans_id}: {successful_chunks} chunks succeeded, {failed_chunks} failed")
        logger.info(f"Memory usage after document: {MemoryGuard.get_memory_usage_mb():.2f}MB")
        
        return successful_chunks, failed_chunks
    
    async def process_batch(self) -> None:
        """Process all pending translated documents with memory monitoring."""
        logger.info("Starting nightly batch processing")
        logger.info(f"Initial memory usage: {MemoryGuard.get_memory_usage_mb():.2f}MB")
        
        # Connect to database
        if not await self.db.connect():
            logger.error("Failed to connect to database, aborting batch")
            return
        
        try:
            # Clear previous vectors
            if not await self.db.clear_vectors():
                logger.error("Failed to clear vectors, aborting batch")
                return
            
            # Track processing stats
            total_docs = 0
            processed_docs = 0
            total_chunks = 0
            failed_chunks = 0
            
            # Process documents one by one
            async with aiohttp.ClientSession() as session:
                async for doc in self.db.stream_translations():
                    # Check memory before each document
                    if not await MemoryGuard.check_memory():
                        logger.error("Memory limit reached, pausing batch processing")
                        # Force GC and wait
                        gc.collect()
                        await asyncio.sleep(5)
                        
                        # Check again
                        if not await MemoryGuard.check_memory():
                            logger.error("Memory still high after pause, aborting batch")
                            break
                    
                    total_docs += 1
                    successes, failures = await self.process_document(doc, session)
                    
                    if successes > 0:
                        processed_docs += 1
                    
                    total_chunks += successes
                    failed_chunks += failures
                    
                    # Log progress and memory usage
                    if total_docs % 5 == 0:
                        logger.info(f"Progress: {processed_docs}/{total_docs} docs, {total_chunks} chunks, memory: {MemoryGuard.get_memory_usage_mb():.2f}MB")
            
            logger.info(f"Batch complete: {processed_docs}/{total_docs} documents, {total_chunks} chunks created, {failed_chunks} chunks failed")
            logger.info(f"Final memory usage: {MemoryGuard.get_memory_usage_mb():.2f}MB")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
        
        finally:
            # Close database connection
            await self.db.disconnect()
            
            # Final GC
            gc.collect()
    
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
            # Keep the service running with periodic memory checks
            while True:
                mem_usage = MemoryGuard.get_memory_usage_mb()
                if mem_usage > MEMORY_LIMIT_MB:
                    logger.warning(f"High memory usage during idle: {mem_usage:.2f}MB")
                    gc.collect()
                
                await asyncio.sleep(60)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Embedder service shutting down")
            self.scheduler.shutdown()

async def run_once() -> None:
    """Run the embedder service once for manual execution."""
    # Log starting memory
    start_mem = MemoryGuard.get_memory_usage_mb()
    logger.info(f"Starting one-time execution with {start_mem:.2f}MB memory usage")
    
    service = OptimizedEmbedderService()
    
    # Setup and run
    setup_success = await service.setup()
    
    if not setup_success:
        logger.error("Failed to set up embedder service")
        return
    
    await service.process_batch()
    
    # Final memory cleanup
    gc.collect()
    end_mem = MemoryGuard.get_memory_usage_mb()
    logger.info(f"Manual execution complete. Memory usage: {end_mem:.2f}MB (change: {end_mem-start_mem:.2f}MB)")

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Memory-optimized embedder service for vector embeddings")
    parser.add_argument("--once", action="store_true", help="Run once and exit (no scheduling)")
    parser.add_argument("--memory-limit", type=int, default=MEMORY_LIMIT_MB, 
                        help=f"Memory limit in MB (default: {MEMORY_LIMIT_MB})")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT_EMBEDDINGS,
                        help=f"Maximum concurrent embedding operations (default: {MAX_CONCURRENT_EMBEDDINGS})")
    
    # Handle both direct arguments and passing after -- in Heroku
    if "--once" in sys.argv:
        args = parser.parse_args()
    else:
        try:
            args = parser.parse_args()
        except:
            args = argparse.Namespace(once=False, memory_limit=MEMORY_LIMIT_MB, concurrency=MAX_CONCURRENT_EMBEDDINGS)
            for i, arg in enumerate(sys.argv):
                if arg == "--" and i+1 < len(sys.argv):
                    remaining = sys.argv[i+1:]
                    if "--once" in remaining:
                        args.once = True
                    if "--memory-limit" in remaining:
                        idx = remaining.index("--memory-limit")
                        if idx+1 < len(remaining):
                            args.memory_limit = int(remaining[idx+1])
                    if "--concurrency" in remaining:
                        idx = remaining.index("--concurrency")
                        if idx+1 < len(remaining):
                            args.concurrency = int(remaining[idx+1])
                    break
    
    # Apply command line arguments
    MEMORY_LIMIT_MB = args.memory_limit
    MAX_CONCURRENT_EMBEDDINGS = args.concurrency
    
    logger.info(f"Using memory limit: {MEMORY_LIMIT_MB}MB, concurrency: {MAX_CONCURRENT_EMBEDDINGS}")
    
    # Run either once or as a scheduled service
    if args.once:
        logger.info("Running embedder service once")
        asyncio.run(run_once())
    else:
        # Run as a service
        logger.info("Starting scheduled embedder service")
        asyncio.run(OptimizedEmbedderService().run())