#!/usr/bin/env python3
"""
Memory-optimized embedder service that processes documents into vector embeddings
with minimal RAM footprint to prevent Heroku memory errors.

This service:
1. Runs on-demand to process documents
2. Streams data from database with server-side cursors (no fetchall)
3. Processes texts with generator-based chunking
4. Throttles concurrent operations with semaphores
5. Monitors memory usage
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
from typing import Dict, List, Any, Optional, AsyncIterator, Iterator, Tuple
from datetime import datetime, timezone
from dotenv import load_dotenv
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("embedder")

# Suppress debug logs from HTTP libraries
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

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
BATCH_DELAY_MS = int(os.getenv("BATCH_DELAY_MS", "1"))
MAX_CONCURRENT_EMBEDDINGS = int(os.getenv("MAX_CONCURRENT_EMBEDDINGS", "1"))
MEMORY_LIMIT_MB = int(os.getenv("MEMORY_LIMIT_MB", "400"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSIONS = 3072

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
            await asyncio.sleep(1)  # Allow event loop to breathe
            
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
            
            # Force recreate all tables with correct dimensions - this is important to avoid dimension mismatch
            logger.info(f"Setting up database schema with {EMBEDDING_DIMENSIONS} dimensions")
            
            # Drop existing tables to ensure dimension consistency
            cursor.execute("DROP TABLE IF EXISTS doc_chunks;")
            cursor.execute("DROP TABLE IF EXISTS doc_summaries;")
            cursor.execute("DROP TABLE IF EXISTS docs;")
            
            # Create tables with correct dimensions
            logger.info("Creating docs table...")
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS docs (
                    id TEXT PRIMARY KEY,
                    metadata JSONB,
                    embedding VECTOR({EMBEDDING_DIMENSIONS})
                );
            """)
            
            logger.info("Creating doc_chunks table...")
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS doc_chunks (
                    id TEXT NOT NULL,
                    chunk_id TEXT PRIMARY KEY,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    metadata JSONB,
                    embedding VECTOR({EMBEDDING_DIMENSIONS})
                );
            """)
            
            logger.info("Creating doc_summaries table...")
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS doc_summaries (
                    id TEXT PRIMARY KEY,
                    summary_text TEXT NOT NULL,
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
            logger.info(f"Database schema setup complete with {EMBEDDING_DIMENSIONS} dimensions")
            return True
            
        except Exception as e:
            logger.error(f"Database setup failed: {str(e)}")
            return False
    
    async def connect(self) -> bool:
        """Create connection to database using asyncpg."""
        try:
            # Create single connection for operations
            self.conn = await asyncpg.connect(self.database_url)
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            return False
    
    async def disconnect(self) -> None:
        """Close the database connection."""
        if self.conn:
            await self.conn.close()
            self.conn = None
    
    async def clear_vectors(self) -> bool:
        """Clear all vectors from all vector tables unconditionally."""
        if not await MemoryGuard.check_memory():
            logger.error("Memory usage too high to clear vectors")
            return False
        
        try:
            # Use TRUNCATE for faster clearing of all vector tables
            logger.info("Unconditionally clearing ALL vector tables")
            
            # Clear main document vectors
            await self.conn.execute("TRUNCATE TABLE docs")
            
            # Clear chunk vectors
            await self.conn.execute("TRUNCATE TABLE doc_chunks")
            
            # Clear summary vectors
            await self.conn.execute("TRUNCATE TABLE doc_summaries")
            
            logger.info("Successfully cleared all vector tables")
            
            # Help garbage collector
            gc.collect()
            await asyncio.sleep(0)
            
            return True
        except Exception as e:
            logger.error(f"Failed to clear vector tables: {str(e)}")
            return False
    
    async def stream_translations(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream translated documents that need processing using a server-side cursor.
        Yields each document individually to minimize memory usage.
        """
        try:
            # Use asyncpg cursor for streaming results
            async with self.conn.transaction():
                # First, count available documents
                count = await self.conn.fetchval("""
                    SELECT COUNT(*) 
                    FROM raw_docs
                    WHERE translated_text IS NOT NULL
                """)
                
                logger.info(f"Found {count} documents with translations")
                
                # Stream the documents that need processing
                async for record in self.conn.cursor("""
                    SELECT id AS trans_id, url, fetched_at, translated_text
                    FROM raw_docs
                    WHERE translated_text IS NOT NULL
                """):
                    # Convert record to dict and yield
                    doc = {
                        "trans_id": record["trans_id"],
                        "url": record["url"],
                        "fetched_at": record["fetched_at"],
                        "translated_text": record["translated_text"]
                    }
                    
                    if await MemoryGuard.check_memory():
                        yield doc
                    else:
                        logger.error(f"Memory limit reached, skipping document {doc['trans_id']}")
                        break
        except Exception as e:
            logger.error(f"Failed to stream translations: {str(e)}")
    
    async def upsert(self, chunk_id: str, embedding: List[float], metadata: Dict[str, Any]) -> bool:
        """Insert or update a document chunk with its embedding in the docs table."""
        if not await MemoryGuard.check_memory():
            return False
            
        try:
            # Convert metadata to JSON string
            metadata_json = json.dumps(metadata)
            
            # Prepare embedding as a string for pgvector
            embedding_str = str(embedding).replace("'", "").replace(", ", ",")
            
            # Perform upsert
            await self.conn.execute("""
                INSERT INTO docs (id, metadata, embedding) 
                VALUES ($1, $2, $3::vector)
                ON CONFLICT (id) DO UPDATE 
                SET metadata = $2, embedding = $3::vector
            """, chunk_id, metadata_json, embedding_str)
            
            return True
        except Exception as e:
            logger.error(f"Failed to upsert document {chunk_id}: {str(e)}")
            return False
    
    async def insert_chunk(self, doc_id: str, chunk_id: str, chunk_index: int, chunk_text: str, 
                         embedding: List[float], metadata: Dict[str, Any]) -> bool:
        """Insert a chunk into the doc_chunks table with full text and embedding."""
        if not await MemoryGuard.check_memory():
            return False
            
        try:
            # Convert metadata to JSON string
            metadata_json = json.dumps(metadata)
            
            # Prepare embedding as a string for pgvector
            embedding_str = str(embedding).replace("'", "").replace(", ", ",")
            
            # Insert into doc_chunks table
            await self.conn.execute("""
                INSERT INTO doc_chunks (id, chunk_id, chunk_index, chunk_text, metadata, embedding) 
                VALUES ($1, $2, $3, $4, $5, $6::vector)
                ON CONFLICT (chunk_id) DO UPDATE 
                SET chunk_text = $4, metadata = $5, embedding = $6::vector
            """, doc_id, chunk_id, chunk_index, chunk_text, metadata_json, embedding_str)
            
            logger.info(f"Inserted chunk {chunk_id} into doc_chunks table ({len(chunk_text)} chars)")
            return True
        except Exception as e:
            logger.error(f"Failed to insert chunk {chunk_id}: {str(e)}")
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

def split_into_sentences(text):
    """Split text into sentences using regex for better boundary detection."""
    # Pattern matches sentence endings (period, question mark, exclamation mark) followed by space/newline
    sentence_endings = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_endings, text)
    # Filter out empty sentences and strip whitespace
    return [s.strip() for s in sentences if s.strip()]

def intelligent_chunk_text(text: str, max_tokens: int, overlap_tokens: int = 200) -> Iterator[str]:
    """
    Create chunks intelligently by respecting sentence boundaries with overlap.
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks (default 200)
        
    Returns:
        Iterator of text chunks
    """
    if not text:
        return
    
    # First split the text into sentences
    sentences = split_into_sentences(text)
    
    # Process in batches to avoid loading everything into memory
    current_chunk = []
    current_token_count = 0
    prev_end_sentences = []  # Store sentences from end of previous chunk for overlap
    
    for sentence in sentences:
        # Count tokens in this sentence
        sentence_tokens = len(encoder.encode(sentence))
        
        # Check if adding this sentence would exceed the limit
        if current_token_count + sentence_tokens > max_tokens and current_chunk:
            # If this sentence would push us over the limit, yield the current chunk
            chunk_text = " ".join(current_chunk)
            yield chunk_text
            
            # Save the last few sentences for overlap with next chunk
            # Take sentences from the end of current chunk that fit within overlap_tokens
            overlap_count = 0
            prev_end_sentences = []
            for s in reversed(current_chunk):
                s_tokens = len(encoder.encode(s))
                if overlap_count + s_tokens <= overlap_tokens:
                    prev_end_sentences.insert(0, s)
                    overlap_count += s_tokens
                else:
                    break
            
            # Start a new chunk with overlap sentences
            current_chunk = prev_end_sentences.copy()
            current_token_count = overlap_count
        
        # Add this sentence to current chunk
        current_chunk.append(sentence)
        current_token_count += sentence_tokens
    
    # Don't forget the last chunk
    if current_chunk:
        yield " ".join(current_chunk)

def stream_chunk_text(text: str, max_tokens: int) -> Iterator[str]:
    """
    Stream-chunk text into slices with intelligent boundaries.
    Uses a generator to yield chunks without loading all at once.
    """
    if not text:
        return
    
    # Process text in segments for large documents to avoid loading all at once
    segment_size = 200000  # Process ~200K chars at a time
    
    # If text is small enough, use intelligent chunking directly
    if len(text) <= segment_size:
        yield from intelligent_chunk_text(text, max_tokens)
        return
    
    # Otherwise, process in segments
    segments = range(0, len(text), segment_size)
    overlap_chars = 5000  # Character overlap between segments to avoid cutting sentences
    
    prev_segment_end = 0
    for segment_idx, segment_start in enumerate(segments):
        # Include overlap with previous segment
        actual_start = max(0, segment_start - overlap_chars) if segment_idx > 0 else 0
        
        # Get segment end with overlap for next segment
        segment_end = min(segment_start + segment_size + overlap_chars, len(text))
        
        # Skip if we've already processed this text
        if actual_start >= prev_segment_end:
            segment = text[actual_start:segment_end]
            
            # Find a good breakpoint (end of paragraph or sentence) if this isn't the start
            if segment_idx > 0:
                # Look for paragraph break near our overlap point
                paragraph_break = segment.find('\n\n', segment_start - actual_start - 100)
                if paragraph_break != -1 and paragraph_break < segment_start - actual_start + 100:
                    # Found a paragraph break in our overlap zone, use it
                    segment = segment[paragraph_break + 2:]
                    actual_start = actual_start + paragraph_break + 2
            
            # Use intelligent chunking on this segment
            for chunk in intelligent_chunk_text(segment, max_tokens):
                yield chunk
                
            prev_segment_end = segment_end
            
        # Help garbage collection between segments
        segment = None
        gc.collect()

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

class EmbedderService:
    """Service for processing documents and generating embeddings."""
    
    def __init__(self):
        """Initialize the embedder service."""
        self.db = AsyncDatabaseConnector(DATABASE_URL)
        self.embedding_semaphore = asyncio.Semaphore(MAX_CONCURRENT_EMBEDDINGS)
    
    async def setup(self) -> bool:
        """Set up the service and database."""
        global EMBEDDING_DIMENSIONS
        
        # Test API dimensions first
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
        except Exception as e:
            logger.error(f"Failed to test embedding dimensions: {str(e)}")
            
        # Set up database with potentially updated dimensions
        db_setup = await self.db.setup_schema()
        if db_setup:
            logger.info(f"Using {EMBEDDING_DIMENSIONS} dimensions")
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
                
                # Insert into doc_chunks table with full text
                success = await self.db.insert_chunk(trans_id, chunk_id, i, chunk_text, embedding, metadata)
                
                if success:
                    successful_chunks += 1
                    logger.info(f"Successfully inserted chunk {i} in doc_chunks table")
                else:
                    failed_chunks += 1
                    logger.warning(f"Failed to insert chunk {i}")
                
                # Apply delay if configured
                delay_time = max(BATCH_DELAY_MS / 1000, 0.01)
                await asyncio.sleep(delay_time)
                    
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
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
        
        return successful_chunks, failed_chunks
    
    async def process_batch(self) -> None:
        """Process all pending translated documents with memory monitoring."""
        logger.info("Starting batch processing")
        
        # Connect to database
        if not await self.db.connect():
            logger.error("Failed to connect to database, aborting batch")
            return
        
        try:
            # ALWAYS completely clear all vector tables on each run
            logger.info("Starting with a fresh vector database - clearing ALL vector tables")
            if not await self.db.clear_vectors():
                logger.error("Failed to clear vector tables, aborting batch")
                return
            logger.info("Vector tables cleared successfully. Starting document processing.")
            
            # Track processing stats
            total_docs = 0
            processed_docs = 0
            total_chunks = 0
            failed_chunks = 0
            
            # Process documents one by one
            async with aiohttp.ClientSession() as session:
                async for doc in self.db.stream_translations():
                    total_docs += 1
                    successes, failures = await self.process_document(doc, session)
                    
                    if successes > 0:
                        processed_docs += 1
                    
                    total_chunks += successes
                    failed_chunks += failures
            
            logger.info(f"Batch complete: {processed_docs}/{total_docs} documents, {total_chunks} chunks created, {failed_chunks} chunks failed")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
        
        finally:
            # Close database connection
            await self.db.disconnect()

async def run_once() -> None:
    """Run the embedder service once."""
    service = EmbedderService()
    
    # Setup and run
    setup_success = await service.setup()
    
    if not setup_success:
        logger.error("Failed to set up embedder service")
        return
    
    await service.process_batch()
    
    # Final memory cleanup
    gc.collect()
    logger.info("Completed processing")

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory-optimized embedder service for vector embeddings")
    parser.add_argument("--memory-limit", type=int, default=MEMORY_LIMIT_MB, 
                        help=f"Memory limit in MB (default: {MEMORY_LIMIT_MB})")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT_EMBEDDINGS,
                        help=f"Maximum concurrent embedding operations (default: {MAX_CONCURRENT_EMBEDDINGS})")
    
    args = parser.parse_args()
    
    # Apply command line arguments
    MEMORY_LIMIT_MB = args.memory_limit
    MAX_CONCURRENT_EMBEDDINGS = args.concurrency
    
    logger.info(f"Using memory limit: {MEMORY_LIMIT_MB}MB, concurrency: {MAX_CONCURRENT_EMBEDDINGS}")
    
    # Run the embedder
    asyncio.run(run_once())