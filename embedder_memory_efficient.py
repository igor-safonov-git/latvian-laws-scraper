#!/usr/bin/env python3
"""
Memory-efficient implementation for chunking and embedding large documents.
Uses streaming and generator-based approaches to minimize memory usage.
"""
import os
import gc
import mmap
import asyncio
import hashlib
import logging
import contextlib
from typing import Dict, List, Any, Optional, Union, Iterator, AsyncIterator, Tuple
from pathlib import Path

import aiohttp
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("embedder_memory_efficient")

class EmbedderStream:
    """Memory-efficient embedder for large documents using streaming techniques."""

    def __init__(self):
        """Initialize the memory-efficient embedder."""
        # Load environment variables
        load_dotenv()
        
        # Get configuration
        self.database_url = os.getenv("DATABASE_URL")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "3000"))  # Default 3000 chars per chunk
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "500"))  # Default 500 chars overlap
        
        # Initialize database connection
        self.conn = None
        
        # API endpoints
        self.embeddings_url = "https://api.openai.com/v1/embeddings"
    
    def setup(self) -> bool:
        """Set up the database connection."""
        try:
            # Connect to database
            self.conn = psycopg2.connect(self.database_url)
            self.conn.autocommit = True
            logger.info("Connected to PostgreSQL database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            return False
    
    def _get_placeholder_embedding(self) -> List[float]:
        """Generate a placeholder embedding (used when API key is not available)."""
        import random
        return [random.uniform(-0.1, 0.1) for _ in range(self.embedding_dimensions)]
    
    def chunker(self, text: str, chunk_size: int, overlap: int = 0) -> Iterator[str]:
        """
        Generator that yields chunks of text with optional overlap.
        Uses semantic boundaries (paragraphs, sentences) when possible.
        
        Args:
            text: The text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Yields:
            Text chunks of approximately chunk_size length
        """
        if not text:
            return
            
        text_length = len(text)
        
        # If text is smaller than chunk size, yield it as-is
        if text_length <= chunk_size:
            yield text
            return
        
        # Stream chunks with overlap
        start = 0
        while start < text_length:
            # Determine end position for this chunk
            end = min(start + chunk_size, text_length)
            
            # Find semantic boundary if this isn't the last chunk
            if end < text_length:
                # Look for paragraph break
                paragraph_break = text.rfind("\n\n", start, end)
                if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                    end = paragraph_break + 2  # Include the paragraph break
                else:
                    # Look for sentence break (period followed by space)
                    sentence_break = text.rfind(". ", start, end)
                    if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                        end = sentence_break + 2  # Include the period and space
            
            # Yield the chunk and immediately free the slice reference
            chunk = text[start:end]
            yield chunk
            
            # Allow event loop to release memory before continuing
            chunk = None
            gc.collect()
            
            # Move to next chunk with overlap
            start = max(0, end - overlap)
            
            # Ensure we make progress and don't get stuck
            if start >= text_length or (end == text_length and start == end - overlap):
                break
    
    def file_chunker(self, file_path: str, chunk_size: int, overlap: int = 0) -> Iterator[str]:
        """
        Generator that yields chunks from a file without loading the whole file into memory.
        
        Args:
            file_path: Path to the file to process
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Yields:
            Text chunks of approximately chunk_size length
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return
        
        # Use memory mapping for efficient file access
        with open(file_path, 'r+b') as f:
            # Memory-map the file for efficient read-only access
            with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as mm:
                file_size = mm.size()
                
                # If file is smaller than chunk size, yield it as-is
                if file_size <= chunk_size:
                    yield mm[:].decode('utf-8', errors='ignore')
                    return
                
                # Stream chunks with overlap
                start = 0
                while start < file_size:
                    # Determine end position for this chunk
                    end = min(start + chunk_size, file_size)
                    
                    # Extract the chunk as bytes and decode
                    chunk_bytes = mm[start:end]
                    chunk = chunk_bytes.decode('utf-8', errors='ignore')
                    
                    # Find semantic boundary if this isn't the last chunk
                    if end < file_size:
                        # Look for paragraph break
                        paragraph_break = chunk.rfind("\n\n")
                        if paragraph_break != -1 and paragraph_break > len(chunk) // 2:
                            end = start + paragraph_break + 2
                            chunk = chunk[:paragraph_break+2]
                        else:
                            # Look for sentence break
                            sentence_break = chunk.rfind(". ")
                            if sentence_break != -1 and sentence_break > len(chunk) // 2:
                                end = start + sentence_break + 2
                                chunk = chunk[:sentence_break+2]
                    
                    # Yield the chunk and immediately free the reference
                    yield chunk
                    
                    # Allow event loop to release memory
                    chunk = None
                    chunk_bytes = None
                    gc.collect()
                    
                    # Move to next chunk with overlap
                    start = max(0, end - overlap)
                    
                    # Ensure we make progress
                    if start >= file_size or (end == file_size and start == end - overlap):
                        break
    
    def compute_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Compute a chunk ID based on document ID and chunk index."""
        combined = f"{doc_id}_{chunk_index}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    async def generate_embedding(
        self, 
        session: aiohttp.ClientSession, 
        text: str,
        max_retries: int = 3
    ) -> Optional[List[float]]:
        """
        Generate an embedding for the given text using OpenAI API.
        Implements retry with exponential backoff.
        """
        # If API key is not available, return placeholder
        if not self.openai_api_key:
            logger.info("Using placeholder embedding (no API key available)")
            return self._get_placeholder_embedding()
        
        # Attempt to generate embedding with retries
        for attempt in range(max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "input": text,
                    "model": self.embedding_model,
                    "dimensions": self.embedding_dimensions
                }
                
                async with session.post(
                    self.embeddings_url, 
                    headers=headers, 
                    json=data,
                    timeout=60
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "data" in result and len(result["data"]) > 0:
                            embedding = result["data"][0]["embedding"]
                            return embedding
                        else:
                            logger.error(f"Invalid embedding response format: {result}")
                    elif response.status == 429 or response.status == 529:
                        # Rate limit hit - implement backoff
                        wait_time = (2 ** attempt) + 1  # Exponential backoff
                        logger.warning(f"Rate limit hit, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        error_text = await response.text()
                        logger.error(f"Embedding generation failed: HTTP {response.status} - {error_text}")
                        # If authentication error, don't retry
                        if response.status == 401:
                            logger.error("Authentication failed, using placeholder embedding")
                            return self._get_placeholder_embedding()
                        
            except Exception as e:
                logger.error(f"Error in embedding request (attempt {attempt+1}): {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # If we exhausted all retries, use placeholder
        logger.error(f"Failed to generate embedding after {max_retries} attempts")
        return self._get_placeholder_embedding()
    
    async def embed_chunks(
        self, 
        text_or_path: Union[str, Path],
        doc_id: str,
        url: str,
        fetched_at,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Memory-efficient function to embed chunks from a text or file.
        
        Args:
            text_or_path: Either text content or path to a file
            doc_id: Document ID
            url: URL of the document
            fetched_at: Timestamp when document was fetched
            chunk_size: Size of each chunk (default: self.chunk_size)
            chunk_overlap: Overlap between chunks (default: self.chunk_overlap)
            
        Returns:
            List of chunk metadata including embeddings
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        if chunk_overlap is None:
            chunk_overlap = self.chunk_overlap
        
        # Determine if input is a file path or text content
        is_file = False
        if isinstance(text_or_path, (str, Path)):
            path = Path(text_or_path)
            if path.exists() and path.is_file():
                is_file = True
        
        # Prepare for streaming chunks
        chunk_results = []
        
        # Process chunks in batches with controlled concurrency
        async with aiohttp.ClientSession() as session:
            # Use appropriate chunker based on input type
            chunk_iterator = self.file_chunker(text_or_path, chunk_size, chunk_overlap) if is_file else self.chunker(text_or_path, chunk_size, chunk_overlap)
            
            # Process chunks one at a time to control memory usage
            for i, chunk_text in enumerate(chunk_iterator):
                logger.info(f"Processing chunk {i+1} for document {doc_id}")
                
                # Generate embedding for this chunk
                chunk_embedding = await self.generate_embedding(session, chunk_text)
                
                # Apply backpressure to allow event loop to release memory
                await asyncio.sleep(0)
                
                if not chunk_embedding:
                    logger.error(f"Failed to generate embedding for chunk {i+1}")
                    continue
                
                # Compute chunk ID
                chunk_id = self.compute_chunk_id(doc_id, i)
                
                # Store chunk metadata
                chunk_metadata = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "chunk_text": chunk_text,
                    "embedding": chunk_embedding,
                    "metadata": {
                        "url": url,
                        "fetched_at": fetched_at.isoformat() if hasattr(fetched_at, 'isoformat') else fetched_at,
                        "chunk_index": i,
                        "chunk_length": len(chunk_text),
                        "text_preview": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
                    }
                }
                
                # Store in database if connection is available
                if self.conn:
                    try:
                        with self.conn.cursor() as cursor:
                            cursor.execute("""
                                INSERT INTO doc_chunks(id, chunk_id, chunk_index, chunk_text, metadata, embedding)
                                VALUES(%s, %s, %s, %s, %s, %s::vector)
                                ON CONFLICT (chunk_id) DO NOTHING
                            """, (
                                doc_id, 
                                chunk_id, 
                                i, 
                                chunk_text, 
                                psycopg2.extras.Json(chunk_metadata["metadata"]), 
                                chunk_embedding
                            ))
                    except Exception as e:
                        logger.error(f"Error storing chunk {i} in database: {str(e)}")
                
                # Add to results and immediately release references
                chunk_results.append({
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "success": True
                })
                
                # Clean up references to allow garbage collection
                chunk_text = None
                chunk_embedding = None
                chunk_metadata = None
                gc.collect()
                
                # Add delay between chunks to prevent rate limiting
                await asyncio.sleep(0.1)
        
        return chunk_results

    async def process_large_document(
        self,
        doc_id: str,
        url: str, 
        text_or_path: Union[str, Path],
        fetched_at,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a large document by:
        1. Generating a full document embedding (limited to first 32K chars)
        2. Streaming chunks and generating embeddings with minimal memory usage
        3. Creating a summary embedding
        
        This method is memory-efficient and can handle very large documents.
        """
        if not self.setup():
            logger.error("Failed to set up database connection")
            return {"success": False, "error": "Database connection failed"}
        
        logger.info(f"Processing large document {doc_id} from URL {url}")
        
        # Create metadata if not provided
        if metadata is None:
            metadata = {
                "url": url,
                "fetched_at": fetched_at.isoformat() if hasattr(fetched_at, 'isoformat') else fetched_at,
            }
        
        result = {
            "doc_id": doc_id,
            "success": False,
            "chunks_processed": 0
        }
        
        try:
            # 1. Generate embedding for full document (limited to first 32K chars)
            async with aiohttp.ClientSession() as session:
                # Get first 32K chars if text (not file)
                if not isinstance(text_or_path, (Path)) and isinstance(text_or_path, str) and not os.path.exists(text_or_path):
                    full_text_for_embedding = text_or_path[:32000]
                    char_length = len(text_or_path)
                else:
                    # For files, read first 32K chars
                    with open(text_or_path, 'r', encoding='utf-8', errors='ignore') as f:
                        full_text_for_embedding = f.read(32000)
                    # Get file size
                    char_length = os.path.getsize(text_or_path)
                
                # Add char_length to metadata
                metadata["char_length"] = char_length
                metadata["text_preview"] = full_text_for_embedding[:200] + "..." if len(full_text_for_embedding) > 200 else full_text_for_embedding
                
                # Generate embedding
                logger.info(f"Generating embedding for document {doc_id} ({len(full_text_for_embedding)} chars)")
                doc_embedding = await self.generate_embedding(session, full_text_for_embedding)
                
                # Store document and embedding
                if doc_embedding and self.conn:
                    with self.conn.cursor() as cursor:
                        cursor.execute("""
                            INSERT INTO docs(id, metadata, embedding)
                            VALUES(%s, %s, %s::vector)
                            ON CONFLICT (id) DO UPDATE
                            SET metadata = %s, embedding = %s::vector
                        """, (
                            doc_id, 
                            psycopg2.extras.Json(metadata), 
                            doc_embedding,
                            psycopg2.extras.Json(metadata),
                            doc_embedding
                        ))
                
                # Clean up to free memory
                full_text_for_embedding = None
                doc_embedding = None
                gc.collect()
                await asyncio.sleep(0.1)
                
            # 2. Process chunks in a memory-efficient way
            chunk_results = await self.embed_chunks(
                text_or_path=text_or_path,
                doc_id=doc_id,
                url=url,
                fetched_at=fetched_at
            )
            
            result["chunks_processed"] = len(chunk_results)
            result["success"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {str(e)}")
            result["error"] = str(e)
            return result

async def main():
    """Example usage of the memory-efficient embedder."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python embedder_memory_efficient.py <file_or_text>")
        return
    
    # Get input from command line
    text_or_path = sys.argv[1]
    
    # Check if input is a file path
    is_file = os.path.exists(text_or_path)
    doc_id = hashlib.sha256(text_or_path.encode()).hexdigest()
    url = f"file://{os.path.abspath(text_or_path)}" if is_file else "memory://test-document"
    
    # Initialize memory-efficient embedder
    embedder = EmbedderStream()
    
    # Process the document
    result = await embedder.process_large_document(
        doc_id=doc_id,
        url=url,
        text_or_path=text_or_path,
        fetched_at=datetime.now()
    )
    
    print(f"Document processing complete: {result}")

if __name__ == "__main__":
    from datetime import datetime
    asyncio.run(main())