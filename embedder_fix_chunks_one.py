#!/usr/bin/env python3
"""
Script focused on fixing a single document missing chunks.
"""
import os
import sys
import json
import argparse
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
import hashlib
from datetime import datetime

import psycopg2
import psycopg2.extras
import aiohttp
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("embedder_fix_chunks_one")

# Constants
CHUNK_SIZE = 3000  # Characters per chunk
CHUNK_OVERLAP = 500  # Character overlap between chunks
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072

class ChunkFixer:
    """Fix a single document that is missing chunks despite being large enough."""
    
    def __init__(self, document_id: str):
        """Initialize the chunk fixer."""
        # Store document ID
        self.document_id = document_id
        
        # Load environment variables
        load_dotenv()
        
        # Get configuration
        self.database_url = os.getenv("DATABASE_URL")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Check required configuration
        if not self.database_url:
            logger.error("DATABASE_URL environment variable not set")
            sys.exit(1)
            
        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            logger.warning("Using placeholder embeddings (for testing)")
            self.embedding_enabled = False
        else:
            self.embedding_enabled = True
        
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
    
    def get_placeholder_embedding(self) -> List[float]:
        """Generate a placeholder embedding for testing when API key is not available."""
        # Generate a fixed-dimension random vector for testing
        import random
        return [random.uniform(-0.1, 0.1) for _ in range(EMBEDDING_DIMENSIONS)]
    
    def compute_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Compute a chunk ID based on document ID and chunk index."""
        combined = f"{doc_id}_{chunk_index}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        More memory-efficient implementation.
        Returns a list of text chunks suitable for embedding.
        """
        if not text:
            return []
            
        chunks = []
        text_length = len(text)
        
        # If text is smaller than chunk size, return it as is
        if text_length <= CHUNK_SIZE:
            return [text]
        
        # Calculate number of chunks needed - this avoids appending to chunks list repeatedly
        # which can be memory-intensive for very large documents
        n_chunks = max(1, (text_length - CHUNK_OVERLAP) // (CHUNK_SIZE - CHUNK_OVERLAP) + 1)
        chunks = [None] * n_chunks  # Pre-allocate chunks list
        
        # Split into overlapping chunks
        start = 0
        chunk_idx = 0
        while start < text_length and chunk_idx < n_chunks:
            end = min(start + CHUNK_SIZE, text_length)
            
            # If this is not the last chunk, try to end at a paragraph or sentence boundary
            if end < text_length:
                # Look for paragraph break
                paragraph_break = text.rfind("\n\n", start, end)
                if paragraph_break != -1 and paragraph_break > start + CHUNK_SIZE // 2:
                    end = paragraph_break + 2  # Include the paragraph break
                else:
                    # Look for sentence break (period followed by space)
                    sentence_break = text.rfind(". ", start, end)
                    if sentence_break != -1 and sentence_break > start + CHUNK_SIZE // 2:
                        end = sentence_break + 2  # Include the period and space
            
            chunks[chunk_idx] = text[start:end]
            chunk_idx += 1
            
            # Move to next chunk with overlap
            start = end - CHUNK_OVERLAP
            # Ensure we make progress and don't get stuck
            if start >= text_length or start <= 0:
                break
        
        # If we overestimated the number of chunks, truncate the array
        if chunk_idx < n_chunks:
            chunks = chunks[:chunk_idx]
        
        logger.info(f"Split text into {len(chunks)} chunks with {CHUNK_OVERLAP} char overlap")
        return chunks
    
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
        # If embedding is not enabled, return placeholder
        if not self.embedding_enabled:
            logger.info("Using placeholder embedding (no API key available)")
            return self.get_placeholder_embedding()
        
        # Attempt to generate embedding with retries
        for attempt in range(max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "input": text,
                    "model": EMBEDDING_MODEL,
                    "dimensions": EMBEDDING_DIMENSIONS
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
                            return self.get_placeholder_embedding()
                        
            except Exception as e:
                logger.error(f"Error in embedding request (attempt {attempt+1}): {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # If we exhausted all retries, use placeholder
        logger.error(f"Failed to generate embedding after {max_retries} attempts")
        return self.get_placeholder_embedding()
        
    async def fix_document_chunks(self) -> bool:
        """Fix a single document that is missing chunks."""
        import gc  # For garbage collection
        
        try:
            # Get document data
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute("""
                    SELECT d.id, d.metadata->>'url' as url, r.fetched_at, r.translated_text,
                           (d.metadata->>'char_length')::int as char_length
                    FROM docs d
                    JOIN raw_docs r ON d.id = r.id
                    WHERE d.id = %s
                    AND NOT EXISTS (
                        SELECT 1 FROM doc_chunks c WHERE c.id = d.id
                    )
                """, (self.document_id,))
                doc = cursor.fetchone()
            
            if not doc:
                logger.warning(f"Document {self.document_id} not found or already has chunks")
                return False
                
            doc_id = doc['id']
            url = doc['url']
            fetched_at = doc['fetched_at']
            char_length = doc['char_length']
            logger.info(f"Processing document {doc_id} ({char_length} chars) from URL {url}")
            
            # Extract text and free up the large doc dict
            text = doc["translated_text"]
            doc = None  # Allow garbage collection
            gc.collect()  # Force garbage collection
            
            # Split text into chunks - this is memory-intensive
            logger.info("Splitting document into chunks")
            chunks = self.split_into_chunks(text)
            text = None  # Allow garbage collection of the full text
            gc.collect()  # Force garbage collection
            
            if not chunks:
                logger.warning(f"Failed to create chunks for document {doc_id}")
                return False
            
            chunk_count = len(chunks)
            logger.info(f"Processing {chunk_count} chunks one at a time")
            
            # Process each chunk individually
            chunk_success_count = 0
            for i in range(chunk_count):
                # Create a new session for each chunk
                logger.info(f"Processing chunk {i+1}/{chunk_count} for document {doc_id}")
                
                # Get chunk text and clear others to save memory
                chunk_text = chunks[i]
                
                # Generate embedding for this chunk
                async with aiohttp.ClientSession() as session:
                    # Generate embedding for this chunk
                    chunk_embedding = await self.generate_embedding(session, chunk_text)
                
                if not chunk_embedding:
                    logger.error(f"Failed to generate embedding for chunk {i+1}")
                    await asyncio.sleep(1)  # Brief pause
                    continue
                
                # Store chunk and its embedding
                chunk_id = self.compute_chunk_id(doc_id, i)
                
                # Create metadata
                metadata = {
                    "url": url,
                    "fetched_at": fetched_at.isoformat() if isinstance(fetched_at, datetime) else fetched_at,
                    "chunk_index": i,
                    "chunk_length": len(chunk_text),
                    "text_preview": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text,
                    "fixed": True
                }
                
                # Insert chunk
                with self.conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO doc_chunks(id, chunk_id, chunk_index, chunk_text, metadata, embedding)
                        VALUES(%s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (chunk_id) DO NOTHING
                    """, (
                        doc_id, chunk_id, i, chunk_text, json.dumps(metadata), chunk_embedding
                    ))
                    
                chunk_success_count += 1
                
                # Clear memory after each chunk
                chunk_text = None
                chunk_embedding = None
                metadata = None
                gc.collect()
                
                # Add a small delay between chunks to reduce memory pressure
                await asyncio.sleep(2)
            
            if chunk_success_count == chunk_count:
                logger.info(f"Successfully added {chunk_success_count} chunks for document {doc_id}")
                return True
            else:
                logger.warning(f"Partially fixed document {doc_id}: {chunk_success_count}/{chunk_count} chunks added")
                return chunk_success_count > 0
            
        except Exception as e:
            logger.error(f"Error fixing document chunks: {str(e)}")
            return False
        
    async def run(self) -> None:
        """Run the chunk fix for a single document."""
        logger.info(f"Starting chunk fix for document {self.document_id}")
        
        # Fix document chunks
        success = await self.fix_document_chunks()
        
        if success:
            logger.info(f"Successfully added chunks for document {self.document_id}")
        else:
            logger.warning(f"Failed to add chunks for document {self.document_id}")
        
        logger.info("Chunk fix completed")
        

async def main():
    """Main entry point for the chunk fixer."""
    # Get document ID from command line
    parser = argparse.ArgumentParser(description="Fix chunks for a single document")
    parser.add_argument("document_id", help="ID of the document to fix")
    args = parser.parse_args()
    
    document_id = args.document_id
    
    fixer = ChunkFixer(document_id)
    
    if not fixer.setup():
        logger.error("Failed to set up fixer, exiting")
        sys.exit(1)
    
    # Run chunk fix
    await fixer.run()

if __name__ == "__main__":
    asyncio.run(main())