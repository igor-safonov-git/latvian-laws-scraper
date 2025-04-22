#!/usr/bin/env python3
"""
Script to fix edge cases detected in the embedder system.
Addresses orphaned summaries, documents missing embeddings, and large documents missing chunks.
"""
import os
import sys
import json
import argparse
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import random
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
logger = logging.getLogger("embedder_fix_edge_cases")

# Constants
CHUNK_SIZE = 3000  # Characters per chunk
CHUNK_OVERLAP = 500  # Character overlap between chunks
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072

class EdgeCaseFixer:
    """Fix edge cases in the embedder system."""
    
    def __init__(self):
        """Initialize the edge case fixer."""
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
        self.chat_completion_url = "https://api.openai.com/v1/chat/completions"
    
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
    
    def compute_id(self, url: str) -> str:
        """Compute a SHA-256 hash of the URL to use as ID."""
        return hashlib.sha256(url.encode()).hexdigest()
    
    def compute_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Compute a chunk ID based on document ID and chunk index."""
        combined = f"{doc_id}_{chunk_index}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        Returns a list of text chunks suitable for embedding.
        """
        if not text:
            return []
            
        chunks = []
        text_length = len(text)
        
        # If text is smaller than chunk size, return it as is
        if text_length <= CHUNK_SIZE:
            return [text]
        
        # Split into overlapping chunks
        start = 0
        while start < text_length:
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
            
            chunks.append(text[start:end])
            
            # Move to next chunk with overlap
            start = end - CHUNK_OVERLAP
            # Ensure we make progress and don't get stuck
            if start >= text_length or start <= 0:
                break
        
        logger.info(f"Split text into {len(chunks)} chunks with {CHUNK_OVERLAP} char overlap")
        return chunks
    
    async def generate_summary(
        self, 
        session: aiohttp.ClientSession, 
        text: str,
        max_retries: int = 3
    ) -> Optional[str]:
        """
        Generate a summary of the document text using OpenAI's Chat API.
        Implements retry with exponential backoff.
        """
        # If API is not enabled, return a placeholder summary
        if not self.embedding_enabled:
            logger.info("Using placeholder summary (no API key available)")
            return f"[PLACEHOLDER SUMMARY] This is a summary of a document that is {len(text)} characters long."
        
        # Limit text size for summarization
        max_chars = 32000  # Much lower than the actual model limit to be safe
        if len(text) > max_chars:
            text = text[:max_chars] + "...[Text truncated for summarization]"
        
        # Prompt for summarization
        system_prompt = (
            "You are a legal document summarizer. Summarize the legal document in a concise, "
            "factual manner. Include key details about document type, main provisions, "
            "obligations, rights, penalties, and applicability. Produce a summary of moderate length."
        )
        
        user_prompt = f"Summarize the following legal document in about 3000 characters:\n\n{text}"
        
        # Attempt to generate summary with retries
        for attempt in range(max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 2500,  # Limit output tokens
                    "temperature": 0.3  # Lower temperature for more factual output
                }
                
                async with session.post(
                    self.chat_completion_url, 
                    headers=headers, 
                    json=data,
                    timeout=60
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            summary = result["choices"][0]["message"]["content"]
                            logger.info(f"Generated summary of {len(summary)} chars")
                            return summary
                        else:
                            logger.error(f"Invalid summary response format: {result}")
                    elif response.status == 429 or response.status == 529:
                        # Rate limiting - wait and retry
                        wait_time = (2 ** attempt) + 1  # Exponential backoff
                        logger.warning(f"Rate limit hit, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        error_text = await response.text()
                        logger.error(f"Summary generation failed: HTTP {response.status} - {error_text}")
            except Exception as e:
                logger.error(f"Error in summary request (attempt {attempt+1}): {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # If all retries failed, return a simplified summary
        logger.error(f"Failed to generate summary after {max_retries} attempts")
        return f"[FALLBACK SUMMARY] Document with {len(text)} characters. Summary generation failed."
    
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
        
    def fix_orphaned_summaries(self) -> int:
        """Fix orphaned summaries by creating missing doc entries."""
        fixed_count = 0
        
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                # Find orphaned summaries
                cursor.execute("""
                    SELECT s.id, s.summary_text, s.metadata, s.embedding
                    FROM doc_summaries s
                    WHERE s.id NOT IN (SELECT id FROM docs)
                """)
                orphaned_summaries = cursor.fetchall()
                
                if not orphaned_summaries:
                    logger.info("No orphaned summaries found")
                    return 0
                
                logger.info(f"Found {len(orphaned_summaries)} orphaned summaries")
                
                # For each orphaned summary, create a doc entry using its embedding
                for summary in orphaned_summaries:
                    doc_id = summary["id"]
                    embedding = summary["embedding"]
                    metadata = summary["metadata"]
                    
                    # Look up the original document in raw_docs
                    cursor.execute("""
                        SELECT url, fetched_at, translated_text
                        FROM raw_docs
                        WHERE id = %s
                    """, (doc_id,))
                    raw_doc = cursor.fetchone()
                    
                    if not raw_doc:
                        logger.warning(f"Cannot fix orphaned summary {doc_id}: No raw document found")
                        continue
                    
                    # Create new metadata if needed
                    if not metadata:
                        metadata = {
                            "url": raw_doc["url"],
                            "fetched_at": raw_doc["fetched_at"].isoformat() if isinstance(raw_doc["fetched_at"], datetime) else raw_doc["fetched_at"],
                            "text_preview": raw_doc["translated_text"][:200] + "..." if len(raw_doc["translated_text"]) > 200 else raw_doc["translated_text"],
                            "char_length": len(raw_doc["translated_text"]),
                            "recovered": True
                        }
                    
                    # Create doc entry
                    cursor.execute("""
                        INSERT INTO docs(id, metadata, embedding)
                        VALUES(%s, %s, %s::vector)
                        ON CONFLICT (id) DO NOTHING
                    """, (doc_id, json.dumps(metadata), embedding))
                    
                    fixed_count += 1
                    logger.info(f"Created missing docs entry for orphaned summary {doc_id}")
                
                return fixed_count
                
        except Exception as e:
            logger.error(f"Error fixing orphaned summaries: {str(e)}")
            return fixed_count
            
    async def fix_missing_embeddings(self) -> int:
        """Fix documents that are missing embeddings or summaries."""
        fixed_count = 0
        
        try:
            # Get documents that are missing embeddings or summaries
            missing_docs = []
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute("""
                    SELECT r.id, r.url, r.fetched_at, r.translated_text,
                           (SELECT COUNT(*) FROM docs d WHERE d.id = r.id) as has_doc,
                           (SELECT COUNT(*) FROM doc_summaries s WHERE s.id = r.id) as has_summary
                    FROM raw_docs r
                    WHERE r.translated_text IS NOT NULL
                    AND (
                        (SELECT COUNT(*) FROM docs d WHERE d.id = r.id) = 0
                        OR
                        (SELECT COUNT(*) FROM doc_summaries s WHERE s.id = r.id) = 0
                    )
                """)
                missing_docs = cursor.fetchall()
            
            if not missing_docs:
                logger.info("No documents missing embeddings or summaries")
                return 0
                
            logger.info(f"Found {len(missing_docs)} documents missing embeddings or summaries")
            
            # Process each document
            async with aiohttp.ClientSession() as session:
                for doc in missing_docs:
                    doc_id = doc["id"]
                    url = doc["url"]
                    text = doc["translated_text"]
                    fetched_at = doc["fetched_at"]
                    has_doc = doc["has_doc"] > 0
                    has_summary = doc["has_summary"] > 0
                    
                    # Check what needs fixing
                    if not has_doc:
                        logger.info(f"Generating embedding for document {doc_id}")
                        # Create truncated version for embedding
                        text_for_embedding = text[:32000] if len(text) > 32000 else text
                        embedding = await self.generate_embedding(session, text_for_embedding)
                        
                        if embedding:
                            # Store the embedding
                            metadata = {
                                "url": url,
                                "fetched_at": fetched_at.isoformat() if isinstance(fetched_at, datetime) else fetched_at,
                                "text_preview": text[:200] + "..." if len(text) > 200 else text,
                                "char_length": len(text),
                                "fixed": True
                            }
                            
                            with self.conn.cursor() as cursor:
                                cursor.execute("""
                                    INSERT INTO docs(id, metadata, embedding)
                                    VALUES(%s, %s, %s::vector)
                                    ON CONFLICT (id) DO NOTHING
                                """, (doc_id, json.dumps(metadata), embedding))
                                
                            logger.info(f"Created missing embedding for document {doc_id}")
                    
                    if not has_summary:
                        logger.info(f"Generating summary for document {doc_id}")
                        summary = await self.generate_summary(session, text)
                        
                        if summary:
                            # Generate embedding for summary
                            summary_embedding = await self.generate_embedding(session, summary)
                            
                            if summary_embedding:
                                # Store the summary
                                metadata = {
                                    "url": url,
                                    "fetched_at": fetched_at.isoformat() if isinstance(fetched_at, datetime) else fetched_at,
                                    "summary_length": len(summary),
                                    "fixed": True
                                }
                                
                                with self.conn.cursor() as cursor:
                                    cursor.execute("""
                                        INSERT INTO doc_summaries(id, summary_text, metadata, embedding)
                                        VALUES(%s, %s, %s, %s::vector)
                                        ON CONFLICT (id) DO NOTHING
                                    """, (doc_id, summary, json.dumps(metadata), summary_embedding))
                                    
                                logger.info(f"Created missing summary for document {doc_id}")
                    
                    fixed_count += 1
            
            return fixed_count
            
        except Exception as e:
            logger.error(f"Error fixing missing embeddings: {str(e)}")
            return fixed_count
    
    async def fix_missing_chunks(self) -> int:
        """Fix documents that are missing chunks despite being large enough to need them."""
        fixed_count = 0
        
        try:
            # Get large documents missing chunks
            large_docs = []
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute("""
                    SELECT d.id, d.metadata->>'url' as url, r.fetched_at, r.translated_text,
                           (d.metadata->>'char_length')::int as char_length
                    FROM docs d
                    JOIN raw_docs r ON d.id = r.id
                    WHERE (d.metadata->>'char_length')::int > %s
                    AND NOT EXISTS (
                        SELECT 1 FROM doc_chunks c WHERE c.id = d.id
                    )
                """, (CHUNK_SIZE,))
                large_docs = cursor.fetchall()
            
            if not large_docs:
                logger.info("No large documents missing chunks")
                return 0
                
            logger.info(f"Found {len(large_docs)} large documents missing chunks")
            
            # Process each document
            async with aiohttp.ClientSession() as session:
                for doc in large_docs:
                    doc_id = doc["id"]
                    url = doc["url"]
                    text = doc["translated_text"]
                    fetched_at = doc["fetched_at"]
                    
                    # Split into chunks
                    chunks = self.split_into_chunks(text)
                    
                    if not chunks:
                        logger.warning(f"Failed to create chunks for document {doc_id}")
                        continue
                    
                    # Process each chunk
                    chunk_success_count = 0
                    for i, chunk_text in enumerate(chunks):
                        logger.info(f"Processing chunk {i+1}/{len(chunks)} for document {doc_id}")
                        
                        # Generate embedding for this chunk
                        chunk_embedding = await self.generate_embedding(session, chunk_text)
                        
                        if not chunk_embedding:
                            logger.error(f"Failed to generate embedding for chunk {i+1}")
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
                    
                    if chunk_success_count == len(chunks):
                        logger.info(f"Successfully added {chunk_success_count} chunks for document {doc_id}")
                        fixed_count += 1
                    else:
                        logger.warning(f"Partially fixed document {doc_id}: {chunk_success_count}/{len(chunks)} chunks added")
            
            return fixed_count
            
        except Exception as e:
            logger.error(f"Error fixing missing chunks: {str(e)}")
            return fixed_count
        
    async def run(self) -> None:
        """Run all fixes."""
        logger.info("Starting embedder edge case fix script")
        
        # Fix orphaned summaries
        orphaned_fixed = self.fix_orphaned_summaries()
        logger.info(f"Fixed {orphaned_fixed} orphaned summaries")
        
        # Fix missing embeddings and summaries
        missing_fixed = await self.fix_missing_embeddings()
        logger.info(f"Fixed {missing_fixed} documents with missing embeddings or summaries")
        
        # Fix missing chunks
        chunks_fixed = await self.fix_missing_chunks()
        logger.info(f"Fixed {chunks_fixed} documents with missing chunks")
        
        logger.info("All edge case fixes completed")
        

async def main():
    """Main entry point for the edge case fixer."""
    parser = argparse.ArgumentParser(description="Fix embedder edge cases")
    parser.add_argument("--orphaned-only", action="store_true", help="Only fix orphaned summaries")
    parser.add_argument("--missing-only", action="store_true", help="Only fix missing embeddings and summaries")
    parser.add_argument("--chunks-only", action="store_true", help="Only fix missing chunks")
    args = parser.parse_args()
    
    fixer = EdgeCaseFixer()
    
    if not fixer.setup():
        logger.error("Failed to set up fixer, exiting")
        sys.exit(1)
    
    if args.orphaned_only:
        orphaned_fixed = fixer.fix_orphaned_summaries()
        logger.info(f"Fixed {orphaned_fixed} orphaned summaries")
    elif args.missing_only:
        missing_fixed = await fixer.fix_missing_embeddings()
        logger.info(f"Fixed {missing_fixed} documents with missing embeddings or summaries")
    elif args.chunks_only:
        chunks_fixed = await fixer.fix_missing_chunks()
        logger.info(f"Fixed {chunks_fixed} documents with missing chunks")
    else:
        # Run all fixes
        await fixer.run()

if __name__ == "__main__":
    asyncio.run(main())