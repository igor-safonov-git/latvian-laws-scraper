#!/usr/bin/env python3
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import aiohttp
import psycopg2
import psycopg2.extras
import pytz
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("embedder_enhanced")

class EnhancedEmbedder:
    """Enhanced embedder service with document chunking and summarization."""

    def __init__(self):
        """Initialize the enhanced embedder service."""
        # Load environment variables
        load_dotenv()
        
        # Get configuration
        self.database_url = os.getenv("DATABASE_URL")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "8192"))  # Default 8192 tokens
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "3000"))  # Default 3000 chars per chunk
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "500"))  # Default 500 chars overlap
        self.summary_length = int(os.getenv("SUMMARY_LENGTH", "3000"))  # Default 3000 chars summary
        
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
        
        # Ensure logs directory exists
        self.logs_dir = Path("./logs")
        self.logs_dir.mkdir(exist_ok=True)
        self.log_file = self.logs_dir / "embedder.log"
        
        # API endpoints
        self.embeddings_url = "https://api.openai.com/v1/embeddings"
        self.chat_completion_url = "https://api.openai.com/v1/chat/completions"
        
        # API verification status
        self.api_verified = False
        
    async def verify_api_key(self, session: aiohttp.ClientSession) -> bool:
        """Verify the OpenAI API key with a small test request."""
        if not self.embedding_enabled:
            logger.warning("API verification skipped - embedding not enabled")
            return False
            
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "input": "This is a test.",
                "model": self.embedding_model,
                "dimensions": self.embedding_dimensions
            }
            
            async with session.post(
                self.embeddings_url, 
                headers=headers, 
                json=data,
                timeout=10
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if "data" in result and len(result["data"]) > 0:
                        logger.info(f"OpenAI API key valid. Embedding model: {self.embedding_model}")
                        return True
                    else:
                        logger.error(f"Invalid embedding response format: {result}")
                        return False
                else:
                    error_text = await response.text()
                    logger.error(f"OpenAI API verification failed: HTTP {response.status} - {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Error verifying OpenAI API key: {str(e)}")
            return False

    def setup(self) -> bool:
        """Set up the necessary database connections and tables."""
        try:
            # Connect to database
            self.conn = psycopg2.connect(self.database_url)
            self.conn.autocommit = True
            logger.info("Connected to PostgreSQL database")
            
            # Create the vector extension if it doesn't exist
            with self.conn.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                logger.info("Enabled vector extension")
                
                # Create the docs table if it doesn't exist
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS docs (
                        id TEXT PRIMARY KEY,
                        metadata JSONB,
                        embedding VECTOR({self.embedding_dimensions})
                    )
                """)
                logger.info("Created docs table")
                
                # Create the doc_chunks table if it doesn't exist
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS doc_chunks (
                        id TEXT NOT NULL,
                        chunk_id TEXT PRIMARY KEY,
                        chunk_index INTEGER NOT NULL,
                        chunk_text TEXT NOT NULL,
                        metadata JSONB,
                        embedding VECTOR({self.embedding_dimensions})
                    )
                """)
                logger.info("Created doc_chunks table")
                
                # Create the doc_summaries table if it doesn't exist
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS doc_summaries (
                        id TEXT PRIMARY KEY,
                        summary_text TEXT NOT NULL,
                        metadata JSONB,
                        embedding VECTOR({self.embedding_dimensions})
                    )
                """)
                logger.info("Created doc_summaries table")
                
                # Create index on doc_chunks.id for faster lookups
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_doc_chunks_id ON doc_chunks (id)
                """)
                logger.info("Created index on doc_chunks.id")
            
            return True
        except Exception as e:
            logger.error(f"Failed to set up database: {str(e)}")
            return False
            
    def check_new_translations(self) -> Tuple[bool, int]:
        """
        Check if new translations are available since last embedder run.
        Returns a tuple of (new_translations_available, total_translated_count).
        """
        try:
            with self.conn.cursor() as cursor:
                # Check if raw_docs table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'raw_docs'
                    )
                """)
                if not cursor.fetchone()[0]:
                    logger.error("raw_docs table does not exist")
                    return False, 0
                
                # Check for untranslated documents
                cursor.execute("""
                    SELECT COUNT(*) FROM raw_docs 
                    WHERE processed = FALSE OR translated_text IS NULL
                """)
                untranslated_count = cursor.fetchone()[0]
                
                if untranslated_count > 0:
                    logger.warning(f"Found {untranslated_count} untranslated documents")
                    return False, 0
                
                # Check if there are any documents at all
                cursor.execute("SELECT COUNT(*) FROM raw_docs")
                total_count = cursor.fetchone()[0]
                
                if total_count == 0:
                    logger.warning("No documents found in raw_docs table")
                    return False, 0
                
                # Check if docs table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'docs'
                    )
                """)
                docs_table_exists = cursor.fetchone()[0]
                
                # If docs table doesn't exist, we need to generate all embeddings
                if not docs_table_exists:
                    logger.info(f"First run: {total_count} documents need embeddings")
                    return True, total_count
                
                # Check if doc_chunks and doc_summaries tables exist
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'doc_chunks'
                    ),
                    EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'doc_summaries'
                    )
                """)
                tables_exist = cursor.fetchone()
                
                # If new tables don't exist, we need to regenerate all embeddings
                if not tables_exist[0] or not tables_exist[1]:
                    logger.info(f"New tables added: {total_count} documents need embeddings")
                    return True, total_count
                
                # Check the count of documents in docs and doc_summaries table
                cursor.execute("""
                    SELECT COUNT(*) FROM docs
                """)
                embedding_count = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT COUNT(*) FROM doc_summaries
                """)
                summary_count = cursor.fetchone()[0]
                
                # Check if all documents have summaries
                cursor.execute("""
                    SELECT COUNT(*) FROM raw_docs
                    WHERE id NOT IN (SELECT id FROM doc_summaries)
                    AND translated_text IS NOT NULL
                """)
                missing_summaries = cursor.fetchone()[0]
                
                # If counts don't match, we need to update
                if embedding_count != total_count or summary_count != total_count or missing_summaries > 0:
                    logger.info(f"Coverage mismatch: {total_count} translated, {embedding_count} with embeddings, {summary_count} with summaries")
                    return True, total_count
                
                # Check for modified translations (processed flag should be false for modified docs)
                cursor.execute("""
                    SELECT MAX(fetched_at) FROM raw_docs
                """)
                latest_translation = cursor.fetchone()[0]
                
                if latest_translation:
                    # Check if we have a record of the last embedding run
                    try:
                        cursor.execute("""
                            SELECT metadata->>'last_embedding_run' FROM system_info LIMIT 1
                        """)
                        result = cursor.fetchone()
                        if result and result[0]:
                            last_run = datetime.fromisoformat(result[0])
                            if latest_translation > last_run:
                                logger.info(f"New translations since last run: latest at {latest_translation}")
                                return True, total_count
                    except Exception:
                        # system_info table might not exist yet
                        pass
                
                logger.info(f"No new translations found: {total_count} documents already have embeddings")
                return False, total_count
                
        except Exception as e:
            logger.error(f"Error checking for new translations: {str(e)}")
            return False, 0
            
    def update_last_run_timestamp(self) -> None:
        """Record the timestamp of the last successful embedding run."""
        try:
            now = datetime.now(pytz.UTC)
            
            with self.conn.cursor() as cursor:
                # Create system_info table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_info (
                        id TEXT PRIMARY KEY,
                        metadata JSONB
                    )
                """)
                
                # Update or insert the last run timestamp
                cursor.execute("""
                    INSERT INTO system_info (id, metadata)
                    VALUES ('embedder', jsonb_build_object('last_embedding_run', %s::text))
                    ON CONFLICT (id) DO UPDATE
                    SET metadata = jsonb_set(
                        COALESCE(system_info.metadata, '{}'::jsonb),
                        '{last_embedding_run}',
                        to_jsonb(%s::text)
                    )
                """, (now.isoformat(), now.isoformat()))
                
                logger.info(f"Updated last embedding run timestamp to {now.isoformat()}")
        except Exception as e:
            logger.error(f"Error updating last run timestamp: {str(e)}")
            
    def get_translated_documents(self) -> List[Dict[str, Any]]:
        """Get all translated documents from the database."""
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute("""
                    SELECT id, url, fetched_at, translated_text
                    FROM raw_docs
                    WHERE translated_text IS NOT NULL
                """)
                
                # Convert to list of dicts
                docs = [dict(row) for row in cursor.fetchall()]
                logger.info(f"Found {len(docs)} translated documents")
                return docs
        except Exception as e:
            logger.error(f"Error fetching translated documents: {str(e)}")
            return []
    
    def clear_embeddings(self) -> bool:
        """Clear all existing embeddings from the docs, doc_chunks, and doc_summaries tables."""
        try:
            with self.conn.cursor() as cursor:
                # Clear main embeddings
                cursor.execute("DELETE FROM docs")
                docs_count = cursor.rowcount
                
                # Clear chunks if table exists
                try:
                    cursor.execute("DELETE FROM doc_chunks")
                    chunks_count = cursor.rowcount
                except:
                    chunks_count = 0
                
                # Clear summaries if table exists
                try:
                    cursor.execute("DELETE FROM doc_summaries")
                    summaries_count = cursor.rowcount
                except:
                    summaries_count = 0
                
                logger.info(f"Cleared {docs_count} document embeddings, {chunks_count} chunks, and {summaries_count} summaries")
                return True
        except Exception as e:
            logger.error(f"Error clearing embeddings: {str(e)}")
            return False
    
    def compute_id(self, url: str) -> str:
        """Compute a SHA-256 hash of the URL to use as ID."""
        return hashlib.sha256(url.encode()).hexdigest()
    
    def compute_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Compute a chunk ID based on document ID and chunk index."""
        combined = f"{doc_id}_{chunk_index}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get_placeholder_embedding(self) -> List[float]:
        """Generate a placeholder embedding for testing when API key is not available."""
        # Generate a fixed-dimension random vector for testing
        import random
        return [random.uniform(-0.1, 0.1) for _ in range(self.embedding_dimensions)]
    
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
        if text_length <= self.chunk_size:
            return [text]
        
        # Split into overlapping chunks
        start = 0
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # If this is not the last chunk, try to end at a paragraph or sentence boundary
            if end < text_length:
                # Look for paragraph break
                paragraph_break = text.rfind("\n\n", start, end)
                if paragraph_break != -1 and paragraph_break > start + self.chunk_size // 2:
                    end = paragraph_break + 2  # Include the paragraph break
                else:
                    # Look for sentence break (period followed by space)
                    sentence_break = text.rfind(". ", start, end)
                    if sentence_break != -1 and sentence_break > start + self.chunk_size // 2:
                        end = sentence_break + 2  # Include the period and space
            
            chunks.append(text[start:end])
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            # Ensure we make progress and don't get stuck
            if start >= text_length or start <= 0:
                break
        
        logger.info(f"Split text into {len(chunks)} chunks with {self.chunk_overlap} char overlap")
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
        
        user_prompt = f"Summarize the following legal document in about {self.summary_length} characters:\n\n{text}"
        
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
                            return self.get_placeholder_embedding()
                        
            except Exception as e:
                logger.error(f"Error in embedding request (attempt {attempt+1}): {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # If we exhausted all retries, use placeholder
        logger.error(f"Failed to generate embedding after {max_retries} attempts")
        return self.get_placeholder_embedding()
    
    def store_embedding(
        self, 
        doc_id: str, 
        url: str, 
        fetched_at: datetime, 
        text: str,
        embedding: List[float]
    ) -> bool:
        """Store document embedding in the database."""
        try:
            # Create metadata JSON
            metadata = {
                "url": url,
                "fetched_at": fetched_at.isoformat() if isinstance(fetched_at, datetime) else fetched_at,
                "text_preview": text[:200] + "..." if len(text) > 200 else text,
                "char_length": len(text)
            }
            
            # Insert into database
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO docs(id, metadata, embedding)
                    VALUES(%s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE 
                    SET metadata = %s, embedding = %s::vector
                """, (doc_id, json.dumps(metadata), embedding, json.dumps(metadata), embedding))
                
            return True
        except Exception as e:
            logger.error(f"Error storing embedding for {doc_id}: {str(e)}")
            return False
    
    def store_summary(
        self,
        doc_id: str,
        url: str,
        fetched_at: datetime,
        summary_text: str,
        embedding: List[float]
    ) -> bool:
        """Store document summary in the database."""
        try:
            # Create metadata JSON
            metadata = {
                "url": url,
                "fetched_at": fetched_at.isoformat() if isinstance(fetched_at, datetime) else fetched_at,
                "summary_length": len(summary_text)
            }
            
            # Insert into database
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO doc_summaries(id, summary_text, metadata, embedding)
                    VALUES(%s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE 
                    SET summary_text = %s, metadata = %s, embedding = %s::vector
                """, (
                    doc_id, summary_text, json.dumps(metadata), embedding,
                    summary_text, json.dumps(metadata), embedding
                ))
                
            return True
        except Exception as e:
            logger.error(f"Error storing summary for {doc_id}: {str(e)}")
            return False
    
    def store_chunk(
        self,
        doc_id: str,
        chunk_index: int,
        chunk_text: str,
        url: str,
        fetched_at: datetime,
        embedding: List[float]
    ) -> bool:
        """Store document chunk in the database."""
        try:
            # Create chunk ID
            chunk_id = self.compute_chunk_id(doc_id, chunk_index)
            
            # Create metadata JSON
            metadata = {
                "url": url,
                "fetched_at": fetched_at.isoformat() if isinstance(fetched_at, datetime) else fetched_at,
                "chunk_index": chunk_index,
                "chunk_length": len(chunk_text),
                "text_preview": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
            }
            
            # Insert into database
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO doc_chunks(id, chunk_id, chunk_index, chunk_text, metadata, embedding)
                    VALUES(%s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (chunk_id) DO UPDATE 
                    SET chunk_text = %s, metadata = %s, embedding = %s::vector
                """, (
                    doc_id, chunk_id, chunk_index, chunk_text, json.dumps(metadata), embedding,
                    chunk_text, json.dumps(metadata), embedding
                ))
                
            return True
        except Exception as e:
            logger.error(f"Error storing chunk {chunk_index} for {doc_id}: {str(e)}")
            return False
    
    async def process_document(
        self, 
        session: aiohttp.ClientSession, 
        doc: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a single document: 
        1. Generate summary
        2. Generate embeddings for full document
        3. Split into chunks and generate embeddings for each
        """
        doc_id = doc["id"]
        url = doc["url"]
        full_text = doc["translated_text"]
        fetched_at = doc["fetched_at"]
        
        now = datetime.now(pytz.UTC)
        
        log_entry = {
            "ts": now.isoformat(),
            "id": doc_id,
            "url": url,
            "size": len(full_text) if full_text else 0,
            "status": "error",
            "error": None,
            "summary_status": "pending",
            "chunks_status": "pending",
            "chunk_count": 0
        }
        
        try:
            # 1. Generate summary for the document
            logger.info(f"Generating summary for document {doc_id} from URL {url}")
            summary = await self.generate_summary(session, full_text)
            
            if not summary:
                log_entry["summary_status"] = "error"
                log_entry["error"] = "Summary generation failed"
            else:
                # Generate embedding for the summary
                summary_embedding = await self.generate_embedding(session, summary)
                
                if not summary_embedding:
                    log_entry["summary_status"] = "error"
                    log_entry["error"] = "Summary embedding generation failed"
                else:
                    # Store summary and its embedding
                    if self.store_summary(doc_id, url, fetched_at, summary, summary_embedding):
                        log_entry["summary_status"] = "ok"
                        log_entry["summary_length"] = len(summary)
                    else:
                        log_entry["summary_status"] = "error"
                        log_entry["error"] = "Failed to store summary"
            
            # 2. Generate embedding for full document (limited to ~8K tokens)
            full_text_for_embedding = full_text[:32000] if len(full_text) > 32000 else full_text
            logger.info(f"Generating embedding for document {doc_id} ({len(full_text_for_embedding)} chars)")
            
            doc_embedding = await self.generate_embedding(session, full_text_for_embedding)
            
            if not doc_embedding:
                log_entry["status"] = "error"
                log_entry["error"] = "Document embedding generation failed"
            else:
                # Store document embedding
                if self.store_embedding(doc_id, url, fetched_at, full_text, doc_embedding):
                    log_entry["status"] = "partial" if log_entry["status"] == "error" else "ok"
                else:
                    log_entry["status"] = "error"
                    log_entry["error"] = "Failed to store document embedding"
            
            # 3. Split document into chunks and generate embeddings
            chunks = self.split_into_chunks(full_text)
            log_entry["chunk_count"] = len(chunks)
            
            if not chunks:
                log_entry["chunks_status"] = "skipped"  # No chunks needed
            else:
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
                    if self.store_chunk(doc_id, i, chunk_text, url, fetched_at, chunk_embedding):
                        chunk_success_count += 1
                    else:
                        logger.error(f"Failed to store chunk {i+1}")
                
                # Update log with chunk processing results
                if chunk_success_count == len(chunks):
                    log_entry["chunks_status"] = "ok"
                elif chunk_success_count > 0:
                    log_entry["chunks_status"] = "partial"
                    log_entry["status"] = "partial"
                else:
                    log_entry["chunks_status"] = "error"
                    log_entry["status"] = "partial" if log_entry["status"] != "error" else "error"
                
                log_entry["chunks_success"] = chunk_success_count
                log_entry["chunks_failed"] = len(chunks) - chunk_success_count
                
        except Exception as e:
            log_entry["error"] = f"Unexpected error: {str(e)}"
            logger.error(f"Error processing document {doc_id}: {str(e)}")
        
        return log_entry
    
    async def run_job(self) -> bool:
        """
        Run the embedder job on all translated documents.
        1. Check for new translations
        2. If new translations found, clear previous embeddings
        3. Get all translated documents
        4. Generate and store embeddings and summaries
        """
        job_start = datetime.now(pytz.UTC)
        logger.info(f"Starting enhanced embedder job at {job_start.isoformat()}")
        
        # Check for new translations
        new_translations_available, total_docs = self.check_new_translations()
        
        # If no new translations, skip the job
        if not new_translations_available:
            logger.info("No new translations found, skipping embedding generation")
            return True
            
        logger.info(f"New translations detected, generating embeddings for {total_docs} documents")
        
        # Clear previous embeddings
        if not self.clear_embeddings():
            logger.error("Failed to clear previous embeddings, aborting")
            return False
        
        # Get translated documents
        docs = self.get_translated_documents()
        if not docs:
            logger.error("No translated documents found, aborting")
            return False
        
        # Process documents concurrently
        log_entries = []
        async with aiohttp.ClientSession() as session:
            # Verify API key first
            if self.embedding_enabled:
                self.api_verified = await self.verify_api_key(session)
                if not self.api_verified:
                    logger.warning("OpenAI API key verification failed - using placeholder embeddings")
            
            # Process documents concurrently with semaphore to limit concurrency
            semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent API calls to avoid rate limits
            
            async def process_with_semaphore(doc):
                async with semaphore:
                    return await self.process_document(session, doc)
            
            tasks = [process_with_semaphore(doc) for doc in docs]
            log_entries = await asyncio.gather(*tasks)
        
        # Save log entries
        try:
            with open(self.log_file, "a") as f:
                for entry in log_entries:
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Error writing to log file: {str(e)}")
        
        # Log summary statistics
        success_count = sum(1 for entry in log_entries if entry["status"] == "ok")
        partial_count = sum(1 for entry in log_entries if entry["status"] == "partial")
        failed_count = len(docs) - success_count - partial_count
        
        summary_success = sum(1 for entry in log_entries if entry["summary_status"] == "ok")
        chunks_success = sum(1 for entry in log_entries if entry["chunks_status"] == "ok" or entry["chunks_status"] == "skipped")
        total_chunks = sum(entry.get("chunk_count", 0) for entry in log_entries)
        
        job_end = datetime.now(pytz.UTC)
        job_duration = (job_end - job_start).total_seconds()
        
        logger.info(f"Completed embedding job in {job_duration:.2f} seconds:")
        logger.info(f"- Documents: {success_count} success, {partial_count} partial, {failed_count} failed")
        logger.info(f"- Summaries: {summary_success}/{len(docs)} generated")
        logger.info(f"- Chunks: {chunks_success}/{len(docs)} documents chunked, {total_chunks} total chunks")
        
        # If any embeddings were successful, update the last run timestamp
        if success_count > 0 or partial_count > 0:
            self.update_last_run_timestamp()
            return True
        else:
            return False
    
    async def start(self) -> None:
        """Start the enhanced embedder service with scheduled job."""
        if not self.setup():
            logger.error("Failed to set up embedder service, exiting")
            return
        
        logger.info(f"Enhanced embedder service started")
        logger.info(f"Using model: {self.embedding_model} ({self.embedding_dimensions} dimensions)")
        logger.info(f"Chunk size: {self.chunk_size} chars with {self.chunk_overlap} char overlap")
        logger.info(f"Summary length: {self.summary_length} chars")
        
        # Run job immediately
        success = await self.run_job()
        if success:
            logger.info("Initial embedding job completed successfully")
        else:
            logger.warning("Initial embedding job failed or no documents were processed")
        
        # Exit after job completion (for one-time runs)
        logger.info("Enhanced embedder job completed, exiting")


async def main() -> None:
    """Main entry point for the enhanced embedder service."""
    embedder = EnhancedEmbedder()
    await embedder.start()


if __name__ == "__main__":
    asyncio.run(main())