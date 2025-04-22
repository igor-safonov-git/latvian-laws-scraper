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
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("embedder")


class Embedder:
    """Service for generating and storing embeddings from translated documents."""

    def __init__(self):
        """Initialize the embedder service."""
        # Load environment variables
        load_dotenv()
        
        # Get configuration
        self.database_url = os.getenv("DATABASE_URL")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "8192"))  # Default 8192 tokens
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
        
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
        
        # OpenAI API endpoint
        self.embeddings_url = "https://api.openai.com/v1/embeddings"
        
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
                        id        TEXT PRIMARY KEY,
                        metadata  JSONB,
                        embedding VECTOR({self.embedding_dimensions})
                    )
                """)
                
                logger.info("Created docs table")
            
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
                
                # Check the count of documents in docs table
                cursor.execute("SELECT COUNT(*) FROM docs")
                embedding_count = cursor.fetchone()[0]
                
                # If counts don't match, we need to update
                if embedding_count != total_count:
                    logger.info(f"New translations found: {total_count} translated, {embedding_count} with embeddings")
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
            
    def check_translator_completed(self) -> bool:
        """
        Legacy method for backward compatibility.
        Check if translator service has completed successfully.
        Verifies that no documents are left untranslated.
        """
        ready, count = self.check_new_translations()
        return ready and count > 0
    
    def get_translated_documents(self) -> List[Dict[str, Any]]:
        """Get all translated documents from the database."""
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute("""
                    SELECT url, fetched_at, translated_text
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
        """Clear all existing embeddings from the docs table."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("DELETE FROM docs")
                deleted_count = cursor.rowcount
                logger.info(f"Cleared {deleted_count} previous embeddings")
                return True
        except Exception as e:
            logger.error(f"Error clearing embeddings: {str(e)}")
            return False
    
    def compute_id(self, url: str) -> str:
        """Compute a SHA-256 hash of the URL to use as ID."""
        return hashlib.sha256(url.encode()).hexdigest()
    
    def get_placeholder_embedding(self) -> List[float]:
        """Generate a placeholder embedding for testing when API key is not available."""
        # Generate a fixed-dimension random vector for testing
        import random
        return [random.uniform(-0.1, 0.1) for _ in range(self.embedding_dimensions)]
    
    def truncate_text(self, text: str, max_tokens: int = 8192) -> str:
        """
        Truncate text to a maximum number of tokens.
        This is a simple approximation - 1 token ≈ 4 chars for English text.
        """
        # Simple approximation: 1 token ≈ 4 characters
        char_limit = max_tokens * 4
        
        if len(text) <= char_limit:
            return text
        
        # Truncate to character limit and add note
        truncated = text[:char_limit]
        return truncated + "\n...[Truncated due to token limit]"
        
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
            
        # Truncate text to avoid token limits
        truncated_text = self.truncate_text(text, self.max_tokens)
        
        # Attempt to generate embedding with retries
        for attempt in range(max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "input": truncated_text,
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
                    elif response.status == 429:
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
        translated_text: str,
        embedding: List[float]
    ) -> bool:
        """Store document embedding in the database."""
        try:
            # Create metadata JSON
            metadata = {
                "url": url,
                "fetched_at": fetched_at.isoformat() if isinstance(fetched_at, datetime) else fetched_at,
                "text_preview": translated_text[:200] + "..." if len(translated_text) > 200 else translated_text
            }
            
            # Insert into database
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO docs(id, metadata, embedding)
                    VALUES(%s, %s, %s)
                """, (doc_id, json.dumps(metadata), embedding))
                
            return True
        except Exception as e:
            logger.error(f"Error storing embedding for {doc_id}: {str(e)}")
            return False
    
    async def process_document(
        self, 
        session: aiohttp.ClientSession, 
        doc: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single document: generate embedding and store in database."""
        url = doc["url"]
        doc_id = self.compute_id(url)
        now = datetime.now(pytz.UTC)
        
        log_entry = {
            "ts": now.isoformat(),
            "id": doc_id,
            "url": url,
            "status": "error",
            "error": None
        }
        
        try:
            # Generate embedding
            logger.info(f"Generating embedding for document {doc_id} from URL {url}")
            embedding = await self.generate_embedding(session, doc["translated_text"])
            
            if not embedding:
                log_entry["error"] = "Failed to generate embedding"
                return log_entry
            
            # Store embedding
            if self.store_embedding(
                doc_id, 
                url, 
                doc["fetched_at"], 
                doc["translated_text"],
                embedding
            ):
                log_entry["status"] = "ok"
            else:
                log_entry["error"] = "Failed to store embedding in database"
                
        except Exception as e:
            log_entry["error"] = f"Unexpected error: {str(e)}"
        
        return log_entry
    
    async def run_job(self) -> bool:
        """
        Run the embedder job on all translated documents.
        1. Check for new translations
        2. If new translations found, clear previous embeddings
        3. Get all translated documents
        4. Generate and store embeddings
        """
        job_start = datetime.now(pytz.UTC)
        logger.info(f"Starting embedder job at {job_start.isoformat()}")
        
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
            semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent API calls
            
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
        
        # Log summary
        success_count = sum(1 for entry in log_entries if entry["status"] == "ok")
        job_end = datetime.now(pytz.UTC)
        job_duration = (job_end - job_start).total_seconds()
        logger.info(f"Completed embedding job in {job_duration:.2f} seconds: {success_count}/{len(docs)} successful, {len(docs) - success_count} failed")
        
        # If any embeddings were successful, update the last run timestamp
        if success_count > 0:
            self.update_last_run_timestamp()
            
        return success_count > 0
    
    async def start(self) -> None:
        """Start the embedder service with scheduled job."""
        if not self.setup():
            logger.error("Failed to set up embedder service, exiting")
            return
        
        logger.info(f"Embedder service started, scheduled for 00:30 UTC daily")
        
        # Set up scheduler
        scheduler = AsyncIOScheduler(timezone=pytz.UTC)
        scheduler.add_job(self.run_job, 'cron', hour=0, minute=30)
        scheduler.start()
        
        # Run job immediately if requested
        if os.getenv("RUN_ON_STARTUP", "false").lower() == "true":
            logger.info("Running initial job on startup")
            await self.run_job()
        
        # Keep the script running
        try:
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down embedder service")
            scheduler.shutdown()


async def main() -> None:
    """Main entry point for the embedder service."""
    embedder = Embedder()
    await embedder.start()


if __name__ == "__main__":
    asyncio.run(main())