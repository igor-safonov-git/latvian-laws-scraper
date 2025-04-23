#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

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
logger = logging.getLogger("translator")


class Translator:
    """Async translator service for Latvian legal documents."""

    def __init__(self):
        """Initialize the translator service."""
        # Load environment variables
        load_dotenv()
        
        # Get configuration
        self.database_url = os.getenv("DATABASE_URL")
        self.deepl_api_key = os.getenv("DEEPL_API_KEY")
        self.poll_interval = int(os.getenv("POLL_INTERVAL", "60"))  # Default 60 seconds
        self.batch_size = int(os.getenv("BATCH_SIZE", "10"))  # Default 10 docs (lowered due to possible rate limits)
        self.max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "4000"))  # Default 4000 chars per chunk
        
        # Check required configuration
        if not self.database_url:
            logger.error("DATABASE_URL environment variable not set")
            sys.exit(1)
            
        if not self.deepl_api_key:
            logger.error("DEEPL_API_KEY environment variable not set")
            logger.error("Translation cannot proceed without a valid DeepL API key")
            self.translation_enabled = False
        else:
            self.translation_enabled = True
        
        # Initialize database connection
        self.conn = None
        
        # Ensure logs directory exists
        self.logs_dir = Path("./logs")
        self.logs_dir.mkdir(exist_ok=True)
        self.log_file = self.logs_dir / "translator.log"
        
        # DeepL API endpoint
        self.translate_url = "https://api-free.deepl.com/v2/translate"
        
        # API verification status
        self.api_verified = False
        
    async def verify_api_key(self, session: aiohttp.ClientSession) -> bool:
        """Verify the DeepL API key by checking usage."""
        if not self.translation_enabled:
            logger.warning("API verification skipped - translation not enabled")
            return False
            
        try:
            headers = {
                "Authorization": f"DeepL-Auth-Key {self.deepl_api_key}"
            }
            
            usage_url = "https://api-free.deepl.com/v2/usage"
            
            async with session.get(usage_url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    usage_data = await response.json()
                    logger.info(f"DeepL API key valid. Character usage: "
                                f"{usage_data.get('character_count', 0)}/{usage_data.get('character_limit', 0)}")
                    return True
                elif response.status == 403:
                    error_text = await response.text()
                    logger.error(f"DeepL API key invalid: {error_text}")
                    return False
                else:
                    error_text = await response.text()
                    logger.error(f"DeepL API verification failed: HTTP {response.status} - {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Error verifying DeepL API key: {str(e)}")
            return False

    def setup(self) -> None:
        """Set up the necessary database connections and update schema."""
        try:
            # Connect to database
            self.conn = psycopg2.connect(self.database_url)
            self.conn.autocommit = True
            logger.info("Connected to PostgreSQL database")
            
            # Update schema if needed
            with self.conn.cursor() as cursor:
                # Check if translated_text column exists
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'raw_docs' AND column_name = 'translated_text'
                """)
                
                if not cursor.fetchone():
                    logger.info("Adding translated_text column to raw_docs table")
                    cursor.execute("""
                        ALTER TABLE raw_docs 
                        ADD COLUMN translated_text TEXT
                    """)
            
            logger.info("Database schema prepared")
        except Exception as e:
            logger.error(f"Failed to set up database: {str(e)}")
            sys.exit(1)
            
    def get_placeholder_translation(self, text: str) -> None:
        """Return None when no API key is available."""
        logger.error("No valid DeepL API key available - translation not possible")
        return None

    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks that won't exceed DeepL's size limit.
        DeepL has a maximum request size, so we need to split long texts.
        We'll try to split at paragraph boundaries to maintain context.
        """
        # If text is small enough, return it as a single chunk
        if len(text) <= self.max_chunk_size:
            return [text]
        
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size, save current chunk and start a new one
            if len(current_chunk) + len(paragraph) + 2 > self.max_chunk_size:
                # If a single paragraph is too large, we need to split it further
                if not current_chunk and len(paragraph) > self.max_chunk_size:
                    # Split paragraph by sentences (approximated by periods followed by space)
                    sentences = paragraph.replace('. ', '.\n').split('\n')
                    sentence_chunk = ""
                    
                    for sentence in sentences:
                        if len(sentence_chunk) + len(sentence) + 1 > self.max_chunk_size:
                            if sentence_chunk:
                                chunks.append(sentence_chunk)
                                sentence_chunk = sentence
                            else:
                                # Even a single sentence is too long, split by character
                                for i in range(0, len(sentence), self.max_chunk_size):
                                    chunks.append(sentence[i:i+self.max_chunk_size])
                        else:
                            if sentence_chunk:
                                sentence_chunk += " " + sentence
                            else:
                                sentence_chunk = sentence
                    
                    if sentence_chunk:
                        chunks.append(sentence_chunk)
                else:
                    # Save the current chunk and start a new one with this paragraph
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = paragraph
            else:
                # Add paragraph to the current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)
            
        logger.info(f"Split text into {len(chunks)} chunks (original size: {len(text)} chars)")
        return chunks

    async def translate_text(self, session: aiohttp.ClientSession, text: str) -> Optional[str]:
        """Translate text from Latvian to English using DeepL API with chunking for large texts."""
        # If translation is not enabled, return None
        if not self.translation_enabled:
            logger.error("Translation not enabled - no valid API key available")
            return None
            
        # Verify API key if it hasn't been verified yet
        if not self.api_verified:
            self.api_verified = await self.verify_api_key(session)
            if not self.api_verified:
                logger.error("DeepL API key verification failed - translation not possible")
                return None
                
        try:
            # Split text into manageable chunks
            chunks = self.split_text_into_chunks(text)
            
            # If we have multiple chunks, log it
            if len(chunks) > 1:
                logger.info(f"Translating document in {len(chunks)} chunks due to size")
            
            translated_chunks = []
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                logger.info(f"Translating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                
                headers = {
                    "Authorization": f"DeepL-Auth-Key {self.deepl_api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "text": [chunk],
                    "source_lang": "LV",  # Latvian
                    "target_lang": "EN",  # English
                }
                
                # Add small delay between chunks to avoid rate limiting
                if i > 0:
                    await asyncio.sleep(1)
                
                try:
                    async with session.post(
                        self.translate_url, 
                        headers=headers, 
                        json=data,
                        timeout=60  # Shorter timeout for smaller chunks
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            if "translations" in result and len(result["translations"]) > 0:
                                translated_chunks.append(result["translations"][0]["text"])
                            else:
                                logger.error(f"Invalid translation response for chunk {i+1}: {result}")
                                return None
                        elif response.status == 403:
                            # Authentication error
                            error_text = await response.text()
                            logger.error(f"Authentication failed for DeepL API: {error_text}")
                            self.api_verified = False
                            return None
                        elif response.status == 456:
                            # Character limit reached
                            logger.error("DeepL API character limit reached")
                            return None
                        elif response.status == 429 or response.status == 529:
                            # Rate limiting - wait and retry once
                            logger.warning(f"Rate limit hit, waiting 5 seconds before retry")
                            await asyncio.sleep(5)
                            
                            # Retry the request
                            async with session.post(
                                self.translate_url, 
                                headers=headers, 
                                json=data,
                                timeout=60
                            ) as retry_response:
                                if retry_response.status == 200:
                                    retry_result = await retry_response.json()
                                    if "translations" in retry_result and len(retry_result["translations"]) > 0:
                                        translated_chunks.append(retry_result["translations"][0]["text"])
                                    else:
                                        logger.error(f"Invalid translation response on retry: {retry_result}")
                                        return None
                                else:
                                    error_text = await retry_response.text()
                                    logger.error(f"Translation failed on retry: HTTP {retry_response.status} - {error_text}")
                                    return None
                        else:
                            error_text = await response.text()
                            logger.error(f"Translation failed for chunk {i+1}: HTTP {response.status} - {error_text}")
                            return None
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.error(f"Network error translating chunk {i+1}: {str(e)}")
                    return None
            
            if not translated_chunks:
                logger.error("No chunks were successfully translated")
                return None
                
            # Join all translated chunks
            full_translation = "\n\n".join(translated_chunks)
            return full_translation
            
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            return None
    
    def get_untranslated_docs(self) -> List[Dict[str, Any]]:
        """
        Get a batch of untranslated documents from the database.
        Only returns documents that:
        1. Have processed = FALSE (either new or content changed)
        2. Have raw_text that's not NULL
        """
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                # First check if we have documents that need translation
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM raw_docs
                    WHERE processed = FALSE AND raw_text IS NOT NULL
                """)
                count = cursor.fetchone()[0]
                
                if count == 0:
                    logger.info("No untranslated documents found")
                    return []
                
                # Get documents that need translation
                cursor.execute(f"""
                    SELECT id, url, raw_text, fetched_at
                    FROM raw_docs
                    WHERE processed = FALSE AND raw_text IS NOT NULL
                    ORDER BY fetched_at DESC
                    LIMIT {self.batch_size}
                """)
                
                # Convert to list of dicts
                docs = [dict(row) for row in cursor.fetchall()]
                logger.info(f"Found {len(docs)} untranslated documents")
                
                # Log document ids and urls for clarity
                for doc in docs:
                    logger.info(f"Document requiring translation: {doc['id']} - {doc['url']}")
                
                return docs
        except Exception as e:
            logger.error(f"Error fetching untranslated documents: {str(e)}")
            return []
            
    def check_content_changed(self, doc_id: str, current_text: str) -> bool:
        """
        Check if the content has actually changed compared to the previous version.
        This is a double-check in case the processed flag was reset but content is identical.
        """
        try:
            with self.conn.cursor() as cursor:
                # Get the previously translated document
                cursor.execute("""
                    SELECT translated_text, raw_text
                    FROM raw_docs
                    WHERE id = %s
                """, (doc_id,))
                
                result = cursor.fetchone()
                if not result or not result[0]:
                    # No previous translation, so yes, we need to translate
                    logger.info(f"Document {doc_id} has no previous translation")
                    return True
                
                # We have a previous translation, get the previously translated raw text
                previous_raw_text = result[1] if len(result) > 1 else None
                
                # Do a more detailed content comparison
                current_stripped = current_text.strip() if current_text else ""
                previous_stripped = previous_raw_text.strip() if previous_raw_text else ""
                
                # Always log the content for debugging
                logger.info(f"Content comparison for {doc_id}:")
                logger.info(f"  - Previous text (first 50 chars): '{previous_stripped[:50]}'")
                logger.info(f"  - Current text (first 50 chars): '{current_stripped[:50]}'")
                logger.info(f"  - Previous length: {len(previous_stripped)}, Current length: {len(current_stripped)}")
                logger.info(f"  - Are they identical: {previous_stripped == current_stripped}")
                
                # If content is identical (ignoring trailing/leading whitespace), don't translate again
                if previous_stripped == current_stripped:
                    # Content is identical to what was previously translated
                    logger.info(f"Document {doc_id} content is unchanged from previous translation")
                    
                    # Update the processed flag to avoid reprocessing
                    cursor.execute("""
                        UPDATE raw_docs
                        SET processed = TRUE
                        WHERE id = %s
                    """, (doc_id,))
                    
                    return False
                else:
                    # Content has changed, needs new translation
                    logger.info(f"Document {doc_id} content has changed and needs new translation")
                    return True
                    
        except Exception as e:
            logger.error(f"Error checking content changes for {doc_id}: {str(e)}")
            # Default to translating in case of error
            return True
    
    def update_doc(self, doc_id: str, translated_text: str) -> bool:
        """Update document with translated text and mark as processed."""
        try:
            with self.conn.cursor() as cursor:
                # First get the current raw_text so we can store it properly
                cursor.execute("""
                    SELECT raw_text FROM raw_docs
                    WHERE id = %s
                """, (doc_id,))
                
                raw_text = cursor.fetchone()
                if not raw_text:
                    logger.warning(f"Document {doc_id} not found for update")
                    return False
                
                # Update the document with translation and store the current raw_text
                # for future content change detection
                cursor.execute("""
                    UPDATE raw_docs
                    SET translated_text = %s, processed = TRUE
                    WHERE id = %s
                """, (translated_text, doc_id))
                
                # Check if update was successful
                if cursor.rowcount > 0:
                    return True
                else:
                    logger.warning(f"No document updated with ID {doc_id}")
                    return False
        except Exception as e:
            logger.error(f"Database error updating document {doc_id}: {str(e)}")
            return False
    
    async def process_doc(self, session: aiohttp.ClientSession, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single document: translate and update database."""
        doc_id = doc["id"]
        url = doc["url"]
        raw_text = doc["raw_text"]
        raw_text_size = len(raw_text) if raw_text else 0
        now = datetime.now(pytz.UTC)
        
        log_entry = {
            "ts": now.isoformat(),
            "id": doc_id,
            "url": url,
            "size": raw_text_size,
            "status": "error",
            "error": None
        }
        
        try:
            # Check if content has actually changed since last translation
            if not self.check_content_changed(doc_id, raw_text):
                logger.info(f"Skipping translation for document {doc_id} - content unchanged")
                log_entry["status"] = "skipped"
                log_entry["reason"] = "content unchanged"
                return log_entry
            
            # Translate text
            logger.info(f"Translating document {doc_id} from URL {url} ({raw_text_size} chars)")
            translated_text = await self.translate_text(session, raw_text)
            
            if not translated_text:
                log_entry["error"] = "Translation failed"
                return log_entry
            
            # Log translation size
            translated_size = len(translated_text)
            log_entry["translated_size"] = translated_size
            logger.info(f"Translation complete: {translated_size} chars (ratio: {translated_size/raw_text_size:.2f}x)")
            
            # Update document with translation
            if self.update_doc(doc_id, translated_text):
                log_entry["status"] = "ok"
            else:
                log_entry["error"] = "Failed to update document in database"
                
        except Exception as e:
            log_entry["error"] = f"Unexpected error: {str(e)}"
        
        return log_entry
    
    async def run_job(self) -> None:
        """Run the translator job on a batch of untranslated documents."""
        logger.info(f"Starting translator job at {datetime.now().isoformat()}")
        
        # Get untranslated documents
        docs = self.get_untranslated_docs()
        if not docs:
            logger.info("No untranslated documents found")
            return
        
        # Process documents concurrently
        log_entries = []
        async with aiohttp.ClientSession() as session:
            tasks = [self.process_doc(session, doc) for doc in docs]
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
        skipped_count = sum(1 for entry in log_entries if entry["status"] == "skipped")
        failed_count = len(docs) - success_count - skipped_count
        
        logger.info(f"Completed processing {len(docs)} documents: {success_count} successful, {skipped_count} skipped (unchanged), {failed_count} failed")
    
    def run_tests(self) -> bool:
        """Run tests to verify translator functionality."""
        logger.info("Running translator tests to verify functionality...")
        try:
            # Run the test_translator.py script
            result = subprocess.run(
                ["python", "test_translator.py"], 
                capture_output=True, 
                text=True
            )
            
            # Check if the test was successful
            if result.returncode == 0:
                logger.info("Translator tests completed successfully!")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"Translator tests failed with code {result.returncode}")
                logger.error(result.stderr)
                return False
        except Exception as e:
            logger.error(f"Error running translator tests: {str(e)}")
            return False
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """Get statistics about translation progress."""
        stats = {
            "total_documents": 0,
            "translated_documents": 0,
            "untranslated_documents": 0,
            "pending_documents": 0,  # Docs that need translation (processed=FALSE)
            "skipped_documents": 0,  # Docs marked for processing but content unchanged
            "translation_percentage": 0,
            "average_translation_length": 0,
        }
        
        try:
            with self.conn.cursor() as cursor:
                # Get total document count
                cursor.execute("SELECT COUNT(*) FROM raw_docs")
                stats["total_documents"] = cursor.fetchone()[0]
                
                # Get translated document count
                cursor.execute("SELECT COUNT(*) FROM raw_docs WHERE translated_text IS NOT NULL")
                stats["translated_documents"] = cursor.fetchone()[0]
                
                # Get untranslated document count
                cursor.execute("SELECT COUNT(*) FROM raw_docs WHERE translated_text IS NULL")
                stats["untranslated_documents"] = cursor.fetchone()[0]
                
                # Get pending document count (marked for processing)
                cursor.execute("SELECT COUNT(*) FROM raw_docs WHERE processed = FALSE")
                stats["pending_documents"] = cursor.fetchone()[0]
                
                # Estimate skipped documents (since we don't store this permanently)
                # This is the number of docs that were marked for processing
                # but already had a translation and unchanged content
                cursor.execute("""
                    SELECT COUNT(*) FROM raw_docs 
                    WHERE translated_text IS NOT NULL 
                    AND processed = TRUE 
                    AND id IN (
                        SELECT id FROM raw_docs 
                        WHERE processed = FALSE 
                        ORDER BY fetched_at DESC 
                        LIMIT 100
                    )
                """)
                stats["skipped_documents"] = cursor.fetchone()[0]
                
                # Calculate percentage
                if stats["total_documents"] > 0:
                    stats["translation_percentage"] = round(
                        (stats["translated_documents"] / stats["total_documents"]) * 100, 2
                    )
                
                # Get average translation length
                cursor.execute("""
                    SELECT AVG(LENGTH(translated_text)) 
                    FROM raw_docs 
                    WHERE translated_text IS NOT NULL
                """)
                avg_length = cursor.fetchone()[0]
                if avg_length is not None:
                    stats["average_translation_length"] = int(avg_length)
                
            return stats
        except Exception as e:
            logger.error(f"Error getting translation stats: {str(e)}")
            return stats

    async def start(self) -> None:
        """Run the translator job once."""
        self.setup()

        # Check API key status
        async with aiohttp.ClientSession() as session:
            self.api_verified = await self.verify_api_key(session)
            if self.api_verified:
                logger.info("DeepL API key verified successfully")
            else:
                logger.error("DeepL API key verification failed - translation cannot proceed")
                logger.error("Update DEEPL_API_KEY in Heroku config with a valid API key")

        # Run translation job
        await self.run_job()

        # Run tests to verify translator functionality
        self.run_tests()


async def main() -> None:
    """Main entry point for the translator service."""
    translator = Translator()
    await translator.start()


if __name__ == "__main__":
    asyncio.run(main())