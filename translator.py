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
        self.batch_size = int(os.getenv("BATCH_SIZE", "100"))  # Default 100 docs
        
        # Check required configuration
        if not self.database_url:
            logger.error("DATABASE_URL environment variable not set")
            sys.exit(1)
            
        if not self.deepl_api_key:
            logger.error("DEEPL_API_KEY environment variable not set")
            sys.exit(1)
        
        # Initialize database connection
        self.conn = None
        
        # Ensure logs directory exists
        self.logs_dir = Path("./logs")
        self.logs_dir.mkdir(exist_ok=True)
        self.log_file = self.logs_dir / "translator.log"
        
        # DeepL API endpoint
        self.translate_url = "https://api-free.deepl.com/v2/translate"
        
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

    async def translate_text(self, session: aiohttp.ClientSession, text: str) -> Optional[str]:
        """Translate text from Latvian to English using DeepL API."""
        try:
            headers = {
                "Authorization": f"DeepL-Auth-Key {self.deepl_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "text": [text],
                "source_lang": "LV",  # Latvian
                "target_lang": "EN",  # English
            }
            
            async with session.post(
                self.translate_url, 
                headers=headers, 
                json=data,
                timeout=120  # Long timeout for large documents
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Translation failed: HTTP {response.status} - {error_text}")
                    return None
                
                result = await response.json()
                if "translations" in result and len(result["translations"]) > 0:
                    return result["translations"][0]["text"]
                else:
                    logger.error(f"Invalid translation response: {result}")
                    return None
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            return None
    
    def get_untranslated_docs(self) -> List[Dict[str, Any]]:
        """Get a batch of untranslated documents from the database."""
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(f"""
                    SELECT id, url, raw_text, fetched_at
                    FROM raw_docs
                    WHERE processed = FALSE AND raw_text IS NOT NULL
                    LIMIT {self.batch_size}
                """)
                
                # Convert to list of dicts
                docs = [dict(row) for row in cursor.fetchall()]
                logger.info(f"Found {len(docs)} untranslated documents")
                return docs
        except Exception as e:
            logger.error(f"Error fetching untranslated documents: {str(e)}")
            return []
    
    def update_doc(self, doc_id: str, translated_text: str) -> bool:
        """Update document with translated text and mark as processed."""
        try:
            with self.conn.cursor() as cursor:
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
        now = datetime.now(pytz.UTC)
        
        log_entry = {
            "ts": now.isoformat(),
            "id": doc_id,
            "url": url,
            "status": "error",
            "error": None
        }
        
        try:
            # Translate text
            logger.info(f"Translating document {doc_id} from URL {url}")
            translated_text = await self.translate_text(session, doc["raw_text"])
            
            if not translated_text:
                log_entry["error"] = "Translation failed"
                return log_entry
            
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
        logger.info(f"Completed processing {len(docs)} documents: {success_count} successful, {len(docs) - success_count} failed")
    
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
        """Start the translator service with a polling interval."""
        self.setup()
        
        logger.info(f"Translator service started, polling every {self.poll_interval} seconds")
        
        # Run job immediately on startup
        await self.run_job()
        
        # Run initial tests
        self.run_tests()
        
        # Keep the script running with polling interval
        try:
            # Track time since last test
            last_test_time = time.time()
            test_interval = 3600  # Run tests hourly
            
            while True:
                await asyncio.sleep(self.poll_interval)
                await self.run_job()
                
                # Run tests periodically
                current_time = time.time()
                if current_time - last_test_time > test_interval:
                    logger.info("Running periodic translator tests...")
                    self.run_tests()
                    
                    # Log translation stats
                    stats = self.get_translation_stats()
                    logger.info(f"Translation stats: {json.dumps(stats)}")
                    
                    last_test_time = current_time
                    
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down translator service")


async def main() -> None:
    """Main entry point for the translator service."""
    translator = Translator()
    await translator.start()


if __name__ == "__main__":
    asyncio.run(main())