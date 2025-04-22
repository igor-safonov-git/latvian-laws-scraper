#!/usr/bin/env python3
import asyncio
import aiohttp
import hashlib
import json
import logging
import os
import sys
import psycopg2
import psycopg2.extras
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pytz


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("scraper")


class Scraper:
    """Async scraper for Latvian legal documents with PostgreSQL storage."""

    def __init__(self):
        """Initialize the scraper."""
        # Load environment variables
        load_dotenv()
        
        # Get configuration
        self.database_url = os.getenv("DATABASE_URL")
        self.links_file = os.getenv("LINKS_FILE", "links.txt")
        
        # Check required configuration
        if not self.database_url:
            logger.error("DATABASE_URL environment variable not set")
            sys.exit(1)
        
        if not Path(self.links_file).exists():
            logger.error(f"Links file not found: {self.links_file}")
            sys.exit(1)
        
        # Initialize database connection
        self.conn = None
        
        # Ensure logs directory exists
        self.logs_dir = Path("./logs")
        self.logs_dir.mkdir(exist_ok=True)
        self.log_file = self.logs_dir / "scraper.log"

    def setup(self) -> None:
        """Set up the necessary connections and database tables."""
        try:
            # Connect to database
            self.conn = psycopg2.connect(self.database_url)
            self.conn.autocommit = True
            logger.info("Connected to PostgreSQL database")
            
            # Create table if it doesn't exist
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS raw_docs (
                        id          TEXT PRIMARY KEY,
                        url         TEXT NOT NULL,
                        fetched_at  TIMESTAMPTZ NOT NULL,
                        raw_text    TEXT NOT NULL,
                        processed   BOOLEAN DEFAULT FALSE
                    )
                """)
            logger.info("Database tables prepared")
        except Exception as e:
            logger.error(f"Failed to set up database: {str(e)}")
            sys.exit(1)

    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch HTML content from a URL."""
        try:
            async with session.get(url, timeout=60) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch {url}: HTTP {response.status}")
                    return None
                return await response.text()
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def extract_text(self, html: str) -> Optional[str]:
        """Extract plain text from HTML content."""
        if not html:
            return None
        
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text(separator="\n")
            
            # Remove extra whitespace
            lines = (line.strip() for line in text.splitlines())
            text = "\n".join(line for line in lines if line)
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return None

    def compute_id(self, url: str) -> str:
        """Compute a SHA-256 hash of the URL to use as ID."""
        return hashlib.sha256(url.encode()).hexdigest()

    def save_to_database(self, doc_id: str, url: str, text: str, timestamp: datetime) -> bool:
        """Save document to database."""
        try:
            # Check if we already have this document with the same content
            with self.conn.cursor() as cursor:
                cursor.execute(
                    "SELECT raw_text FROM raw_docs WHERE id = %s",
                    (doc_id,)
                )
                existing_doc = cursor.fetchone()
            
            # Determine if we need to mark as unprocessed
            content_changed = True
            if existing_doc and existing_doc[0] == text:
                content_changed = False
            
            # Insert or update the document
            with self.conn.cursor() as cursor:
                if content_changed:
                    cursor.execute("""
                        INSERT INTO raw_docs(id, url, fetched_at, raw_text, processed)
                        VALUES(%s, %s, %s, %s, FALSE)
                        ON CONFLICT (id) DO UPDATE
                        SET raw_text = EXCLUDED.raw_text,
                            fetched_at = EXCLUDED.fetched_at,
                            processed = FALSE
                    """, (doc_id, url, timestamp, text))
                else:
                    cursor.execute("""
                        INSERT INTO raw_docs(id, url, fetched_at, raw_text, processed)
                        VALUES(%s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE
                        SET fetched_at = EXCLUDED.fetched_at
                    """, (doc_id, url, timestamp, text, existing_doc is not None))
            
            return True
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return False

    async def process_url(self, session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
        """Process a single URL: fetch, extract text, save to database."""
        logger.info(f"Processing {url}")
        now = datetime.now(pytz.UTC)
        log_entry = {
            "ts": now.isoformat(),
            "url": url,
            "status": "error",
            "error": None
        }
        
        try:
            # Fetch HTML
            html = await self.fetch_url(session, url)
            if not html:
                log_entry["error"] = "Failed to fetch URL"
                return log_entry
            
            # Extract text
            text = self.extract_text(html)
            if not text:
                log_entry["error"] = "Failed to extract text"
                return log_entry
            
            # Compute ID
            doc_id = self.compute_id(url)
            
            # Save to database
            if self.save_to_database(doc_id, url, text, now):
                log_entry["status"] = "ok"
            else:
                log_entry["error"] = "Failed to save to database"
                
        except Exception as e:
            log_entry["error"] = f"Unexpected error: {str(e)}"
        
        return log_entry

    async def run_job(self) -> None:
        """Run the scraper job on all URLs in the links file."""
        logger.info(f"Starting scraper job at {datetime.now().isoformat()}")
        
        # Read URLs from links file
        try:
            with open(self.links_file, "r") as f:
                urls = [line.strip() for line in f if line.strip()]
            logger.info(f"Found {len(urls)} URLs to process")
        except Exception as e:
            logger.error(f"Error reading links file: {str(e)}")
            return
        
        # Process URLs concurrently
        log_entries = []
        async with aiohttp.ClientSession() as session:
            tasks = [self.process_url(session, url) for url in urls]
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
        logger.info(f"Completed processing {len(urls)} URLs: {success_count} successful, {len(urls) - success_count} failed")

    async def start(self) -> None:
        """Start the scraper with a scheduled job."""
        self.setup()
        
        # Set up scheduler
        scheduler = AsyncIOScheduler(timezone=pytz.UTC)
        scheduler.add_job(self.run_job, 'cron', hour=0, minute=0)
        scheduler.start()
        logger.info("Scheduler started, job will run daily at 00:00 UTC")
        
        # Run job immediately on startup
        await self.run_job()
        
        # Keep the script running
        try:
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down scheduler")
            scheduler.shutdown()


async def main() -> None:
    """Main entry point for the scraper."""
    scraper = Scraper()
    await scraper.start()


if __name__ == "__main__":
    asyncio.run(main())