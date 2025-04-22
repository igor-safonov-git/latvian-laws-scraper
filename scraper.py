#!/usr/bin/env python3
import asyncio
import aiohttp
import json
import os
import logging
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import psycopg2
from psycopg2.extras import Json
import pytz
import urllib.parse as urlparse


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("scraper")


def get_db_connection():
    """Get a connection to the PostgreSQL database."""
    url = os.environ.get('DATABASE_URL')
    if not url:
        logger.error("DATABASE_URL environment variable not set")
        return None
    
    try:
        # Parse the connection URL
        result = urlparse.urlparse(url)
        username = result.username
        password = result.password
        database = result.path[1:]
        hostname = result.hostname
        port = result.port
        
        # Connect to the database
        connection = psycopg2.connect(
            database=database,
            user=username,
            password=password,
            host=hostname,
            port=port
        )
        connection.autocommit = True
        return connection
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        return None


def setup_database():
    """Create necessary tables if they don't exist."""
    connection = get_db_connection()
    if not connection:
        return False
    
    try:
        with connection.cursor() as cursor:
            # Create law_texts table to store scraped data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS law_texts (
                    id SERIAL PRIMARY KEY,
                    url TEXT NOT NULL,
                    fetched_at TIMESTAMP NOT NULL,
                    raw_text TEXT NOT NULL,
                    UNIQUE(url, fetched_at)
                )
            """)
            
            # Create scrape_logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scrape_logs (
                    id SERIAL PRIMARY KEY,
                    url TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    success BOOLEAN NOT NULL,
                    message TEXT NOT NULL
                )
            """)
            
        logger.info("Database setup completed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to setup database: {str(e)}")
        return False
    finally:
        connection.close()


async def fetch_url(session, url):
    """Fetch a URL and return its HTML content."""
    try:
        async with session.get(url, timeout=60) as response:
            if response.status != 200:
                logger.error(f"Failed to fetch {url}: HTTP {response.status}")
                return None
            return await response.text()
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None


def extract_text(html):
    """Extract plain text from HTML."""
    if not html:
        return None
    
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


def log_result(url, success, message):
    """Log the result of processing a URL to the database."""
    connection = get_db_connection()
    if not connection:
        logger.error(f"Could not log result for {url}: Database connection failed")
        return
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO scrape_logs (url, timestamp, success, message)
                VALUES (%s, %s, %s, %s)
            """, (url, datetime.now(), success, message))
        
        # Log to console as well
        if success:
            logger.info(f"SUCCESS: {url} - {message}")
        else:
            logger.error(f"FAILURE: {url} - {message}")
    except Exception as e:
        logger.error(f"Failed to log result to database: {str(e)}")
    finally:
        connection.close()


def save_to_database(url, text):
    """Save the scraped text to the database."""
    connection = get_db_connection()
    if not connection:
        logger.error(f"Could not save data for {url}: Database connection failed")
        return False
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO law_texts (url, fetched_at, raw_text)
                VALUES (%s, %s, %s)
                ON CONFLICT (url, fetched_at) DO UPDATE 
                SET raw_text = EXCLUDED.raw_text
            """, (url, datetime.now(), text))
        
        return True
    except Exception as e:
        logger.error(f"Failed to save data to database: {str(e)}")
        return False
    finally:
        connection.close()


async def process_url(session, url):
    """Process a single URL: fetch, extract text, save to database."""
    logger.info(f"Processing {url}")
    
    html = await fetch_url(session, url)
    if not html:
        log_result(url, False, "Failed to fetch URL")
        return
    
    text = extract_text(html)
    if not text:
        log_result(url, False, "Failed to extract text")
        return
    
    # Save to database
    if save_to_database(url, text):
        log_result(url, True, f"Saved to database")
    else:
        log_result(url, False, "Failed to save to database")


async def run_scraper():
    """Main function to run the scraper on all URLs."""
    logger.info("Starting scraper run")
    
    # Read URLs from links.txt
    try:
        with open("links.txt", "r") as f:
            urls = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error reading links.txt: {str(e)}")
        return
    
    logger.info(f"Found {len(urls)} URLs to process")
    
    # Process URLs concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [process_url(session, url) for url in urls]
        await asyncio.gather(*tasks)
    
    logger.info("Scraper run completed")


def setup_scheduler():
    """Setup the scheduler to run the scraper at midnight."""
    scheduler = AsyncIOScheduler(timezone=pytz.UTC)
    scheduler.add_job(run_scraper, 'cron', hour=0, minute=0)
    logger.info("Scheduled scraper to run daily at 00:00 UTC")
    return scheduler


async def main():
    """Main entry point for the application."""
    # Setup the database first
    if not setup_database():
        logger.error("Failed to setup database, exiting")
        return
    
    # Setup the scheduler
    scheduler = setup_scheduler()
    scheduler.start()
    
    # Also run immediately once
    await run_scraper()
    
    try:
        # Keep the script running
        while True:
            await asyncio.sleep(3600)  # Sleep for an hour
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down scheduler")
        scheduler.shutdown()


if __name__ == "__main__":
    asyncio.run(main())