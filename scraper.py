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
import asyncpg
import pytz
import urllib.parse as urlparse


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("scraper")


async def get_db_pool():
    """Get a connection pool to the PostgreSQL database."""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        return None
    
    # Convert Heroku style postgres:// URLs to postgresql://
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    try:
        pool = await asyncpg.create_pool(database_url)
        return pool
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        return None


async def setup_database(pool):
    """Create necessary tables if they don't exist."""
    if not pool:
        return False
    
    try:
        async with pool.acquire() as conn:
            # Create law_texts table to store scraped data
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS law_texts (
                    id SERIAL PRIMARY KEY,
                    url TEXT NOT NULL,
                    fetched_at TIMESTAMP NOT NULL,
                    raw_text TEXT NOT NULL,
                    UNIQUE(url, fetched_at)
                )
            """)
            
            # Create scrape_logs table
            await conn.execute("""
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


async def log_result(pool, url, success, message):
    """Log the result of processing a URL to the database."""
    if not pool:
        logger.error(f"Could not log result for {url}: Database connection failed")
        return
    
    try:
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO scrape_logs (url, timestamp, success, message)
                VALUES ($1, $2, $3, $4)
            """, url, datetime.now(), success, message)
        
        # Log to console as well
        if success:
            logger.info(f"SUCCESS: {url} - {message}")
        else:
            logger.error(f"FAILURE: {url} - {message}")
    except Exception as e:
        logger.error(f"Failed to log result to database: {str(e)}")


async def save_to_database(pool, url, text):
    """Save the scraped text to the database."""
    if not pool:
        logger.error(f"Could not save data for {url}: Database connection failed")
        return False
    
    try:
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO law_texts (url, fetched_at, raw_text)
                VALUES ($1, $2, $3)
                ON CONFLICT (url, fetched_at) DO UPDATE 
                SET raw_text = $3
            """, url, datetime.now(), text)
        
        return True
    except Exception as e:
        logger.error(f"Failed to save data to database: {str(e)}")
        return False


async def process_url(pool, session, url):
    """Process a single URL: fetch, extract text, save to database."""
    logger.info(f"Processing {url}")
    
    html = await fetch_url(session, url)
    if not html:
        await log_result(pool, url, False, "Failed to fetch URL")
        return
    
    text = extract_text(html)
    if not text:
        await log_result(pool, url, False, "Failed to extract text")
        return
    
    # Save to database
    if await save_to_database(pool, url, text):
        await log_result(pool, url, True, f"Saved to database")
    else:
        await log_result(pool, url, False, "Failed to save to database")


async def run_scraper():
    """Main function to run the scraper on all URLs."""
    logger.info("Starting scraper run")
    
    # Get database connection pool
    pool = await get_db_pool()
    if not pool:
        logger.error("Failed to create database connection pool, exiting")
        return
    
    # Setup the database tables
    if not await setup_database(pool):
        logger.error("Failed to setup database, exiting")
        await pool.close()
        return
    
    # Read URLs from links.txt
    try:
        with open("links.txt", "r") as f:
            urls = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error reading links.txt: {str(e)}")
        await pool.close()
        return
    
    logger.info(f"Found {len(urls)} URLs to process")
    
    # Process URLs concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [process_url(pool, session, url) for url in urls]
        await asyncio.gather(*tasks)
    
    logger.info("Scraper run completed")
    await pool.close()


def setup_scheduler():
    """Setup the scheduler to run the scraper at midnight."""
    scheduler = AsyncIOScheduler(timezone=pytz.UTC)
    scheduler.add_job(run_scraper, 'cron', hour=0, minute=0)
    logger.info("Scheduled scraper to run daily at 00:00 UTC")
    return scheduler


async def main():
    """Main entry point for the application."""
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