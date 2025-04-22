#!/usr/bin/env python3
import asyncio
import aiohttp
import json
import os
import logging
from bs4 import BeautifulSoup
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pytz
from pathlib import Path


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("scraper")


def ensure_directories():
    """Ensure that necessary directories exist."""
    data_dir = Path("./data")
    raw_dir = data_dir / "raw"
    logs_dir = data_dir / "logs"
    
    data_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    return raw_dir, logs_dir


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


def log_result(logs_dir, url, success, message):
    """Log the result of processing a URL to a JSON file."""
    log_entry = {
        "url": url,
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "message": message
    }
    
    # Create log filename based on the current date
    log_file = logs_dir / f"scrape_log_{datetime.now().strftime('%Y-%m-%d')}.json"
    
    # Append to log file
    try:
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
        else:
            logs = []
        
        logs.append(log_entry)
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error writing to log file: {str(e)}")
    
    # Log to console as well
    if success:
        logger.info(f"SUCCESS: {url} - {message}")
    else:
        logger.error(f"FAILURE: {url} - {message}")


def save_to_file(raw_dir, url, text):
    """Save the scraped text to a JSON file."""
    # Create output data
    now = datetime.now().isoformat()
    data = {
        "url": url,
        "fetched_at": now,
        "raw_text": text
    }
    
    # Generate filename based on URL and timestamp
    url_safe = url.replace("://", "_").replace("/", "_").replace(".", "_")
    filename = f"{url_safe}_{now.replace(':', '-')}.json"
    filepath = raw_dir / filename
    
    # Save to file
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True, str(filepath)
    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}")
        return False, None


async def process_url(raw_dir, logs_dir, session, url):
    """Process a single URL: fetch, extract text, save to file."""
    logger.info(f"Processing {url}")
    
    html = await fetch_url(session, url)
    if not html:
        log_result(logs_dir, url, False, "Failed to fetch URL")
        return
    
    text = extract_text(html)
    if not text:
        log_result(logs_dir, url, False, "Failed to extract text")
        return
    
    # Save to file
    success, filepath = save_to_file(raw_dir, url, text)
    if success:
        log_result(logs_dir, url, True, f"Saved to {filepath}")
    else:
        log_result(logs_dir, url, False, "Failed to save to file")


async def run_scraper():
    """Main function to run the scraper on all URLs."""
    logger.info("Starting scraper run")
    
    # Ensure directories exist
    raw_dir, logs_dir = ensure_directories()
    
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
        tasks = [process_url(raw_dir, logs_dir, session, url) for url in urls]
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