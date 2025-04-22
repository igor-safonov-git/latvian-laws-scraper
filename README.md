# Latvian Laws Scraper

An asynchronous scraper for Latvian legal documents with PostgreSQL storage.

## Features

- Scheduled scraping (daily at midnight UTC)
- Asynchronous fetching for speed
- HTML to text conversion
- PostgreSQL database storage
- Detailed logging

## Usage

1. Add URLs to `links.txt` (one per line)
2. The scraper will:
   - Run daily at 00:00 UTC
   - Save raw text to PostgreSQL database
   - Log results to PostgreSQL database

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL=your_postgres_connection_string

# Run the scraper
python scraper.py
```

## Database Schema

Two tables are created automatically:

1. `law_texts` - Stores the scraped content:
   - id (SERIAL PRIMARY KEY)
   - url (TEXT)
   - fetched_at (TIMESTAMP)
   - raw_text (TEXT)

2. `scrape_logs` - Stores operation logs:
   - id (SERIAL PRIMARY KEY)
   - url (TEXT)
   - timestamp (TIMESTAMP)
   - success (BOOLEAN)
   - message (TEXT)

## Deployment

This application is deployed on Heroku with:
- PostgreSQL database
- Worker process for scheduled scraping
- Environment variables for API keys