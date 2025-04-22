# Latvian Laws Scraper

An asynchronous scraper for Latvian legal documents.

## Features

- Scheduled scraping (daily at midnight UTC)
- Asynchronous fetching for speed
- HTML to text conversion
- JSON output for each document
- Detailed logging

## Usage

1. Add URLs to `links.txt` (one per line)
2. The scraper will:
   - Run daily at 00:00 UTC
   - Save raw text to `/data/raw/` in JSON format
   - Log results to `/data/logs/`

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the scraper
python scraper.py
```

## Deployment

This application is deployed on Heroku with the following configuration:
- PostgreSQL database
- Environment variables for API keys
- Worker process for scheduled scraping