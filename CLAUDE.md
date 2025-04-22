# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run services (Heroku only):
  - Scraper: `python scraper.py`
  - Translator: `python translator.py`
  - Embedder: `python embedder_enhanced.py`
  - Web: `python app.py`
- Run tests (Heroku only, never locally):
  - All tests: Run each test script separately on Heroku
  - Scraper: `heroku run python test_db.py [-v/--verbose] --app latvian-laws`
  - Translator: `heroku run python test_translator.py [-v/--verbose] [--check-one] --app latvian-laws`
  - Embedder: `heroku run python test_embedder_enhanced.py [-v/--verbose] --app latvian-laws`
  - Single test focus: Use the `--check-one` flag for translator tests
- Deploy: `git push heroku main`
- Dyno management:
  - Start/stop scraper: `heroku ps:scale scraper=1/0 --app latvian-laws`
  - Start/stop translator: `heroku ps:scale translator=1/0 --app latvian-laws`
  - Start/stop embedder: `heroku ps:scale embedder=1/0 --app latvian-laws`
  - Start/stop web: `heroku ps:scale web=1/0 --app latvian-laws`
- Logs: `heroku logs --tail --app latvian-laws`
- Database: `heroku pg:psql --app latvian-laws`

## Code Style Guidelines
- Python: Follow PEP 8 with 4-space indentation, 88-char line limit
- Naming: snake_case for variables/functions, PascalCase for classes
- Imports: Group (1)stdlib (2)third-party (3)local, alpha-sorted
- Type hints: Required for all function parameters and return values
- Error handling: Use try/except with specific exceptions, log detailed errors
- Logging: Use structured JSON logging with timestamp, status and relevant IDs
- Async: Use aiohttp for HTTP, asyncio.gather for concurrency
- Docstrings: Required for all functions and classes (Google-style)
- Security: No hardcoded credentials, all secrets in env variables
- Content change detection: Compare stripped content to avoid whitespace false positives

## Infrastructure
- Heroku app: latvian-laws (EU region)
- Database: PostgreSQL (via DATABASE_URL)
  - pgvector extension for vector embeddings
- Python version: 3.11.8 (specified in runtime.txt)
- Services:
  - Scraper: Fetches and parses Latvian law documents (daily at 00:00 UTC)
  - Translator: Translates documents from Latvian to English (polls every 60s)
  - Embedder: Generates document chunks, summaries, and embeddings (daily at 00:30 UTC)
  - Web: Flask-based status dashboard (/ and /status endpoints)