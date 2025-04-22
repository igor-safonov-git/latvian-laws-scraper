# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run services:
  - Scraper: `python scraper.py`
  - Translator: `python translator.py`
  - Embedder: `python embedder.py`
  - Web: `python app.py`
- Run tests:
  - Scraper: `python test_db.py [-v/--verbose]`
  - Translator: `python test_translator.py [-v/--verbose] [--check-one]`
  - Embedder: `python test_embedder.py [-v/--verbose] [--sample]`
- Deploy: `git push heroku main`
- Dyno management:
  - Start/stop scraper: `heroku ps:scale scraper=1/0 --app latvian-laws`
  - Start/stop translator: `heroku ps:scale translator=1/0 --app latvian-laws`
  - Start/stop embedder: `heroku ps:scale embedder=1/0 --app latvian-laws`
  - Start/stop web: `heroku ps:scale web=1/0 --app latvian-laws`
- Logs: `heroku logs --tail --app latvian-laws`
- Database: `heroku pg:psql --app latvian-laws`
- Linting: `flake8 *.py`

## Code Style Guidelines
- Python: Follow PEP 8 with 4-space indentation, 88-char line limit
- Naming: snake_case for variables/functions, PascalCase for classes
- Imports: Group (1)stdlib (2)third-party (3)local, alpha-sorted
- Type hints: Required for all function parameters and return values
- Error handling: Use specific exceptions with meaningful error messages
- Async: Use aiohttp for HTTP, asyncio.gather for concurrency
- Docstrings: Required for all functions and classes (Google-style)
- Security: No hardcoded credentials, all secrets in env variables

## Infrastructure
- Heroku app: latvian-laws (EU region)
- Database: PostgreSQL (via DATABASE_URL)
  - pgvector extension for vector embeddings
- Python version: 3.11.8 (specified in runtime.txt)
- Services:
  - Scraper: Fetches and parses Latvian law documents (daily at 00:00 UTC)
  - Translator: Translates documents from Latvian to English (polls every 60s)
  - Embedder: Generates vector embeddings for translated documents (daily at 00:30 UTC)
  - Web: Flask-based status dashboard (/ and /status endpoints)