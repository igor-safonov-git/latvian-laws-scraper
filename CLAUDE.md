# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run services (Heroku only):
  - Scraper: `python scraper.py`
  - Translator: `python translator.py`
  - Embedder: `python embedder_enhanced.py` or `python embedder_memory_efficient.py`
  - Web: `python app.py`
- Run tests (Heroku only, never locally):
  - Scraper: `heroku run python test_db.py [-v/--verbose] --app latvian-laws`
  - Translator: `heroku run python test_translator.py [-v/--verbose] [--check-one] --app latvian-laws`
  - Embedder: `heroku run python test_embedder_enhanced.py [-v/--verbose] --app latvian-laws`
  - Embedder Edge: `heroku run python test_embedder_edge_cases.py [-v/--verbose] --app latvian-laws`
  - Memory Usage: `heroku run python test_memory_usage.py [--size N] --app latvian-laws`
  - Extreme Cases: `heroku run python test_extreme_docs.py --app latvian-laws`
- One-time operations:
  - Run embedder once: `heroku run python run_enhanced_embedder_once.py --app latvian-laws`
  - Fix embeddings: `heroku run python embedder_fix.py --app latvian-laws`
- Service management:
  - Start/stop: `heroku ps:scale [service]=1/0 --app latvian-laws` (service: scraper/translator/embedder/web)
- Logs: `heroku logs --tail --app latvian-laws`
- Database: `heroku pg:psql --app latvian-laws`

## Code Style Guidelines
- Python: Follow PEP 8 with 4-space indentation, 88-char line limit
- Naming: snake_case for variables/functions, PascalCase for classes
- Imports: Group (1)stdlib (2)third-party (3)local, alpha-sorted
- Type hints: Required for all function parameters and return values
- Error handling: Use try/except with specific exceptions, log detailed errors
- Logging: Structured JSON logging with timestamp, status and relevant IDs
- Async: Use aiohttp for HTTP requests, asyncio.gather for concurrency
- Docstrings: Required for all functions/classes (Google-style)
- Memory usage: Prefer streaming for large documents (use embedder_memory_efficient.py)
- Security: No hardcoded credentials, all secrets in env variables

## Infrastructure
- Heroku app: latvian-laws (EU region)
- Database: PostgreSQL with pgvector extension (via DATABASE_URL)
- Python version: 3.11.8 (runtime.txt)
- Data pipeline:
  - Scraper → Translator → Embedder workflow
  - Vector embeddings at document/chunk/summary levels
  - Content chunking with semantic boundary detection
  - Memory-efficient processing for large documents