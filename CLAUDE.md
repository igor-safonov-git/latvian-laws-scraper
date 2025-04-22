# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run Flask app: `flask run`
- Deploy: `git push heroku main`
- Logs: `heroku logs --tail`
- Database: `heroku pg:psql`
- Tests: `pytest` (once implemented)
- Linting: `flake8 .` (once configured)

## Code Style Guidelines
- Python: Follow PEP 8 standards with 4-space indentation
- JS/TS: Use 2 spaces for indentation
- Naming: snake_case for Python, camelCase for JS/TS
- Imports: Group and sort (1)stdlib (2)third-party (3)local
- Error handling: Use try/except with specific exception types
- Type hints: Add Python type annotations for all functions
- Async: Use asyncio and async/await consistently
- Security: No hardcoded secrets, validate all inputs

## Infrastructure
- Heroku app: latvian-laws (EU region)
- Database: PostgreSQL (accessed via DATABASE_URL)
- API Keys: Telegram, DeepL, OpenAI (in env variables)
- URL: https://latvian-laws-06e89c613b8a.herokuapp.com/

## Project Structure
This repository processes Latvian legal documents:
- Scraping functionality fetches and parses legal texts
- API endpoints serve processed data
- Data storage in PostgreSQL with efficient querying
- Multilingual translation and search capabilities