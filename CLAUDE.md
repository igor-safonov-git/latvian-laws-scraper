# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Build: TBD based on chosen tech stack
- Deploy: `git push heroku main`
- Logs: `heroku logs --tail`
- Database console: `heroku pg:psql`

## Code Style Guidelines
- Formatting: Use 2 spaces for indentation
- Naming: camelCase for variables/functions, PascalCase for classes
- Imports: Group by (1)standard library (2)third-party (3)local
- Error handling: Try/catch with specific error types
- Async: Use async/await pattern consistently
- Documentation: JSDoc-style comments for all functions
- Security: No secrets in code, validate all inputs

## Infrastructure
- Heroku app: latvian-laws (EU region)
- Database: PostgreSQL (connection in DATABASE_URL)
- API Keys: Telegram, DeepL, OpenAI
- URL: https://latvian-laws-06e89c613b8a.herokuapp.com/

## Project Structure
This repository processes Latvian legal data with:
- Data fetchers and parsers
- API endpoints with appropriate rate limiting
- PostgreSQL storage and query utilities
- Multilingual translation and search functions