# Heroku Commands for latvian-laws application

## Deployment and Status
```bash
# Deploy to Heroku
git push heroku main

# Check app status
heroku ps --app latvian-laws

# View logs
heroku logs --tail --app latvian-laws

# Open app in browser
heroku open --app latvian-laws
```

## Database
```bash
# Connect to PostgreSQL
heroku pg:psql --app latvian-laws

# Reset database (CAUTION: destructive operation)
heroku pg:reset DATABASE --confirm latvian-laws --app latvian-laws
```

## Service Management
```bash
# Start/stop scraper
heroku ps:scale scraper=1 --app latvian-laws
heroku ps:scale scraper=0 --app latvian-laws

# Start/stop translator
heroku ps:scale translator=1 --app latvian-laws
heroku ps:scale translator=0 --app latvian-laws

# Start/stop embedder
heroku ps:scale embedder=1 --app latvian-laws
heroku ps:scale embedder=0 --app latvian-laws

# Start/stop enhanced embedder
heroku ps:scale enhanced_embedder=1 --app latvian-laws
heroku ps:scale enhanced_embedder=0 --app latvian-laws

# Start/stop web interface
heroku ps:scale web=1 --app latvian-laws
heroku ps:scale web=0 --app latvian-laws
```

## Testing
```bash
# Run tests on Heroku
heroku run python test_db.py [-v/--verbose] --app latvian-laws
heroku run python test_translator.py [-v/--verbose] [--check-one] --app latvian-laws
heroku run python test_embedder.py [-v/--verbose] [--sample] --app latvian-laws
heroku run python test_embedder_enhanced.py [-v/--verbose] --app latvian-laws
```

## One-Time Operations
```bash
# Run one-time embedder operation
heroku run python run_embedder.py --app latvian-laws

# Run one-time enhanced embedder operation
heroku run python run_enhanced_embedder_once.py --app latvian-laws

# Fix embeddings (orphaned records, token limit issues)
heroku run python embedder_fix.py --app latvian-laws

# Upgrade embeddings from small to large model
heroku run python embedder_upgrade.py --app latvian-laws
```

## Configuration
```bash
# Set environment variables
heroku config:set EMBEDDING_MODEL=text-embedding-3-large --app latvian-laws
heroku config:set EMBEDDING_DIMENSIONS=3072 --app latvian-laws
heroku config:set CHUNK_SIZE=3000 --app latvian-laws
heroku config:set CHUNK_OVERLAP=500 --app latvian-laws
heroku config:set SUMMARY_LENGTH=3000 --app latvian-laws

# View current configuration
heroku config --app latvian-laws
```