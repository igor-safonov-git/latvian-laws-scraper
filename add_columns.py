#!/usr/bin/env python3
"""
Script to add the required columns to the raw_docs table for the formatter.
Adds structured_format (TEXT) and formatted_at (TIMESTAMP) columns.
"""
import os
import logging
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("add_columns")

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

def add_columns():
    """Add required columns to the raw_docs table."""
    try:
        logger.info("Connecting to database...")
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if columns already exist
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'raw_docs' 
            AND column_name IN ('structured_format', 'formatted_at')
        """)
        
        existing_columns = [row[0] for row in cursor.fetchall()]
        
        # Add structured_format column if it doesn't exist
        if 'structured_format' not in existing_columns:
            logger.info("Adding structured_format column...")
            cursor.execute("""
                ALTER TABLE raw_docs 
                ADD COLUMN structured_format TEXT
            """)
            logger.info("structured_format column added successfully")
        else:
            logger.info("structured_format column already exists")
        
        # Add formatted_at column if it doesn't exist
        if 'formatted_at' not in existing_columns:
            logger.info("Adding formatted_at column...")
            cursor.execute("""
                ALTER TABLE raw_docs 
                ADD COLUMN formatted_at TIMESTAMP
            """)
            logger.info("formatted_at column added successfully")
        else:
            logger.info("formatted_at column already exists")
        
        # Close database connection
        cursor.close()
        conn.close()
        
        logger.info("Database update completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error updating database: {str(e)}")
        return False

if __name__ == "__main__":
    add_columns() 