#!/usr/bin/env python3
"""
Script to mark documents as unprocessed in the database so they can be formatted.
This resets the structured_format and formatted_at fields for selected documents.
"""
import os
import logging
import argparse
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("reset_processing")

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

def list_documents(limit=20):
    """List documents in the database with their processing status."""
    try:
        logger.info("Connecting to database...")
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Get document count
        cursor.execute("SELECT COUNT(*) FROM raw_docs WHERE raw_text IS NOT NULL")
        total_count = cursor.fetchone()[0]
        
        # Get processed count
        cursor.execute("SELECT COUNT(*) FROM raw_docs WHERE structured_format IS NOT NULL")
        processed_count = cursor.fetchone()[0]
        
        logger.info(f"Total documents with text: {total_count}")
        logger.info(f"Documents already formatted: {processed_count}")
        logger.info(f"Documents not yet formatted: {total_count - processed_count}")
        
        # List sample documents
        cursor.execute("""
            SELECT id, LEFT(url, 60) as short_url, 
                  structured_format IS NOT NULL as is_formatted,
                  formatted_at
            FROM raw_docs 
            WHERE raw_text IS NOT NULL
            ORDER BY formatted_at DESC NULLS LAST
            LIMIT %s
        """, (limit,))
        
        print("\nSample documents:")
        print("-" * 80)
        print(f"{'ID':<15} | {'URL':<60} | {'Formatted':<10} | {'Formatted At'}")
        print("-" * 80)
        
        for row in cursor.fetchall():
            doc_id, url, is_formatted, formatted_at = row
            print(f"{doc_id:<15} | {url:<60} | {str(is_formatted):<10} | {formatted_at}")
        
        # Close database connection
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")

def reset_processing(doc_ids=None, reset_all=False, limit=None):
    """
    Mark documents as unprocessed by resetting the structured_format and formatted_at fields.
    
    Args:
        doc_ids: List of document IDs to reset, or None
        reset_all: If True, reset all documents
        limit: Maximum number of documents to reset
    """
    try:
        logger.info("Connecting to database...")
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        cursor = conn.cursor()
        
        if reset_all:
            # Reset all documents, optionally with a limit
            if limit:
                cursor.execute("""
                    UPDATE raw_docs 
                    SET structured_format = NULL, formatted_at = NULL
                    WHERE raw_text IS NOT NULL
                    LIMIT %s
                """, (limit,))
            else:
                cursor.execute("""
                    UPDATE raw_docs 
                    SET structured_format = NULL, formatted_at = NULL
                    WHERE raw_text IS NOT NULL
                """)
            
            rows_updated = cursor.rowcount
            logger.info(f"Reset {rows_updated} documents to unprocessed state")
            
        elif doc_ids:
            # Reset specific documents by ID
            placeholders = ','.join(['%s'] * len(doc_ids))
            cursor.execute(f"""
                UPDATE raw_docs 
                SET structured_format = NULL, formatted_at = NULL
                WHERE id IN ({placeholders})
            """, doc_ids)
            
            rows_updated = cursor.rowcount
            logger.info(f"Reset {rows_updated} documents to unprocessed state")
            
        else:
            logger.error("No documents specified to reset")
        
        # Close database connection
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error resetting documents: {str(e)}")
        return False

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Reset document processing status")
    
    # Create a mutually exclusive group for the reset options
    reset_group = parser.add_mutually_exclusive_group(required=True)
    reset_group.add_argument("--list", action="store_true", help="List documents and their status")
    reset_group.add_argument("--reset-all", action="store_true", help="Reset all documents")
    reset_group.add_argument("--reset-ids", nargs="+", help="Reset specific document IDs")
    
    # Optional parameters
    parser.add_argument("--limit", type=int, help="Limit the number of documents to list or reset")
    
    args = parser.parse_args()
    
    if args.list:
        # List documents
        list_documents(limit=args.limit or 20)
    elif args.reset_all:
        # Reset all documents
        reset_processing(reset_all=True, limit=args.limit)
    elif args.reset_ids:
        # Reset specific document IDs
        reset_processing(doc_ids=args.reset_ids)

if __name__ == "__main__":
    main() 