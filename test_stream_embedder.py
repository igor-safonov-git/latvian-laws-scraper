#!/usr/bin/env python3
"""
Test script for the streaming embedder service that handles dimension detection
and memory-efficient document processing.
"""
import os
import sys
import hashlib
import random
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any

import psycopg2
import psycopg2.extras
import aiohttp
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_stream_embedder")

# Import the streaming embedder directly
from embedder_optimized import OptimizedEmbedderService, AsyncDatabaseConnector

def get_connection():
    """Get a database connection."""
    # Load environment variables
    load_dotenv()
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)
    
    try:
        # Connect to database
        conn = psycopg2.connect(database_url)
        conn.autocommit = True
        return conn, database_url
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        sys.exit(1)

def generate_random_text(size_kb: int) -> str:
    """Generate random text with the specified size in kilobytes."""
    target_size = size_kb * 1024  # Convert to bytes
    paragraphs = []
    
    # Generate paragraphs until we reach the target size
    while sum(len(p) + 2 for p in paragraphs) < target_size:  # +2 for newlines
        # Generate random paragraph
        words = []
        for _ in range(random.randint(50, 150)):
            word_len = random.randint(3, 12)
            word = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(word_len))
            words.append(word)
        
        paragraph = ' '.join(words)
        paragraphs.append(paragraph)
    
    # Join paragraphs with double newlines
    text = '\n\n'.join(paragraphs)
    
    # Ensure exact size
    if len(text) > target_size:
        text = text[:target_size]
    
    return text

def create_test_document(conn, size_kb: int) -> Dict[str, Any]:
    """Create a test document with the specified size."""
    # Generate random text
    text = generate_random_text(size_kb)
    
    # Create a unique URL and ID
    url = f"https://test-stream-embedder-{size_kb}kb.example.com"
    doc_id = hashlib.sha256(url.encode()).hexdigest()
    
    try:
        # Insert into raw_docs
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO raw_docs (id, url, fetched_at, raw_text, translated_text, processed)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE 
                SET fetched_at = %s, raw_text = %s, translated_text = %s, processed = %s
                RETURNING id
            """, (
                doc_id, url, datetime.now(), text, text, False,
                datetime.now(), text, text, False
            ))
            result = cursor.fetchone()[0]
        
        logger.info(f"Created test document with ID {doc_id} and size {size_kb}KB ({len(text)} chars)")
        
        return {
            "id": doc_id,
            "url": url,
            "size": len(text),
            "size_kb": size_kb
        }
    
    except Exception as e:
        logger.error(f"Error creating test document: {str(e)}")
        return None

async def test_streaming_embedder(database_url: str, doc_info: Dict[str, Any]):
    """Test the streaming embedder with the specified document."""
    logger.info(f"Testing streaming embedder with {doc_info['size_kb']}KB document")
    
    # Initialize the optimized embedder service
    service = OptimizedEmbedderService()
    
    # Set up the service
    setup_success = await service.setup()
    if not setup_success:
        logger.error("Failed to set up embedder service")
        return False
    
    # Connect to database
    if not await service.db.connect():
        logger.error("Failed to connect to database")
        return False
    
    try:
        # Use a modified approach to fetch the document directly
        async with service.db.conn.transaction():
            # Directly fetch our test document
            query = """
                SELECT id AS trans_id, url, fetched_at, translated_text
                FROM raw_docs
                WHERE id = $1 AND translated_text IS NOT NULL
            """
            record = await service.db.conn.fetchrow(query, doc_info["id"])
            
            if not record:
                logger.error(f"Test document {doc_info['id']} not found")
                return False
                
            # Convert to a dictionary
            doc = {
                "trans_id": record["trans_id"],
                "url": record["url"],
                "fetched_at": record["fetched_at"],
                "translated_text": record["translated_text"]
            }
            
            logger.info(f"Found test document with {len(doc['translated_text'])} characters")
            
            # Process the document
            async with aiohttp.ClientSession() as session:
                start_time = asyncio.get_event_loop().time()
                success_chunks, failed_chunks = await service.process_document(doc, session)
                end_time = asyncio.get_event_loop().time()
            
        # Log results
        logger.info(f"Document processing completed in {end_time - start_time:.2f}s")
        logger.info(f"Successful chunks: {success_chunks}, Failed chunks: {failed_chunks}")
        
        # Check if document was marked as processed
        with psycopg2.connect(database_url).cursor() as cursor:
            cursor.execute("SELECT processed FROM raw_docs WHERE id = %s", (doc_info["id"],))
            processed = cursor.fetchone()[0]
            
            if processed:
                logger.info("Document was successfully marked as processed")
            else:
                logger.error("Document was not marked as processed")
            
            # Get vector counts
            cursor.execute("SELECT COUNT(*) FROM docs WHERE metadata->>'url' = %s", (doc_info["url"],))
            vector_count = cursor.fetchone()[0]
            logger.info(f"Found {vector_count} vectors in the database for this document")
        
        return success_chunks > 0 and failed_chunks == 0
    
    except Exception as e:
        logger.error(f"Error during streaming embedder test: {str(e)}")
        return False
    
    finally:
        # Close database connection
        await service.db.disconnect()

async def main():
    """Main function to run the test."""
    conn, database_url = get_connection()
    
    try:
        # Create a test document (100KB)
        doc_info = create_test_document(conn, 100)
        
        if not doc_info:
            logger.error("Failed to create test document")
            return
        
        # Test the streaming embedder
        success = await test_streaming_embedder(database_url, doc_info)
        
        if success:
            logger.info("Streaming embedder test completed successfully!")
        else:
            logger.error("Streaming embedder test failed")
        
    finally:
        conn.close()

if __name__ == "__main__":
    asyncio.run(main())