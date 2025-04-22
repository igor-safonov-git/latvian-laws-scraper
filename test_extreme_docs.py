#!/usr/bin/env python3
"""
Test script to generate and process documents with extreme sizes.
Creates very large and very small documents, processes them,
and then runs edge case tests to verify they're handled correctly.
"""
import os
import sys
import hashlib
import json
import random
import string
import logging
import argparse
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_extreme_docs")

# Import embedder_enhanced directly for processing
from embedder_enhanced import EnhancedEmbedder

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
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        sys.exit(1)

def compute_id(url: str) -> str:
    """Compute a SHA-256 hash of the URL to use as ID."""
    return hashlib.sha256(url.encode()).hexdigest()

def generate_random_text(length: int) -> str:
    """Generate random text with the specified length."""
    # For very small documents, create a simple message
    if length < 100:
        return f"This is a test document with {length} characters. It's used for testing embedder edge cases."[:length]
    
    # For large documents, create structured content with paragraphs
    paragraphs = []
    chars_remaining = length
    min_paragraph_size = 100
    max_paragraph_size = 300
    
    while chars_remaining > 0:
        paragraph_size = min(chars_remaining, random.randint(min_paragraph_size, max_paragraph_size))
        
        # Generate paragraph with sentence structure
        paragraph = generate_paragraph(paragraph_size)
        paragraphs.append(paragraph)
        
        chars_remaining -= len(paragraph)
    
    # Join paragraphs with double newlines
    result = "\n\n".join(paragraphs)
    
    # Ensure we hit exactly the desired length
    if len(result) < length:
        # Add padding if needed
        padding = "X" * (length - len(result))
        result += padding
    elif len(result) > length:
        # Truncate if needed
        result = result[:length]
    
    return result

def generate_paragraph(size: int) -> str:
    """Generate a paragraph with approximately the desired size."""
    words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", 
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "document", "legal", "law", "regulation", "article", "section", "provision",
        "paragraph", "Latvian", "Europe", "court", "judgment", "case", "rights",
        "obligation", "tax", "penalty", "compliance", "requirement", "authority"
    ]
    
    # Generate sentences
    paragraph = ""
    while len(paragraph) < size:
        # Generate a sentence
        sentence_length = random.randint(5, 15)
        sentence = " ".join(random.choice(words) for _ in range(sentence_length)) + ". "
        paragraph += sentence
    
    # Trim to size
    return paragraph[:size]

def create_extreme_document(conn, url: str, text: str, doc_type: str) -> str:
    """Create a document with the provided text in the database."""
    doc_id = compute_id(url)
    now = datetime.now()
    
    try:
        # Insert into raw_docs
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO raw_docs (id, url, fetched_at, raw_text, translated_text, processed)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE 
                SET fetched_at = %s, raw_text = %s, translated_text = %s, processed = %s
            """, (
                doc_id, url, now, text, text, True,
                now, text, text, True
            ))
        
        logger.info(f"Created {doc_type} document with ID {doc_id} and {len(text)} characters")
        return doc_id
    
    except Exception as e:
        logger.error(f"Error creating extreme document: {str(e)}")
        return None

async def create_test_documents(conn):
    """Create extremely large and very small test documents."""
    # Create very large document (200K characters)
    large_url = "https://test-extreme-large.example.com"
    large_text = generate_random_text(200000)
    large_id = create_extreme_document(conn, large_url, large_text, "extremely large")
    
    # Create very small document (50 characters)
    small_url = "https://test-extreme-small.example.com"
    small_text = generate_random_text(50)
    small_id = create_extreme_document(conn, small_url, small_text, "very small")
    
    # Return the IDs of created documents
    return {"large": large_id, "small": small_id}

async def process_documents(doc_ids: Dict[str, str]):
    """Process the test documents using the enhanced embedder."""
    embedder = EnhancedEmbedder()
    
    if not embedder.setup():
        logger.error("Failed to set up embedder, exiting")
        return False
    
    # For each document, run the full processing
    async with aiohttp.ClientSession() as session:
        for doc_type, doc_id in doc_ids.items():
            if not doc_id:
                continue
                
            logger.info(f"Processing {doc_type} document with ID {doc_id}")
            
            with embedder.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                # Get document from raw_docs
                cursor.execute("""
                    SELECT id, url, fetched_at, translated_text
                    FROM raw_docs
                    WHERE id = %s
                """, (doc_id,))
                doc = cursor.fetchone()
                
                if doc:
                    # Process the document
                    result = await embedder.process_document(session, doc)
                    logger.info(f"Document processing result: {result}")
                else:
                    logger.error(f"Document {doc_id} not found in raw_docs")
    
    return True

async def run_edge_case_tests():
    """Run the edge case tests to verify document handling."""
    logger.info("Running edge case tests to verify extreme document handling")
    os.system("python test_embedder_edge_cases.py")

async def main():
    """Main function that coordinates the test."""
    parser = argparse.ArgumentParser(description="Test extreme document sizes")
    parser.add_argument("--clean", action="store_true", help="Clean up test documents after running")
    args = parser.parse_args()
    
    conn = get_connection()
    
    try:
        # Create extreme test documents
        doc_ids = await create_test_documents(conn)
        
        # Process documents
        await process_documents(doc_ids)
        
        # Run edge case tests
        await run_edge_case_tests()
        
        # Clean up if requested
        if args.clean:
            logger.info("Cleaning up test documents")
            with conn.cursor() as cursor:
                for doc_type, doc_id in doc_ids.items():
                    if doc_id:
                        # Delete from doc_chunks
                        cursor.execute("DELETE FROM doc_chunks WHERE id = %s", (doc_id,))
                        
                        # Delete from doc_summaries
                        cursor.execute("DELETE FROM doc_summaries WHERE id = %s", (doc_id,))
                        
                        # Delete from docs
                        cursor.execute("DELETE FROM docs WHERE id = %s", (doc_id,))
                        
                        # Delete from raw_docs
                        cursor.execute("DELETE FROM raw_docs WHERE id = %s", (doc_id,))
                        
                        logger.info(f"Deleted test document {doc_id}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    asyncio.run(main())