#!/usr/bin/env python3
"""
Test script to compare memory usage between regular and memory-efficient embedders.
Generates an extremely large document and processes it with both embedders,
measuring peak memory usage for each.
"""
import os
import gc
import sys
import time
import asyncio
import hashlib
import random
import logging
import resource
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple

import psutil
import psycopg2
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_memory_usage")

# Import both embedders
from embedder_enhanced import EnhancedEmbedder
from embedder_memory_efficient import EmbedderStream

def get_memory_usage() -> Tuple[float, float]:
    """Get current memory usage in MB and peak memory usage."""
    process = psutil.Process(os.getpid())
    current_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    
    # Get peak memory usage (Linux/Unix only)
    try:
        peak_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Convert to MB
    except:
        # Fallback for other platforms
        peak_usage = current_usage
        
    return current_usage, peak_usage

def generate_large_text(size_mb: int) -> str:
    """Generate a large text document of the specified size in MB."""
    logger.info(f"Generating {size_mb}MB text document...")
    
    # Estimate chars needed (1 char ≈ 1-2 bytes depending on encoding)
    chars_per_mb = 1000000  # Approximation of chars per MB
    target_chars = chars_per_mb * size_mb
    
    # Generate paragraphs
    paragraphs = []
    chars_generated = 0
    paragraph_size = 1000  # Each paragraph is about 1000 chars
    
    while chars_generated < target_chars:
        # Generate a paragraph with some sentence variation
        sentences = []
        sentence_count = random.randint(3, 8)
        
        for _ in range(sentence_count):
            # Generate random sentence of 10-20 words
            word_count = random.randint(10, 20)
            words = []
            
            for _ in range(word_count):
                # Generate random word length between 2-12 chars
                word_length = random.randint(2, 12)
                word = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(word_length))
                words.append(word)
            
            sentence = ' '.join(words) + '.'
            sentences.append(sentence)
        
        paragraph = ' '.join(sentences)
        paragraphs.append(paragraph)
        chars_generated += len(paragraph) + 2  # +2 for newlines
    
    # Join paragraphs with double newlines
    text = '\n\n'.join(paragraphs)
    
    # Truncate if we generated too much
    if len(text) > target_chars:
        text = text[:target_chars]
    
    logger.info(f"Generated text of {len(text)} characters ({len(text)/chars_per_mb:.2f}MB)")
    return text

async def test_regular_embedder(text: str, doc_id: str) -> Dict[str, Any]:
    """Test the regular (EnhancedEmbedder) with a large document."""
    logger.info("Testing EnhancedEmbedder...")
    
    # Force garbage collection before starting
    gc.collect()
    start_memory, _ = get_memory_usage()
    logger.info(f"Starting memory usage: {start_memory:.2f}MB")
    
    # Create embedder and setup
    embedder = EnhancedEmbedder()
    if not embedder.setup():
        logger.error("Failed to set up regular embedder")
        return {"success": False, "error": "Setup failed"}
    
    # Record time and memory before processing
    start_time = time.time()
    
    # Process document
    async with aiohttp.ClientSession() as session:
        doc = {
            "id": doc_id,
            "url": "memory://test-large-document",
            "fetched_at": datetime.now(),
            "translated_text": text
        }
        
        # Process the document and record peak memory
        await embedder.process_document(session, doc)
    
    # Record results
    end_time = time.time()
    _, peak_memory = get_memory_usage()
    
    return {
        "embedder": "EnhancedEmbedder",
        "document_size": len(text),
        "processing_time": end_time - start_time,
        "start_memory": start_memory,
        "peak_memory": peak_memory,
        "memory_increase": peak_memory - start_memory
    }

async def test_memory_efficient_embedder(text: str, doc_id: str) -> Dict[str, Any]:
    """Test the memory-efficient embedder with a large document."""
    logger.info("Testing EmbedderStream...")
    
    # Force garbage collection before starting
    gc.collect()
    start_memory, _ = get_memory_usage()
    logger.info(f"Starting memory usage: {start_memory:.2f}MB")
    
    # Create embedder
    embedder = EmbedderStream()
    
    # Record time and memory before processing
    start_time = time.time()
    
    # Process document
    result = await embedder.process_large_document(
        doc_id=doc_id,
        url="memory://test-large-document",
        text_or_path=text,
        fetched_at=datetime.now()
    )
    
    # Record results
    end_time = time.time()
    _, peak_memory = get_memory_usage()
    
    return {
        "embedder": "EmbedderStream",
        "document_size": len(text),
        "processing_time": end_time - start_time,
        "start_memory": start_memory,
        "peak_memory": peak_memory,
        "memory_increase": peak_memory - start_memory,
        "result": result
    }

async def run_tests(size_mb: int = 10):
    """Run memory tests for both embedders with the same document."""
    # Generate large text
    text = generate_large_text(size_mb)
    doc_id = hashlib.sha256(b"test-large-document").hexdigest()
    
    # Test regular embedder
    regular_results = await test_regular_embedder(text, doc_id)
    logger.info(f"Regular embedder results:")
    logger.info(f"  - Time: {regular_results['processing_time']:.2f} seconds")
    logger.info(f"  - Peak memory: {regular_results['peak_memory']:.2f}MB")
    logger.info(f"  - Memory increase: {regular_results['memory_increase']:.2f}MB")
    
    # Force cleanup
    gc.collect()
    await asyncio.sleep(1)
    
    # Test memory-efficient embedder
    efficient_results = await test_memory_efficient_embedder(text, doc_id)
    logger.info(f"Memory-efficient embedder results:")
    logger.info(f"  - Time: {efficient_results['processing_time']:.2f} seconds")
    logger.info(f"  - Peak memory: {efficient_results['peak_memory']:.2f}MB")
    logger.info(f"  - Memory increase: {efficient_results['memory_increase']:.2f}MB")
    
    # Compare results
    memory_saving = regular_results['peak_memory'] - efficient_results['peak_memory']
    memory_saving_percent = (memory_saving / regular_results['peak_memory']) * 100
    
    print("\n=== COMPARISON RESULTS ===")
    print(f"Document size: {size_mb}MB ({len(text)} characters)")
    print(f"Regular embedder peak memory: {regular_results['peak_memory']:.2f}MB")
    print(f"Memory-efficient embedder peak memory: {efficient_results['peak_memory']:.2f}MB")
    print(f"Memory saving: {memory_saving:.2f}MB ({memory_saving_percent:.1f}%)")
    
    # Check if efficient embedder processed all chunks successfully
    if efficient_results.get('result', {}).get('success', False):
        print(f"Memory-efficient embedder processed {efficient_results['result']['chunks_processed']} chunks successfully.")
    else:
        print(f"Memory-efficient embedder failed: {efficient_results.get('result', {}).get('error', 'Unknown error')}")

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test memory usage of embedders")
    parser.add_argument("--size", type=int, default=10, help="Size of test document in MB (default: 10)")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Check database connection
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)
    
    try:
        conn = psycopg2.connect(database_url)
        conn.close()
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        sys.exit(1)
    
    # Run memory usage tests
    await run_tests(args.size)

if __name__ == "__main__":
    import aiohttp
    asyncio.run(main())