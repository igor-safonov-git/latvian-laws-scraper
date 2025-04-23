#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
import sys
from tabulate import tabulate
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("test_embedder_enhanced")

def get_database_stats():
    """Get statistics from the database about enhanced embeddings tables."""
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)
    
    try:
        conn = psycopg2.connect(database_url)
        conn.autocommit = True
        
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            # Check if tables exist
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'docs'
                ) AS docs_exists,
                EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'doc_chunks'
                ) AS chunks_exists,
                EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'doc_summaries'
                ) AS summaries_exists
            """)
            
            tables_status = cursor.fetchone()
            
            if not all([tables_status["docs_exists"], 
                       tables_status["chunks_exists"], 
                       tables_status["summaries_exists"]]):
                logger.error("Some enhanced embedder tables don't exist")
                for table, exists in zip(
                    ["docs", "doc_chunks", "doc_summaries"], 
                    [tables_status["docs_exists"], tables_status["chunks_exists"], tables_status["summaries_exists"]]
                ):
                    logger.info(f"Table {table}: {'EXISTS' if exists else 'MISSING'}")
                return
            
            # Get statistics
            stats = []
            
            # Count documents
            cursor.execute("SELECT COUNT(*) FROM docs")
            docs_count = cursor.fetchone()[0]
            stats.append(["Documents", docs_count])
            
            # Count chunks
            cursor.execute("SELECT COUNT(*) FROM doc_chunks")
            chunks_count = cursor.fetchone()[0]
            stats.append(["Document Chunks", chunks_count])
            
            # Average chunks per document
            if docs_count > 0:
                cursor.execute("""
                    SELECT AVG(doc_chunk_count) FROM (
                        SELECT id, COUNT(*) as doc_chunk_count 
                        FROM doc_chunks 
                        GROUP BY id
                    ) as chunk_counts
                """)
                avg_chunks = cursor.fetchone()[0]
                stats.append(["Average Chunks per Document", f"{avg_chunks:.2f}"])
            
            # Count summaries
            cursor.execute("SELECT COUNT(*) FROM doc_summaries")
            summaries_count = cursor.fetchone()[0]
            stats.append(["Document Summaries", summaries_count])
            
            # Check coverage
            cursor.execute("""
                SELECT id FROM docs
                WHERE id NOT IN (SELECT id FROM doc_summaries)
            """)
            missing_summaries = cursor.fetchall()
            if missing_summaries:
                stats.append(["Documents Missing Summaries", len(missing_summaries)])
            
            cursor.execute("""
                SELECT id FROM docs
                WHERE id NOT IN (SELECT DISTINCT id FROM doc_chunks)
                AND (metadata->>'char_length')::int > 3000
            """)
            large_docs_no_chunks = cursor.fetchall()
            if large_docs_no_chunks:
                stats.append(["Large Documents Missing Chunks", len(large_docs_no_chunks)])
            
            # Sample data for verification
            if docs_count > 0:
                cursor.execute("""
                    SELECT id, metadata->>'url' as url, metadata->>'char_length' as length
                    FROM docs
                    ORDER BY RANDOM()
                    LIMIT 1
                """)
                sample_doc = cursor.fetchone()
                
                if sample_doc:
                    doc_id = sample_doc["id"]
                    stats.append(["Sample Document ID", doc_id])
                    stats.append(["Sample Document URL", sample_doc["url"]])
                    stats.append(["Sample Document Length", sample_doc["length"]])
                    
                    # Get summary for this document
                    cursor.execute("""
                        SELECT summary_text, metadata->>'summary_length' as length
                        FROM doc_summaries
                        WHERE id = %s
                    """, (doc_id,))
                    summary = cursor.fetchone()
                    
                    if summary:
                        stats.append(["Sample Summary Length", summary["length"]])
                        preview = summary["summary_text"][:100] + "..." if len(summary["summary_text"]) > 100 else summary["summary_text"]
                        stats.append(["Sample Summary Preview", preview])
                    
                    # Get chunks for this document
                    cursor.execute("""
                        SELECT COUNT(*) 
                        FROM doc_chunks
                        WHERE id = %s
                    """, (doc_id,))
                    doc_chunks = cursor.fetchone()[0]
                    stats.append(["Sample Document Chunks", doc_chunks])
            
            print("\n" + tabulate(stats, headers=["Metric", "Value"], tablefmt="pretty"))
        
        conn.close()
    
    except Exception as e:
        logger.error(f"Error querying database: {str(e)}")
        sys.exit(1)

def check_vector_dimensions():
    """Check if vector dimensions are consistent across tables."""
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)
    
    try:
        conn = psycopg2.connect(database_url)
        conn.autocommit = True
        
        with conn.cursor() as cursor:
            # Check docs table
            cursor.execute("""
                SELECT array_length(embedding, 1) as dimensions 
                FROM docs 
                LIMIT 1
            """)
            docs_dimensions = cursor.fetchone()
            
            # Check doc_chunks table
            cursor.execute("""
                SELECT array_length(embedding, 1) as dimensions 
                FROM doc_chunks 
                LIMIT 1
            """)
            chunks_dimensions = cursor.fetchone()
            
            # Check doc_summaries table
            cursor.execute("""
                SELECT array_length(embedding, 1) as dimensions 
                FROM doc_summaries 
                LIMIT 1
            """)
            summaries_dimensions = cursor.fetchone()
            
            results = [
                ["Full Documents", docs_dimensions[0] if docs_dimensions else "No data"],
                ["Document Chunks", chunks_dimensions[0] if chunks_dimensions else "No data"],
                ["Document Summaries", summaries_dimensions[0] if summaries_dimensions else "No data"]
            ]
            
            print("\nVector Dimensions:")
            print(tabulate(results, headers=["Table", "Dimensions"], tablefmt="pretty"))
        
        conn.close()
    
    except Exception as e:
        logger.error(f"Error checking vector dimensions: {str(e)}")

def main():
    """Run the enhanced embedder tests."""
    parser = argparse.ArgumentParser(description="Test the enhanced document embedder.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Testing enhanced embedder...")
    get_database_stats()
    check_vector_dimensions()
    logger.info("Enhanced embedder test completed.")

if __name__ == "__main__":
    main()