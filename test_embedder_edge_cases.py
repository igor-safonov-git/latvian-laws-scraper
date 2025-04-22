#!/usr/bin/env python3
"""
Edge case testing for the enhanced embedder functionality.
This script tests various challenging scenarios for the embedder system.
"""
import os
import sys
import json
import argparse
import logging
import random
from typing import Dict, Any, List, Optional
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from tabulate import tabulate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_embedder_edge")

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

def test_extremely_long_documents(conn, verbose: bool = False) -> List[Dict[str, Any]]:
    """Test edge case: Handling of extremely long documents."""
    results = []
    
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
        # Check for very long documents (>100K characters)
        cursor.execute("""
            SELECT id, url, LENGTH(translated_text) AS text_length
            FROM raw_docs
            WHERE LENGTH(translated_text) > 100000
            ORDER BY LENGTH(translated_text) DESC
            LIMIT 5
        """)
        long_docs = cursor.fetchall()
        
        if not long_docs:
            results.append({
                "test": "Extremely Long Documents",
                "status": "Skipped",
                "details": "No documents longer than 100K characters found"
            })
            return results
        
        long_doc_count = len(long_docs)
        results.append({
            "test": "Extremely Long Documents",
            "status": "Found",
            "details": f"Found {long_doc_count} documents longer than 100K characters"
        })
        
        if verbose:
            doc_details = []
            for doc in long_docs:
                doc_details.append([doc["id"][:10] + "...", doc["url"], f"{doc['text_length']:,} chars"])
            
            print("\nExtremely Long Documents:")
            print(tabulate(doc_details, headers=["ID", "URL", "Length"], tablefmt="pretty"))
        
        # Check if these documents have chunks
        for doc in long_docs:
            doc_id = doc["id"]
            cursor.execute("""
                SELECT COUNT(*) 
                FROM doc_chunks
                WHERE id = %s
            """, (doc_id,))
            chunk_count = cursor.fetchone()[0]
            
            # Calculate expected chunk count (very rough estimate)
            text_length = doc["text_length"]
            chunk_size = int(os.getenv("CHUNK_SIZE", "3000"))
            chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "500"))
            effective_chunk_size = chunk_size - chunk_overlap
            expected_chunks = max(1, text_length // effective_chunk_size)
            
            if chunk_count < expected_chunks * 0.5:  # Allow for some flexibility in chunk count
                results.append({
                    "test": f"Chunks for Long Doc {doc_id[:8]}...",
                    "status": "Warning",
                    "details": f"Expected ~{expected_chunks} chunks, found {chunk_count}"
                })
            else:
                results.append({
                    "test": f"Chunks for Long Doc {doc_id[:8]}...",
                    "status": "Pass",
                    "details": f"Found {chunk_count} chunks (expected ~{expected_chunks})"
                })
    
    return results

def test_empty_or_tiny_documents(conn, verbose: bool = False) -> List[Dict[str, Any]]:
    """Test edge case: Handling of empty or very small documents."""
    results = []
    
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
        # Check for very small documents (<100 characters)
        cursor.execute("""
            SELECT id, url, LENGTH(translated_text) AS text_length
            FROM raw_docs
            WHERE LENGTH(translated_text) < 100 AND translated_text IS NOT NULL
            ORDER BY LENGTH(translated_text) ASC
            LIMIT 5
        """)
        small_docs = cursor.fetchall()
        
        if not small_docs:
            results.append({
                "test": "Very Small Documents",
                "status": "Skipped",
                "details": "No documents smaller than 100 characters found"
            })
            return results
        
        small_doc_count = len(small_docs)
        results.append({
            "test": "Very Small Documents",
            "status": "Found",
            "details": f"Found {small_doc_count} documents smaller than 100 characters"
        })
        
        if verbose:
            doc_details = []
            for doc in small_docs:
                doc_details.append([doc["id"][:10] + "...", doc["url"], f"{doc['text_length']} chars"])
            
            print("\nVery Small Documents:")
            print(tabulate(doc_details, headers=["ID", "URL", "Length"], tablefmt="pretty"))
        
        # Check if these documents have embeddings and summaries
        for doc in small_docs:
            doc_id = doc["id"]
            
            # Check for embeddings
            cursor.execute("""
                SELECT COUNT(*) 
                FROM docs
                WHERE id = %s
            """, (doc_id,))
            embedding_exists = cursor.fetchone()[0] > 0
            
            # Check for summaries
            cursor.execute("""
                SELECT COUNT(*) 
                FROM doc_summaries
                WHERE id = %s
            """, (doc_id,))
            summary_exists = cursor.fetchone()[0] > 0
            
            # Chunks should not exist for tiny docs (they should be handled as single chunks)
            cursor.execute("""
                SELECT COUNT(*) 
                FROM doc_chunks
                WHERE id = %s
            """, (doc_id,))
            chunk_count = cursor.fetchone()[0]
            
            if embedding_exists and summary_exists:
                results.append({
                    "test": f"Processing of Small Doc {doc_id[:8]}...",
                    "status": "Pass",
                    "details": f"Has embedding and summary, {chunk_count} chunks"
                })
            else:
                results.append({
                    "test": f"Processing of Small Doc {doc_id[:8]}...",
                    "status": "Fail",
                    "details": f"Missing {'embedding' if not embedding_exists else ''} {'summary' if not summary_exists else ''}"
                })
    
    return results

def test_special_characters(conn, verbose: bool = False) -> List[Dict[str, Any]]:
    """Test edge case: Documents with unusual characters or formatting."""
    results = []
    
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
        # Check for documents with unusual characters (high percentage of non-alphanumeric)
        cursor.execute("""
            SELECT id, url, translated_text
            FROM raw_docs
            WHERE translated_text ~ '[^a-zA-Z0-9\\s]'
            LIMIT 10
        """)
        special_char_docs = cursor.fetchall()
        
        if not special_char_docs:
            results.append({
                "test": "Special Characters",
                "status": "Skipped",
                "details": "No documents with special characters found"
            })
            return results
        
        # Sample a few docs to analyze
        sampled_docs = random.sample(special_char_docs, min(3, len(special_char_docs)))
        
        for doc in sampled_docs:
            doc_id = doc["id"]
            text = doc["translated_text"]
            
            # Count special characters
            alpha_count = sum(1 for c in text if c.isalnum())
            space_count = sum(1 for c in text if c.isspace())
            special_count = len(text) - alpha_count - space_count
            special_ratio = special_count / len(text) if len(text) > 0 else 0
            
            # Check if embedding exists
            cursor.execute("""
                SELECT COUNT(*) 
                FROM docs
                WHERE id = %s
            """, (doc_id,))
            has_embedding = cursor.fetchone()[0] > 0
            
            # Check if summary exists
            cursor.execute("""
                SELECT COUNT(*) 
                FROM doc_summaries
                WHERE id = %s
            """, (doc_id,))
            has_summary = cursor.fetchone()[0] > 0
            
            if has_embedding and has_summary:
                results.append({
                    "test": f"Special Chars Doc {doc_id[:8]}...",
                    "status": "Pass",
                    "details": f"{special_count} special chars ({special_ratio:.1%}), has embedding and summary"
                })
            else:
                results.append({
                    "test": f"Special Chars Doc {doc_id[:8]}...",
                    "status": "Fail",
                    "details": f"Missing {'embedding' if not has_embedding else ''} {'summary' if not has_summary else ''}"
                })
            
            if verbose:
                special_chars = set(c for c in text if not c.isalnum() and not c.isspace())
                print(f"\nSpecial characters in doc {doc_id[:8]}...:")
                print(f"Total special characters: {special_count} ({special_ratio:.1%} of text)")
                print(f"Unique special characters: {sorted(special_chars)}")
    
    return results

def test_null_placeholder_embeddings(conn, verbose: bool = False) -> List[Dict[str, Any]]:
    """Test edge case: Check for NULL or placeholder embeddings."""
    results = []
    
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
        # Check for NULL embeddings in main docs table
        cursor.execute("""
            SELECT COUNT(*) 
            FROM docs
            WHERE embedding IS NULL
        """)
        null_embeddings = cursor.fetchone()[0]
        
        if null_embeddings > 0:
            results.append({
                "test": "NULL Embeddings (docs)",
                "status": "Fail",
                "details": f"Found {null_embeddings} records with NULL embeddings"
            })
        else:
            results.append({
                "test": "NULL Embeddings (docs)",
                "status": "Pass",
                "details": "No NULL embeddings found"
            })
        
        # Check for NULL embeddings in chunks table
        cursor.execute("""
            SELECT COUNT(*) 
            FROM doc_chunks
            WHERE embedding IS NULL
        """)
        null_chunk_embeddings = cursor.fetchone()[0]
        
        if null_chunk_embeddings > 0:
            results.append({
                "test": "NULL Embeddings (chunks)",
                "status": "Fail",
                "details": f"Found {null_chunk_embeddings} chunks with NULL embeddings"
            })
        else:
            results.append({
                "test": "NULL Embeddings (chunks)",
                "status": "Pass",
                "details": "No NULL chunk embeddings found"
            })
        
        # Check for NULL embeddings in summaries table
        cursor.execute("""
            SELECT COUNT(*) 
            FROM doc_summaries
            WHERE embedding IS NULL
        """)
        null_summary_embeddings = cursor.fetchone()[0]
        
        if null_summary_embeddings > 0:
            results.append({
                "test": "NULL Embeddings (summaries)",
                "status": "Fail",
                "details": f"Found {null_summary_embeddings} summaries with NULL embeddings"
            })
        else:
            results.append({
                "test": "NULL Embeddings (summaries)",
                "status": "Pass",
                "details": "No NULL summary embeddings found"
            })
        
        # Check for placeholder embeddings (if metadata indicates placeholder)
        cursor.execute("""
            SELECT COUNT(*) 
            FROM docs
            WHERE metadata->>'placeholder' = 'true'
        """)
        placeholder_count = cursor.fetchone()[0]
        
        if placeholder_count > 0:
            # This is not necessarily a fail, but worth noting
            results.append({
                "test": "Placeholder Embeddings",
                "status": "Warning",
                "details": f"Found {placeholder_count} placeholder embeddings"
            })
            
            if verbose:
                # Get a sample of docs with placeholder embeddings
                cursor.execute("""
                    SELECT id, metadata->>'url' as url
                    FROM docs
                    WHERE metadata->>'placeholder' = 'true'
                    LIMIT 3
                """)
                placeholders = cursor.fetchall()
                
                print("\nSample Documents with Placeholder Embeddings:")
                for doc in placeholders:
                    print(f"- ID: {doc['id'][:10]}..., URL: {doc['url']}")
        else:
            results.append({
                "test": "Placeholder Embeddings",
                "status": "Pass",
                "details": "No placeholder embeddings found"
            })
    
    return results

def test_orphaned_records(conn, verbose: bool = False) -> List[Dict[str, Any]]:
    """Test edge case: Check for orphaned records or reference integrity issues."""
    results = []
    
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
        # Check for docs with embeddings but no raw_docs entry
        cursor.execute("""
            SELECT COUNT(*) 
            FROM docs
            WHERE id NOT IN (SELECT id FROM raw_docs)
        """)
        orphaned_embeddings = cursor.fetchone()[0]
        
        if orphaned_embeddings > 0:
            results.append({
                "test": "Orphaned Embeddings",
                "status": "Fail",
                "details": f"Found {orphaned_embeddings} embeddings without raw documents"
            })
            
            if verbose:
                cursor.execute("""
                    SELECT id, metadata->>'url' as url
                    FROM docs
                    WHERE id NOT IN (SELECT id FROM raw_docs)
                    LIMIT 3
                """)
                orphans = cursor.fetchall()
                
                print("\nSample Orphaned Embeddings:")
                for doc in orphans:
                    print(f"- ID: {doc['id'][:10]}..., URL: {doc.get('url', 'N/A')}")
        else:
            results.append({
                "test": "Orphaned Embeddings",
                "status": "Pass",
                "details": "No orphaned embeddings found"
            })
        
        # Check for chunks without parent docs
        cursor.execute("""
            SELECT COUNT(*) 
            FROM doc_chunks
            WHERE id NOT IN (SELECT id FROM docs)
        """)
        orphaned_chunks = cursor.fetchone()[0]
        
        if orphaned_chunks > 0:
            results.append({
                "test": "Orphaned Chunks",
                "status": "Fail",
                "details": f"Found {orphaned_chunks} chunks without parent documents"
            })
        else:
            results.append({
                "test": "Orphaned Chunks",
                "status": "Pass",
                "details": "No orphaned chunks found"
            })
        
        # Check for summaries without parent docs
        cursor.execute("""
            SELECT COUNT(*) 
            FROM doc_summaries
            WHERE id NOT IN (SELECT id FROM docs)
        """)
        orphaned_summaries = cursor.fetchone()[0]
        
        if orphaned_summaries > 0:
            results.append({
                "test": "Orphaned Summaries",
                "status": "Fail",
                "details": f"Found {orphaned_summaries} summaries without parent documents"
            })
        else:
            results.append({
                "test": "Orphaned Summaries",
                "status": "Pass",
                "details": "No orphaned summaries found"
            })
        
        # Check for documents missing chunks (only for docs that should have chunks)
        cursor.execute("""
            SELECT COUNT(*) 
            FROM docs d
            WHERE (metadata->>'char_length')::int > 3000
            AND NOT EXISTS (
                SELECT 1 FROM doc_chunks c WHERE c.id = d.id
            )
        """)
        docs_missing_chunks = cursor.fetchone()[0]
        
        if docs_missing_chunks > 0:
            results.append({
                "test": "Docs Missing Chunks",
                "status": "Warning",
                "details": f"Found {docs_missing_chunks} large docs without chunks"
            })
        else:
            results.append({
                "test": "Docs Missing Chunks",
                "status": "Pass",
                "details": "All large documents have chunks"
            })
    
    return results

def test_embedding_dimensions(conn, verbose: bool = False) -> List[Dict[str, Any]]:
    """Test edge case: Check for consistency in embedding dimensions."""
    results = []
    dimensions = {}
    
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
        # Check docs table dimensions
        cursor.execute("""
            SELECT DISTINCT array_length(embedding, 1) as dimensions
            FROM docs
            LIMIT 10
        """)
        doc_dimensions = [row[0] for row in cursor.fetchall()]
        dimensions["docs"] = doc_dimensions
        
        # Check chunks table dimensions
        cursor.execute("""
            SELECT DISTINCT array_length(embedding, 1) as dimensions
            FROM doc_chunks
            LIMIT 10
        """)
        chunk_dimensions = [row[0] for row in cursor.fetchall()]
        dimensions["chunks"] = chunk_dimensions
        
        # Check summaries table dimensions
        cursor.execute("""
            SELECT DISTINCT array_length(embedding, 1) as dimensions
            FROM doc_summaries
            LIMIT 10
        """)
        summary_dimensions = [row[0] for row in cursor.fetchall()]
        dimensions["summaries"] = summary_dimensions
        
        # Check if dimensions are consistent within each table
        for table, dims in dimensions.items():
            if not dims:
                results.append({
                    "test": f"Embedding Dimensions ({table})",
                    "status": "Skipped",
                    "details": f"No embeddings found in {table} table"
                })
            elif len(dims) > 1:
                results.append({
                    "test": f"Embedding Dimensions ({table})",
                    "status": "Fail",
                    "details": f"Inconsistent dimensions found: {dims}"
                })
            else:
                results.append({
                    "test": f"Embedding Dimensions ({table})",
                    "status": "Pass",
                    "details": f"Consistent dimension: {dims[0]}"
                })
        
        # Check if dimensions are consistent across tables
        all_dims = set()
        for dims in dimensions.values():
            all_dims.update(dims)
        
        if len(all_dims) > 1:
            results.append({
                "test": "Cross-Table Dimensions",
                "status": "Fail",
                "details": f"Dimensions vary across tables: {all_dims}"
            })
        elif len(all_dims) == 1:
            results.append({
                "test": "Cross-Table Dimensions",
                "status": "Pass",
                "details": f"All tables use dimension: {list(all_dims)[0]}"
            })
    
    return results

def main():
    """Run edge case tests for the embedder functionality."""
    parser = argparse.ArgumentParser(description="Test embedder edge cases")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    
    conn = get_connection()
    
    # Run all edge case tests
    all_results = []
    
    print("\n=== RUNNING EMBEDDER EDGE CASE TESTS ===\n")
    
    # Test extremely long documents
    print("Testing extremely long documents...")
    all_results.extend(test_extremely_long_documents(conn, args.verbose))
    
    # Test empty or tiny documents
    print("Testing very small documents...")
    all_results.extend(test_empty_or_tiny_documents(conn, args.verbose))
    
    # Test documents with special characters
    print("Testing documents with special characters...")
    all_results.extend(test_special_characters(conn, args.verbose))
    
    # Test for NULL or placeholder embeddings
    print("Testing for NULL or placeholder embeddings...")
    all_results.extend(test_null_placeholder_embeddings(conn, args.verbose))
    
    # Test for orphaned records
    print("Testing for orphaned records...")
    all_results.extend(test_orphaned_records(conn, args.verbose))
    
    # Test embedding dimensions
    print("Testing embedding dimensions...")
    all_results.extend(test_embedding_dimensions(conn, args.verbose))
    
    # Summarize all test results
    print("\n=== EMBEDDER EDGE CASE TEST RESULTS ===\n")
    
    # Format results for tabulate
    table_data = []
    for result in all_results:
        status_symbol = {
            "Pass": "✅",
            "Fail": "❌",
            "Warning": "⚠️",
            "Skipped": "⏭️",
            "Found": "🔍"
        }.get(result["status"], "")
        
        table_data.append([
            result["test"],
            f"{status_symbol} {result['status']}",
            result["details"]
        ])
    
    print(tabulate(table_data, headers=["Test", "Status", "Details"], tablefmt="pretty"))
    
    # Summary
    pass_count = sum(1 for r in all_results if r["status"] == "Pass")
    fail_count = sum(1 for r in all_results if r["status"] == "Fail")
    warning_count = sum(1 for r in all_results if r["status"] == "Warning")
    skipped_count = sum(1 for r in all_results if r["status"] in ["Skipped", "Found"])
    
    print(f"\nSummary: {pass_count} passed, {fail_count} failed, {warning_count} warnings, {skipped_count} skipped/informational")
    
    if fail_count > 0:
        print("\n⚠️ Some tests failed - review issues above")
        return 1
    elif warning_count > 0:
        print("\n⚠️ Some tests generated warnings - review potential issues above")
        return 0
    else:
        print("\n✅ All tests passed successfully!")
        return 0

if __name__ == "__main__":
    sys.exit(main())