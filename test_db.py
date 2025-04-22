#!/usr/bin/env python3
import os
import sys
import psycopg2
from dotenv import load_dotenv
import argparse
from tabulate import tabulate

def main():
    """
    Tests the database to ensure the scraper has processed content.
    Verifies that there is actual content in the database after the scraper runs.
    """
    parser = argparse.ArgumentParser(description='Test the database for scraper content')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Error: DATABASE_URL environment variable not set")
        sys.exit(1)

    try:
        # Connect to the database
        print("Connecting to database...")
        conn = psycopg2.connect(database_url)
        conn.autocommit = True
        
        # Test 1: Check if raw_docs table exists
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'raw_docs'
                );
            """)
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                print("❌ Test failed: 'raw_docs' table does not exist")
                sys.exit(1)
            else:
                print("✅ Test passed: 'raw_docs' table exists")

        # Test 2: Check if there are any records in raw_docs
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM raw_docs")
            record_count = cursor.fetchone()[0]
            
            if record_count == 0:
                print("❌ Test failed: No records found in 'raw_docs' table")
                sys.exit(1)
            else:
                print(f"✅ Test passed: Found {record_count} records in 'raw_docs' table")

        # Test 3: Check if records have actual content
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) FROM raw_docs 
                WHERE raw_text IS NULL OR LENGTH(raw_text) = 0
            """)
            empty_count = cursor.fetchone()[0]
            
            if empty_count > 0:
                print(f"❌ Test failed: {empty_count} records have empty raw_text")
                sys.exit(1)
            else:
                print("✅ Test passed: All records have content in raw_text field")
                
        # Test 3.5: Check for translated_text column
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'raw_docs' AND column_name = 'translated_text'
            """)
            has_translated_column = cursor.fetchone() is not None
            
            if not has_translated_column:
                print("⚠️ Note: 'translated_text' column not found in raw_docs table")
            else:
                print("✅ Test passed: 'translated_text' column exists in raw_docs table")
                
                # If we have the column, check for translations
                cursor.execute("""
                    SELECT COUNT(*) FROM raw_docs 
                    WHERE processed = TRUE AND translated_text IS NOT NULL
                """)
                translated_count = cursor.fetchone()[0]
                total_count = record_count  # from previous test
                
                if translated_count > 0:
                    print(f"✅ Info: {translated_count}/{total_count} records have been translated")
        
        # Test 4: Check URL uniqueness
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT url, COUNT(*) 
                FROM raw_docs 
                GROUP BY url 
                HAVING COUNT(*) > 1
            """)
            duplicates = cursor.fetchall()
            
            if duplicates:
                print("❌ Test failed: Found duplicate URLs:")
                for url, count in duplicates:
                    print(f"  - {url}: {count} occurrences")
                sys.exit(1)
            else:
                print("✅ Test passed: No duplicate URLs found")

        # If verbose, display the scraped content information
        if args.verbose:
            with conn.cursor() as cursor:
                # Check if translated_text column exists
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'raw_docs' AND column_name = 'translated_text'
                """)
                has_translated_column = cursor.fetchone() is not None
                
                if has_translated_column:
                    cursor.execute("""
                        SELECT id, url, fetched_at, 
                               SUBSTRING(raw_text, 1, 40) || '...' AS text_preview,
                               processed,
                               CASE WHEN translated_text IS NOT NULL 
                                    THEN SUBSTRING(translated_text, 1, 40) || '...' 
                                    ELSE NULL 
                               END AS translation_preview
                        FROM raw_docs
                        ORDER BY fetched_at DESC
                    """)
                    rows = cursor.fetchall()
                    
                    if rows:
                        print("\nContent:")
                        headers = ["ID", "URL", "Fetched At", "Raw Text Preview", "Processed", "Translation Preview"]
                        print(tabulate(rows, headers=headers, tablefmt="grid"))
                    else:
                        print("No content to display")
                else:
                    cursor.execute("""
                        SELECT id, url, fetched_at, 
                               SUBSTRING(raw_text, 1, 50) || '...' AS text_preview,
                               processed
                        FROM raw_docs
                        ORDER BY fetched_at DESC
                    """)
                    rows = cursor.fetchall()
                    
                    if rows:
                        print("\nContent:")
                        headers = ["ID", "URL", "Fetched At", "Text Preview", "Processed"]
                        print(tabulate(rows, headers=headers, tablefmt="grid"))
                    else:
                        print("No content to display")

        print("\nAll tests passed! The scraper is working correctly.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()