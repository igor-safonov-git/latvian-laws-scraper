#!/usr/bin/env python3
import os
import sys
import json
import argparse
import psycopg2
from dotenv import load_dotenv
from tabulate import tabulate

def main():
    """
    Tests the translator functionality to ensure proper translation of documents.
    Verifies that documents have been translated and marked as processed.
    """
    parser = argparse.ArgumentParser(description='Test the translator functionality')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--check-one', '-c', action='store_true', help='Check a single document')
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
        
        # Test 1: Check if translated_text column exists
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'raw_docs' AND column_name = 'translated_text'
            """)
            column_exists = cursor.fetchone()
            
            if not column_exists:
                print("❌ Test failed: 'translated_text' column does not exist in raw_docs table")
                sys.exit(1)
            else:
                print("✅ Test passed: 'translated_text' column exists in raw_docs table")

        # Test 2: Check if any documents have been translated
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) FROM raw_docs 
                WHERE translated_text IS NOT NULL
            """)
            translated_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM raw_docs")
            total_count = cursor.fetchone()[0]
            
            if translated_count == 0:
                print(f"❌ Test failed: No documents have been translated (0/{total_count})")
                if not args.check_one:
                    sys.exit(1)
            else:
                print(f"✅ Test passed: {translated_count}/{total_count} documents have been translated")
                
                # Additional check for translation quality
                if args.verbose:
                    # Get percentage
                    percentage = (translated_count / total_count) * 100 if total_count > 0 else 0
                    print(f"   Translation progress: {percentage:.2f}%")

        # Test 3: Check if translated documents are marked as processed
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) FROM raw_docs 
                WHERE translated_text IS NOT NULL AND processed = FALSE
            """)
            inconsistent_count = cursor.fetchone()[0]
            
            if inconsistent_count > 0:
                print(f"❌ Test failed: {inconsistent_count} translated documents are not marked as processed")
                sys.exit(1)
            else:
                print("✅ Test passed: All translated documents are correctly marked as processed")

        # Test 4: Check translation content (if any translations exist)
        if translated_count > 0:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT COUNT(*) FROM raw_docs 
                    WHERE translated_text IS NOT NULL AND LENGTH(translated_text) < 10
                """)
                short_translations = cursor.fetchone()[0]
                
                if short_translations > 0:
                    print(f"⚠️ Warning: {short_translations} documents have suspiciously short translations")
                else:
                    print("✅ Test passed: All translations have reasonable length")

        # If check-one is specified, check a single document
        if args.check_one:
            with conn.cursor() as cursor:
                # Find a document that needs translation
                cursor.execute("""
                    SELECT id, url, raw_text 
                    FROM raw_docs 
                    WHERE translated_text IS NULL AND raw_text IS NOT NULL
                    LIMIT 1
                """)
                doc = cursor.fetchone()
                
                if doc:
                    doc_id, url, raw_text = doc
                    print(f"\nFound document needing translation: {url}")
                    print(f"Document ID: {doc_id}")
                    
                    # Check if we have DeepL API key
                    deepl_api_key = os.getenv("DEEPL_API_KEY")
                    if not deepl_api_key:
                        print("⚠️ Warning: DEEPL_API_KEY environment variable not set")
                        print("   Cannot test translation functionality")
                    else:
                        print("DeepL API key found, manually testing translation...")
                        
                        # Test snippet of text (first 200 chars)
                        test_text = raw_text[:200] if len(raw_text) > 200 else raw_text
                        
                        # Use curl to test the DeepL API directly
                        import subprocess
                        try:
                            curl_cmd = [
                                "curl", "-s", "-X", "POST",
                                "https://api-free.deepl.com/v2/translate",
                                "-H", f"Authorization: DeepL-Auth-Key {deepl_api_key}",
                                "-H", "Content-Type: application/json",
                                "-d", json.dumps({
                                    "text": [test_text],
                                    "source_lang": "LV",
                                    "target_lang": "EN"
                                })
                            ]
                            result = subprocess.run(curl_cmd, capture_output=True, text=True)
                            
                            if result.returncode == 0:
                                response = json.loads(result.stdout)
                                if "translations" in response and len(response["translations"]) > 0:
                                    translation = response["translations"][0]["text"]
                                    print("\nSample translation test:")
                                    print(f"Original (Latvian): {test_text}")
                                    print(f"Translated (English): {translation}")
                                    print("\n✅ Translation API test successful!")
                                else:
                                    print(f"❌ Translation API returned invalid response: {response}")
                            else:
                                print(f"❌ Translation API test failed: {result.stderr}")
                        except Exception as e:
                            print(f"❌ Error testing translation: {str(e)}")
                else:
                    print("\nNo documents found that need translation")

        # If verbose, display the documents with their translation status
        if args.verbose:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        id, 
                        url, 
                        fetched_at, 
                        processed,
                        SUBSTRING(raw_text, 1, 40) || '...' AS raw_preview,
                        CASE 
                            WHEN translated_text IS NOT NULL THEN 
                                SUBSTRING(translated_text, 1, 40) || '...'
                            ELSE 'Not translated'
                        END AS translation_preview,
                        CASE
                            WHEN translated_text IS NOT NULL THEN 
                                LENGTH(translated_text)
                            ELSE 0
                        END AS translation_length
                    FROM raw_docs
                    ORDER BY processed ASC, fetched_at DESC
                """)
                rows = cursor.fetchall()
                
                if rows:
                    print("\nDocument Translation Status:")
                    headers = ["ID", "URL", "Fetched At", "Processed", "Raw Text Preview", "Translation Preview", "Trans. Length"]
                    print(tabulate(rows, headers=headers, tablefmt="grid"))
                else:
                    print("No documents to display")

        print("\nAll translator tests completed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()