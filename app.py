#!/usr/bin/env python3
from flask import Flask, jsonify, send_from_directory
import os
import sys
import psycopg2
from dotenv import load_dotenv
import json
from pathlib import Path
from datetime import datetime, timezone

app = Flask(__name__, static_folder='static')

# Load environment variables
load_dotenv()
database_url = os.getenv("DATABASE_URL")

@app.route('/')
def index():
    """Simple status page showing the project is alive."""
    # Check if exports are available
    if "DYNO" in os.environ:
        export_dir = "/tmp/export"
    else:
        export_dir = os.path.join(app.static_folder, 'export')
    
    export_available = os.path.exists(os.path.join(export_dir, "index.html"))
    
    return jsonify({
        "status": "online",
        "name": "Latvian Laws Scraper",
        "description": "Asynchronous scraper for Latvian legal documents",
        "time": datetime.now(timezone.utc).isoformat(),
        "export_url": "/export" if export_available else None
    })

@app.route('/export')
def export_index():
    """Serve the index.html file from the export directory."""
    # Check if running on Heroku
    if "DYNO" in os.environ:
        export_dir = "/tmp/export"
    else:
        export_dir = os.path.join(app.static_folder, 'export')
    
    # Ensure index exists
    if not os.path.exists(os.path.join(export_dir, 'index.html')):
        return jsonify({
            "error": "No exports available.",
            "run_export": "To generate exports, visit /run-export endpoint"
        }), 404
    
    return send_from_directory(export_dir, 'index.html')

@app.route('/run-export')
def run_export():
    """Run the export_documents script to generate exports."""
    try:
        # Import the export_documents module
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from export_documents import export_documents
        
        # Set up export directory (use /tmp on Heroku)
        export_dir = "/tmp/export" if "DYNO" in os.environ else os.path.join(app.static_folder, 'export')
        
        # Run the export function
        success = export_documents(export_dir)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Document export completed successfully",
                "url": "/export"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to export documents"
            }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error running export: {str(e)}"
        }), 500

@app.route('/export/<path:filename>')
def export_file(filename):
    """Serve a specific file from the export directory."""
    # Check if running on Heroku
    if "DYNO" in os.environ:
        export_dir = "/tmp/export"
    else:
        export_dir = os.path.join(app.static_folder, 'export')
    
    # Ensure file exists
    if not os.path.exists(os.path.join(export_dir, filename)):
        return jsonify({
            "error": f"File '{filename}' not found."
        }), 404
        
    return send_from_directory(export_dir, filename)

@app.route('/status')
def status():
    """Detailed status information about the scraper, translator and embedder."""
    status_data = {
        "database": {"status": "unknown"},
        "scraper": {"status": "unknown"},
        "translator": {"status": "unknown"},
        "embedder": {"status": "unknown"},
        "latest_run": None
    }
    
    # Check database connection
    try:
        conn = psycopg2.connect(database_url)
        conn.autocommit = True
        status_data["database"]["status"] = "connected"
        
        # Get record count
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM raw_docs")
            record_count = cursor.fetchone()[0]
            status_data["database"]["record_count"] = record_count
            
            # Check for translated_text column
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'raw_docs' AND column_name = 'translated_text'
            """)
            has_translated_column = cursor.fetchone() is not None
            
            # Get translation stats if column exists
            if has_translated_column:
                cursor.execute("SELECT COUNT(*) FROM raw_docs WHERE translated_text IS NOT NULL")
                translated_count = cursor.fetchone()[0]
                status_data["database"]["translated_count"] = translated_count
                
                cursor.execute("SELECT COUNT(*) FROM raw_docs WHERE processed = TRUE")
                processed_count = cursor.fetchone()[0]
                status_data["database"]["processed_count"] = processed_count
                
                cursor.execute("SELECT COUNT(*) FROM raw_docs WHERE processed = FALSE")
                pending_count = cursor.fetchone()[0]
                status_data["database"]["pending_count"] = pending_count
                
                # Translation percentage
                if record_count > 0:
                    status_data["database"]["translation_percentage"] = round((translated_count / record_count) * 100, 2)
                    
                # Try to get information about recently skipped documents
                try:
                    cursor.execute("""
                        SELECT COUNT(*) FROM raw_docs 
                        WHERE translated_text IS NOT NULL 
                        AND processed = TRUE 
                        AND id IN (
                            SELECT id FROM raw_docs 
                            WHERE processed = FALSE 
                            ORDER BY fetched_at DESC 
                            LIMIT 100
                        )
                    """)
                    status_data["database"]["recently_skipped_count"] = cursor.fetchone()[0]
                except Exception:
                    # This is an optional metric, don't fail if it's not available
                    pass
            
            # Get latest record
            if has_translated_column:
                cursor.execute("""
                    SELECT url, fetched_at, LENGTH(raw_text) AS size, 
                           processed, LENGTH(translated_text) AS translated_size
                    FROM raw_docs
                    ORDER BY fetched_at DESC
                    LIMIT 1
                """)
                latest = cursor.fetchone()
                if latest:
                    status_data["database"]["latest_record"] = {
                        "url": latest[0],
                        "fetched_at": latest[1].isoformat(),
                        "size": latest[2],
                        "processed": latest[3],
                        "translated_size": latest[4] if latest[4] else 0
                    }
            else:
                cursor.execute("""
                    SELECT url, fetched_at, LENGTH(raw_text) AS size
                    FROM raw_docs
                    ORDER BY fetched_at DESC
                    LIMIT 1
                """)
                latest = cursor.fetchone()
                if latest:
                    status_data["database"]["latest_record"] = {
                        "url": latest[0],
                        "fetched_at": latest[1].isoformat(),
                        "size": latest[2]
                    }
                
        conn.close()
    except Exception as e:
        status_data["database"]["status"] = "error"
        status_data["database"]["error"] = str(e)
    
    # Check scraper log file for latest run
    scraper_log_path = Path("./logs/scraper.log")
    if scraper_log_path.exists():
        status_data["scraper"]["status"] = "logs_found"
        try:
            # Read the last few lines of the log to find latest run
            with open(scraper_log_path, "r") as f:
                # Get file size and seek near the end if it's large
                f.seek(max(0, os.path.getsize(scraper_log_path) - 10000))
                # Skip partial line
                if f.tell() > 0:
                    f.readline()
                # Read the last lines
                last_lines = f.readlines()
            
            # Parse JSON entries
            scraper_entries = []
            for line in last_lines:
                try:
                    data = json.loads(line.strip())
                    if isinstance(data, dict) and "ts" in data:
                        scraper_entries.append(data)
                except:
                    continue  # Skip non-JSON lines
            
            if scraper_entries:
                scraper_entries.sort(key=lambda x: x.get("ts", ""), reverse=True)
                status_data["latest_run"] = scraper_entries[0]
        except Exception as e:
            status_data["scraper"]["log_error"] = str(e)
    else:
        status_data["scraper"]["status"] = "no_logs"
        
    # Check translator log file
    translator_log_path = Path("./logs/translator.log")
    if translator_log_path.exists():
        status_data["translator"]["status"] = "logs_found"
        try:
            # Read the last few lines of the log to find latest translation
            with open(translator_log_path, "r") as f:
                # Get file size and seek near the end if it's large
                f.seek(max(0, os.path.getsize(translator_log_path) - 10000))
                # Skip partial line
                if f.tell() > 0:
                    f.readline()
                # Read the last lines
                last_lines = f.readlines()
            
            # Parse JSON entries
            translator_entries = []
            for line in last_lines:
                try:
                    data = json.loads(line.strip())
                    if isinstance(data, dict) and "ts" in data:
                        translator_entries.append(data)
                except:
                    continue  # Skip non-JSON lines
            
            if translator_entries:
                translator_entries.sort(key=lambda x: x.get("ts", ""), reverse=True)
                status_data["translator"]["latest_run"] = translator_entries[0]
                
                # Count successes, failures, and skipped
                success_count = sum(1 for entry in translator_entries if entry.get("status") == "ok")
                error_count = sum(1 for entry in translator_entries if entry.get("status") == "error")
                skipped_count = sum(1 for entry in translator_entries if entry.get("status") == "skipped")
                
                status_data["translator"]["stats"] = {
                    "successful_translations": success_count,
                    "failed_translations": error_count,
                    "skipped_translations": skipped_count,
                    "total_attempts": len(translator_entries)
                }
        except Exception as e:
            status_data["translator"]["log_error"] = str(e)
    else:
        status_data["translator"]["status"] = "no_logs"
        
    # Check embedder log file
    embedder_log_path = Path("./logs/embedder.log")
    if embedder_log_path.exists():
        status_data["embedder"]["status"] = "logs_found"
        try:
            # Read the last few lines of the log to find latest embedding run
            with open(embedder_log_path, "r") as f:
                # Get file size and seek near the end if it's large
                f.seek(max(0, os.path.getsize(embedder_log_path) - 10000))
                # Skip partial line
                if f.tell() > 0:
                    f.readline()
                # Read the last lines
                last_lines = f.readlines()
            
            # Parse JSON entries
            embedder_entries = []
            for line in last_lines:
                try:
                    data = json.loads(line.strip())
                    if isinstance(data, dict) and "ts" in data:
                        embedder_entries.append(data)
                except:
                    continue  # Skip non-JSON lines
            
            if embedder_entries:
                embedder_entries.sort(key=lambda x: x.get("ts", ""), reverse=True)
                status_data["embedder"]["latest_run"] = embedder_entries[0]
                
                # Count successes and failures
                success_count = sum(1 for entry in embedder_entries if entry.get("status") == "ok")
                error_count = sum(1 for entry in embedder_entries if entry.get("status") == "error")
                
                status_data["embedder"]["stats"] = {
                    "successful_embeddings": success_count,
                    "failed_embeddings": error_count,
                    "total_attempts": len(embedder_entries)
                }
        except Exception as e:
            status_data["embedder"]["log_error"] = str(e)
    else:
        status_data["embedder"]["status"] = "no_logs"
        
    # Check embeddings stats from database
    try:
        # Make sure we have a valid connection
        if 'conn' in locals() and conn:
            with conn.cursor() as cursor:
                # Check if vector extension is enabled
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 
                        FROM pg_extension 
                        WHERE extname = 'vector'
                    )
                """)
                vector_enabled = cursor.fetchone()[0]
                status_data["embedder"]["vector_enabled"] = vector_enabled
                
                # Check if docs table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'docs'
                    )
                """)
                docs_table_exists = cursor.fetchone()[0]
                
                if docs_table_exists:
                    # Get embedding count
                    cursor.execute("SELECT COUNT(*) FROM docs")
                    embedding_count = cursor.fetchone()[0]
                    status_data["embedder"]["embedding_count"] = embedding_count
                    
                    # Get embedding dimensions if any exist
                    if embedding_count > 0:
                        cursor.execute("SELECT ARRAY_LENGTH(embedding, 1) FROM docs LIMIT 1")
                        dimensions = cursor.fetchone()[0]
                        status_data["embedder"]["dimensions"] = dimensions
                    
                    # Check for last embedding run timestamp
                    try:
                        cursor.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = 'system_info'
                            )
                        """)
                        if cursor.fetchone()[0]:
                            cursor.execute("""
                                SELECT metadata->>'last_embedding_run' 
                                FROM system_info 
                                WHERE id = 'embedder'
                            """)
                            last_run = cursor.fetchone()
                            if last_run and last_run[0]:
                                status_data["embedder"]["last_run"] = last_run[0]
                    except Exception:
                        pass  # Don't fail if we can't get last run info
    except Exception as e:
        # Just log the error but don't fail
        print(f"Error getting embedder stats: {str(e)}")
    
    return jsonify(status_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)