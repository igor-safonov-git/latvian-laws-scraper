#!/usr/bin/env python3
from flask import Flask, jsonify
import os
import psycopg2
from dotenv import load_dotenv
import json
from pathlib import Path
from datetime import datetime, timezone

app = Flask(__name__)

# Load environment variables
load_dotenv()
database_url = os.getenv("DATABASE_URL")

@app.route('/')
def index():
    """Simple status page showing the project is alive."""
    return jsonify({
        "status": "online",
        "name": "Latvian Laws Scraper",
        "description": "Asynchronous scraper for Latvian legal documents",
        "time": datetime.now(timezone.utc).isoformat()
    })

@app.route('/status')
def status():
    """Detailed status information about the scraper."""
    status_data = {
        "database": {"status": "unknown"},
        "scraper": {"status": "unknown"},
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
                
                # Translation percentage
                if record_count > 0:
                    status_data["database"]["translation_percentage"] = round((translated_count / record_count) * 100, 2)
            
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
    
    # Check log file for latest run
    log_path = Path("./logs/scraper.log")
    if log_path.exists():
        status_data["scraper"]["status"] = "logs_found"
        try:
            # Read the last few lines of the log to find latest run
            with open(log_path, "r") as f:
                # Get file size and seek near the end if it's large
                f.seek(max(0, os.path.getsize(log_path) - 10000))
                # Skip partial line
                if f.tell() > 0:
                    f.readline()
                # Read the last lines
                last_lines = f.readlines()
            
            # Parse JSON entries
            entries = []
            for line in last_lines:
                try:
                    data = json.loads(line.strip())
                    if isinstance(data, dict) and "ts" in data:
                        entries.append(data)
                except:
                    continue  # Skip non-JSON lines
            
            if entries:
                entries.sort(key=lambda x: x.get("ts", ""), reverse=True)
                status_data["latest_run"] = entries[0]
        except Exception as e:
            status_data["scraper"]["log_error"] = str(e)
    else:
        status_data["scraper"]["status"] = "no_logs"
    
    return jsonify(status_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)