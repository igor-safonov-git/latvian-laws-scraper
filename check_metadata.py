import os
import json
import psycopg2

def print_metadata():
    conn = psycopg2.connect(os.environ.get('DATABASE_URL'))
    cur = conn.cursor()
    cur.execute('SELECT id, metadata FROM docs LIMIT 5')
    rows = cur.fetchall()
    
    for i, row in enumerate(rows):
        print(f"Document {i}:")
        print(f"  ID: {row[0]}")
        print(f"  Metadata type: {type(row[1])}")
        
        if hasattr(row[1], 'keys'):
            print(f"  Metadata keys: {list(row[1].keys())}")
            if 'text_preview' in row[1]:
                print(f"  text_preview: {row[1]['text_preview']}")
            else:
                print("  No text_preview field in metadata")
        else:
            print("  Metadata is not a dictionary-like object")
            
        print("---")
    
    conn.close()

if __name__ == "__main__":
    print_metadata()