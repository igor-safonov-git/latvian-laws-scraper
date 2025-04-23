# Exported Documents Directory

This directory contains exported document files from the database. 

## How to generate exports

Run the following command to export all documents from the database:

```bash
python export_documents.py
```

This will create text files for each document and an index.html file that lists all documents.

## Options

- `--output-dir DIR` - Directory to write files to (default: ./static/export)
- `--limit N` - Limit to N most recent documents (default: all documents)

Example:

```bash
# Export the 10 most recent documents
python export_documents.py --limit 10

# Export to a different directory
python export_documents.py --output-dir ./custom-export-dir
```

## Web Access

Once the files are generated, they can be accessed via the web interface at:

https://latvian-laws-06e89c613b8a.herokuapp.com/export

This page shows an index of all exported documents with links to view them.