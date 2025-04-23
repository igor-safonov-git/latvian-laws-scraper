-- SQL script to add required columns for the formatter
-- Adds structured_format (TEXT) and formatted_at (TIMESTAMP) columns to raw_docs table

-- Check if structured_format column exists and add if it doesn't
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'raw_docs' 
        AND column_name = 'structured_format'
    ) THEN
        ALTER TABLE raw_docs ADD COLUMN structured_format TEXT;
        RAISE NOTICE 'Added structured_format column';
    ELSE
        RAISE NOTICE 'structured_format column already exists';
    END IF;
END $$;

-- Check if formatted_at column exists and add if it doesn't
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'raw_docs' 
        AND column_name = 'formatted_at'
    ) THEN
        ALTER TABLE raw_docs ADD COLUMN formatted_at TIMESTAMP;
        RAISE NOTICE 'Added formatted_at column';
    ELSE
        RAISE NOTICE 'formatted_at column already exists';
    END IF;
END $$; 