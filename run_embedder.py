#!/usr/bin/env python3
import asyncio
from embedder import Embedder

async def main():
    """Run the embedder job with proper setup."""
    embedder = Embedder()
    
    # Set up the service first
    if not embedder.setup():
        print("Failed to set up embedder service, exiting")
        return
    
    # Force a full regeneration of embeddings by clearing the old ones
    if not embedder.clear_embeddings():
        print("Failed to clear embeddings, exiting")
        return
    
    # Run the job
    success = await embedder.run_job()
    
    if success:
        print("Embedder job completed successfully!")
    else:
        print("Embedder job failed.")

if __name__ == "__main__":
    asyncio.run(main())