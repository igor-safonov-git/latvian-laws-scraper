#!/usr/bin/env python3
"""
Script to run the enhanced embedder once without scheduling.
Use this for testing or one-off generation of enhanced embeddings.
"""
import asyncio
from embedder_enhanced import EnhancedEmbedder

async def main():
    """Run the enhanced embedder job once."""
    print("Starting one-time enhanced embedder job...")
    embedder = EnhancedEmbedder()
    
    # Set up the service first
    if not embedder.setup():
        print("Failed to set up enhanced embedder service, exiting")
        return
    
    # Run the job
    success = await embedder.run_job()
    
    if success:
        print("Enhanced embedder job completed successfully!")
    else:
        print("Enhanced embedder job failed.")

if __name__ == "__main__":
    asyncio.run(main())