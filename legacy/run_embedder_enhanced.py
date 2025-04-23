#!/usr/bin/env python3
import asyncio
from embedder_enhanced import EnhancedEmbedder

async def main():
    """Run the enhanced embedder job with proper setup."""
    embedder = EnhancedEmbedder()
    
    # Start the service, which will set up tables and run the job
    await embedder.start()
    
if __name__ == "__main__":
    asyncio.run(main())