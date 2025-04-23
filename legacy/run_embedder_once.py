#!/usr/bin/env python3
"""
Script to run the embedder once for testing.
"""
import asyncio
from embedder_optimized import OptimizedEmbedderService, run_once

if __name__ == "__main__":
    asyncio.run(run_once())