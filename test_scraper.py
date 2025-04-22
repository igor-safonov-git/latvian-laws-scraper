#!/usr/bin/env python3
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import json
from datetime import datetime

async def fetch_url(session, url):
    """Fetch a URL and return its HTML content."""
    try:
        async with session.get(url, timeout=60) as response:
            if response.status != 200:
                print(f"Failed to fetch {url}: HTTP {response.status}")
                return None
            return await response.text()
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return None

def extract_text(html):
    """Extract plain text from HTML."""
    if not html:
        return None
    
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    
    # Get text
    text = soup.get_text(separator="\n")
    
    # Remove extra whitespace
    lines = (line.strip() for line in text.splitlines())
    text = "\n".join(line for line in lines if line)
    
    return text

async def process_url(session, url):
    """Process a single URL: fetch, extract text, save to JSON."""
    print(f"Processing {url}")
    
    html = await fetch_url(session, url)
    if not html:
        print(f"Failed to fetch URL: {url}")
        return None
    
    text = extract_text(html)
    if not text:
        print(f"Failed to extract text: {url}")
        return None
    
    # Create output data
    now = datetime.now().isoformat()
    data = {
        "url": url,
        "fetched_at": now,
        "raw_text": text[:1000]  # Just get the first 1000 chars for the test
    }
    
    return data

async def main():
    """Main function to process an example URL."""
    url = "https://likumi.lv/ta/id/56880-par-iedzivotaju-ienakuma-nodokli"
    
    async with aiohttp.ClientSession() as session:
        result = await process_url(session, url)
        
    if result:
        print("Success! First 100 characters of text:")
        print(result["raw_text"][:100])
        
        # Save to file
        with open("example_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("Saved to example_result.json")

if __name__ == "__main__":
    asyncio.run(main())