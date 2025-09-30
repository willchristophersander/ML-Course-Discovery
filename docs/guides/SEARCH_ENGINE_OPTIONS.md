# Search Engine Options Guide

## Why ChatGPT Can Google Search vs. Our System

### **ChatGPT's Approach:**
- **Official API Access**: Uses Google's official Search API or similar services
- **Rate Limiting & Authentication**: Proper API keys and rate limiting
- **Structured Results**: Returns clean, structured data
- **No Scraping**: Doesn't scrape HTML like our system was trying to do

### **Our System's Previous Problem:**
- **Web Scraping**: We were trying to scrape Google's search results HTML
- **Bot Detection**: Google actively blocks automated scraping
- **No API Access**: We don't have official Google Search API access

## Available Search Engine Options

### **1. DuckDuckGo (FREE - No API Key Required)**
```python
from src.core.search_engines import DuckDuckGoSearch

# Instantiate and use
ddg = DuckDuckGoSearch()
results = await ddg.search("University of California Berkeley course search")
```

**Pros:**
-  Completely free
-  No API key required
-  Privacy-focused
-  Good for basic searches

**Cons:**
-  Limited results (Instant Answer API)
-  Not as comprehensive as Google
-  May not find all course search pages

### **2. Bing Search API (Requires API Key)**
```python
from src.core.search_engines import BingSearchAPI

# Get API key from: https://www.microsoft.com/en-us/bing/apis/bing-web-search-api
bing = BingSearchAPI(api_key="your_bing_api_key")
results = await bing.search("University of California Berkeley course search")
```

**Pros:**
-  Comprehensive search results
-  Good for finding course pages
-  Reliable and fast
-  1000 free searches per month

**Cons:**
-  Requires API key
-  Limited free tier
-  Paid after free tier

### **3. SerpAPI (Paid Service)**
```python
from src.core.search_engines import SerpAPISearch

# Get API key from: https://serpapi.com/
serp = SerpAPISearch(api_key="your_serp_api_key")
results = await serp.search("University of California Berkeley course search")
```

**Pros:**
-  Access to Google search results
-  Very comprehensive
-  Good for finding course pages
-  Multiple search engines supported

**Cons:**
-  Paid service
-  Requires API key
-  Can be expensive for high volume

### **4. Google Custom Search API (Requires API Key)**
```python
from src.core.search_engines import GoogleCustomSearch

# Get API key from: https://developers.google.com/custom-search/v1/overview
# Create search engine at: https://cse.google.com/
google = GoogleCustomSearch(
    api_key="your_google_api_key",
    engine_id="your_search_engine_id"
)
results = await google.search("University of California Berkeley course search")
```

**Pros:**
-  Official Google search results
-  Most comprehensive
-  Best for finding course pages
-  100 free searches per day

**Cons:**
-  Requires API key and search engine setup
-  Limited free tier
-  Paid after free tier

### **5. Direct Domain Construction (Current Approach)**
```python
# This is what we're currently using successfully
# Tries common patterns like:
# - https://classes.{domain}
# - https://courses.{domain}
# - https://www.{domain}/classes
# - https://www.{domain}/courses
```

**Pros:**
-  No API keys required
-  Very fast
-  Works well for known universities
-  Free and unlimited

**Cons:**
-  Limited to known patterns
-  May miss some course pages
-  Requires domain mapping

## Recommended Setup

### **For Development/Testing (FREE):**
```python
from src.core.search_engines import MultiSearchEngine

# Use DuckDuckGo (free) + Direct domain construction
multi_search = MultiSearchEngine()
results = await multi_search.search_all("University of California Berkeley course search")
```

### **For Production (PAID):**
```python
from src.core.search_engines import MultiSearchEngine

# Use all available engines
multi_search = MultiSearchEngine(
    bing_api_key="your_bing_key",
    serp_api_key="your_serp_key",
    google_api_key="your_google_key",
    google_engine_id="your_engine_id"
)
results = await multi_search.search_all("University of California Berkeley course search")
```

## How to Get API Keys

### **Bing Search API:**
1. Go to: https://www.microsoft.com/en-us/bing/apis/bing-web-search-api
2. Sign up for Azure account
3. Create a Bing Search resource
4. Get your API key from the Azure portal

### **SerpAPI:**
1. Go to: https://serpapi.com/
2. Sign up for an account
3. Get your API key from the dashboard
4. Choose a plan (starts at $50/month)

### **Google Custom Search:**
1. Go to: https://developers.google.com/custom-search/v1/overview
2. Create a Google Cloud project
3. Enable Custom Search API
4. Get your API key
5. Go to: https://cse.google.com/
6. Create a custom search engine
7. Get your search engine ID

## Configuration

Add your API keys to `config/api_keys.json`:

```json
{
    "search_engines": {
        "bing_api_key": "your_bing_key_here",
        "serp_api_key": "your_serp_key_here",
        "google_custom_search_api_key": "your_google_key_here",
        "google_custom_search_engine_id": "your_engine_id_here"
    }
}
```

## Current Status

Our system is currently working well with:
-  **Direct domain construction** (finding course pages)
-  **Known course page database** (major universities)
-  **Fallback strategies** (multiple search methods)

The search engine options above can be integrated to enhance the system further, but the current approach is already finding course search pages successfully!

## Testing

Test different search engines:

```bash
# Test DuckDuckGo (free)
python src/core/search_engines.py

# Test with your API keys
python -c "
import asyncio
from src.core.search_engines import MultiSearchEngine
multi = MultiSearchEngine(bing_api_key='your_key')
results = asyncio.run(multi.search_all('UC Berkeley course search'))
print(f'Found {len(results)} results')
"
``` 