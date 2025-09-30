#!/usr/bin/env python3
"""
Multiple Search Engine Options

This module provides different search approaches:
1. DuckDuckGo (no API key needed)
2. Bing Search API (requires API key)
3. SerpAPI (paid service)
4. RapidAPI (free tier available)
5. Serper.dev (free tier available)
6. Direct domain construction (current approach)
"""

import asyncio
import json
import logging
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from urllib.parse import quote_plus
import time
import random

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result"""
    title: str
    url: str
    snippet: str
    domain: str
    source: str
    confidence: float = 0.0

class DuckDuckGoSearch:
    """DuckDuckGo search implementation (no API key required)"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using DuckDuckGo API (free, no key required)"""
        try:
            logger.info(f"Searching DuckDuckGo API for: {query}")
            
            # Use DuckDuckGo's instant answer API (free, no key required)
            api_url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
            
            response = self.session.get(api_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Get instant answer if available
            if data.get('AbstractURL'):
                results.append(SearchResult(
                    title=data.get('Abstract', 'DuckDuckGo Result'),
                    url=data.get('AbstractURL', ''),
                    snippet=data.get('Abstract', ''),
                    domain=self._extract_domain(data.get('AbstractURL', '')),
                    source='duckduckgo_api',
                    confidence=0.8
                ))
            
            # Get related topics
            for topic in data.get('RelatedTopics', [])[:max_results-1]:
                if isinstance(topic, dict) and topic.get('FirstURL'):
                    results.append(SearchResult(
                        title=topic.get('Text', 'Related Topic'),
                        url=topic.get('FirstURL', ''),
                        snippet=topic.get('Text', ''),
                        domain=self._extract_domain(topic.get('FirstURL', '')),
                        source='duckduckgo_api',
                        confidence=0.6
                    ))
            
            logger.info(f"Found {len(results)} DuckDuckGo API results")
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo API search failed: {e}")
            return []
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return url

class RapidAPISearch:
    """RapidAPI search implementation (free tier available)"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.endpoint = "https://google-search3.p.rapidapi.com/api/v1/search"
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using RapidAPI Google Search (free tier: 100 requests/month)"""
        if not self.api_key:
            logger.warning("RapidAPI key not provided")
            return []
        
        try:
            logger.info(f"Searching RapidAPI for: {query}")
            
            headers = {
                'X-RapidAPI-Key': self.api_key,
                'X-RapidAPI-Host': 'google-search3.p.rapidapi.com'
            }
            
            params = {
                'q': query,
                'num': max_results
            }
            
            response = requests.get(self.endpoint, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('results', []):
                results.append(SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    domain=self._extract_domain(item.get('link', '')),
                    source='rapidapi',
                    confidence=0.8
                ))
            
            logger.info(f"Found {len(results)} RapidAPI results")
            return results
            
        except Exception as e:
            logger.error(f"RapidAPI search failed: {e}")
            return []
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return url

class SerperDevSearch:
    """Serper.dev search implementation (free tier available)"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.endpoint = "https://google.serper.dev/search"
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using Serper.dev (free tier: 100 requests/month)"""
        if not self.api_key:
            logger.warning("Serper.dev API key not provided")
            return []
        
        try:
            logger.info(f"Searching Serper.dev for: {query}")
            
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'q': query,
                'num': max_results
            }
            
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('organic', []):
                results.append(SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    domain=self._extract_domain(item.get('link', '')),
                    source='serper_dev',
                    confidence=0.8
                ))
            
            logger.info(f"Found {len(results)} Serper.dev results")
            return results
            
        except Exception as e:
            logger.error(f"Serper.dev search failed: {e}")
            return []
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return url

class BingSearchAPI:
    """Bing Search API implementation (requires API key)"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using Bing Search API"""
        if not self.api_key:
            logger.warning("Bing Search API key not provided")
            return []
        
        try:
            logger.info(f"Searching Bing for: {query}")
            
            headers = {
                'Ocp-Apim-Subscription-Key': self.api_key
            }
            
            params = {
                'q': query,
                'count': max_results,
                'mkt': 'en-US'
            }
            
            response = requests.get(self.endpoint, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('webPages', {}).get('value', []):
                results.append(SearchResult(
                    title=item.get('name', ''),
                    url=item.get('url', ''),
                    snippet=item.get('snippet', ''),
                    domain=self._extract_domain(item.get('url', '')),
                    source='bing_api',
                    confidence=0.7
                ))
            
            logger.info(f"Found {len(results)} Bing results")
            return results
            
        except Exception as e:
            logger.error(f"Bing search failed: {e}")
            return []
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return url

class SerpAPISearch:
    """SerpAPI implementation (paid service)"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.endpoint = "https://serpapi.com/search"
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using SerpAPI"""
        if not self.api_key:
            logger.warning("SerpAPI key not provided")
            return []
        
        try:
            logger.info(f"Searching SerpAPI for: {query}")
            
            params = {
                'q': query,
                'api_key': self.api_key,
                'engine': 'google',
                'num': max_results
            }
            
            response = requests.get(self.endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('organic_results', []):
                results.append(SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    domain=self._extract_domain(item.get('link', '')),
                    source='serpapi',
                    confidence=0.8
                ))
            
            logger.info(f"Found {len(results)} SerpAPI results")
            return results
            
        except Exception as e:
            logger.error(f"SerpAPI search failed: {e}")
            return []
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return url

class GoogleCustomSearch:
    """Google Custom Search API implementation (requires API key)"""
    
    def __init__(self, api_key: str = None, engine_id: str = None):
        self.api_key = api_key
        self.engine_id = engine_id
        self.endpoint = "https://www.googleapis.com/customsearch/v1"
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using Google Custom Search API"""
        if not self.api_key or not self.engine_id:
            logger.warning("Google Custom Search API key or engine ID not provided")
            return []
        
        try:
            logger.info(f"Searching Google Custom Search for: {query}")
            
            params = {
                'key': self.api_key,
                'cx': self.engine_id,
                'q': query,
                'num': min(max_results, 10)  # Google CSE max is 10
            }
            
            response = requests.get(self.endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', []):
                results.append(SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    domain=self._extract_domain(item.get('link', '')),
                    source='google_cse',
                    confidence=0.9
                ))
            
            logger.info(f"Found {len(results)} Google Custom Search results")
            return results
            
        except Exception as e:
            logger.error(f"Google Custom Search failed: {e}")
            return []
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return url

class MultiSearchEngine:
    """Multi-search engine that tries multiple sources"""
    
    def __init__(self, bing_api_key: str = None, serp_api_key: str = None, 
                 google_api_key: str = None, google_engine_id: str = None,
                 rapid_api_key: str = None, serper_api_key: str = None):
        self.engines = []
        
        # Add free engines first
        self.engines.append(('duckduckgo', DuckDuckGoSearch()))
        
        # Add paid engines if keys provided
        if rapid_api_key:
            self.engines.append(('rapidapi', RapidAPISearch(rapid_api_key)))
        if serper_api_key:
            self.engines.append(('serper_dev', SerperDevSearch(serper_api_key)))
        if bing_api_key:
            self.engines.append(('bing', BingSearchAPI(bing_api_key)))
        if serp_api_key:
            self.engines.append(('serpapi', SerpAPISearch(serp_api_key)))
        if google_api_key and google_engine_id:
            self.engines.append(('google_cse', GoogleCustomSearch(google_api_key, google_engine_id)))
    
    async def search_all(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using all available engines"""
        all_results = []
        
        for engine_name, engine in self.engines:
            try:
                results = await engine.search(query, max_results)
                all_results.extend(results)
                logger.info(f"Engine {engine_name} found {len(results)} results")
            except Exception as e:
                logger.warning(f"Engine {engine_name} failed: {e}")
                continue
        
        # Remove duplicates while preserving order
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        logger.info(f"Total unique results: {len(unique_results)}")
        return unique_results[:max_results]

class SimpleWebSearch:
    """Simple web search using multiple sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using multiple web sources"""
        try:
            logger.info(f"Searching web for: {query}")
            
            results = []
            
            # Try multiple search approaches
            search_methods = [
                self._search_duckduckgo_html,
                self._search_brave_search,
                self._search_ecosia
            ]
            
            for method in search_methods:
                try:
                    method_results = await method(query, max_results)
                    results.extend(method_results)
                    if len(results) >= max_results:
                        break
                except Exception as e:
                    logger.debug(f"Search method {method.__name__} failed: {e}")
                    continue
            
            # Remove duplicates
            seen_urls = set()
            unique_results = []
            for result in results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    unique_results.append(result)
            
            logger.info(f"Found {len(unique_results)} web search results")
            return unique_results[:max_results]
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    async def _search_duckduckgo_html(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using DuckDuckGo HTML"""
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            
            # Try multiple selectors for DuckDuckGo
            selectors = [
                'a.result__a',
                'a[href^="http"]',
                '.result__title a',
                '.result a',
                'a[data-testid="result-title"]'
            ]
            
            for selector in selectors:
                try:
                    result_links = soup.select(selector)
                    for link in result_links[:max_results]:
                        try:
                            url = link.get('href')
                            if url and url.startswith('http') and not url.startswith('https://duckduckgo.com'):
                                title = link.get_text(strip=True)
                                if title and len(title) > 5:  # Filter out very short titles
                                    domain = self._extract_domain(url)
                                    
                                    results.append(SearchResult(
                                        title=title,
                                        url=url,
                                        snippet="",
                                        domain=domain,
                                        source='duckduckgo_html',
                                        confidence=0.7
                                    ))
                        except Exception as e:
                            continue
                    
                    if results:  # If we found results, stop trying other selectors
                        break
                        
                except Exception as e:
                    continue
            
            return results
            
        except Exception as e:
            logger.debug(f"DuckDuckGo HTML search failed: {e}")
            return []
    
    async def _search_brave_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Brave Search"""
        try:
            search_url = f"https://search.brave.com/search?q={quote_plus(query)}"
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            
            # Try multiple selectors for Brave Search
            selectors = [
                'a[href^="http"]',
                '.result a',
                '.web-result a',
                'a[data-testid="result-title"]',
                'h3 a'
            ]
            
            for selector in selectors:
                try:
                    result_links = soup.select(selector)
                    for link in result_links[:max_results]:
                        try:
                            url = link.get('href')
                            if url and url.startswith('http') and not url.startswith('https://search.brave.com'):
                                title = link.get_text(strip=True)
                                if title and len(title) > 5:  # Filter out very short titles
                                    domain = self._extract_domain(url)
                                    
                                    results.append(SearchResult(
                                        title=title,
                                        url=url,
                                        snippet="",
                                        domain=domain,
                                        source='brave_search',
                                        confidence=0.6
                                    ))
                        except Exception as e:
                            continue
                    
                    if results:  # If we found results, stop trying other selectors
                        break
                        
                except Exception as e:
                    continue
            
            return results
            
        except Exception as e:
            logger.debug(f"Brave search failed: {e}")
            return []
    
    async def _search_ecosia(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Ecosia"""
        try:
            search_url = f"https://www.ecosia.org/search?q={quote_plus(query)}"
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            
            # Try multiple selectors for Ecosia
            selectors = [
                'a[href^="http"]',
                '.result a',
                '.web-result a',
                'h3 a'
            ]
            
            for selector in selectors:
                try:
                    result_links = soup.select(selector)
                    for link in result_links[:max_results]:
                        try:
                            url = link.get('href')
                            if url and url.startswith('http') and not url.startswith('https://www.ecosia.org'):
                                title = link.get_text(strip=True)
                                if title and len(title) > 5:  # Filter out very short titles
                                    domain = self._extract_domain(url)
                                    
                                    results.append(SearchResult(
                                        title=title,
                                        url=url,
                                        snippet="",
                                        domain=domain,
                                        source='ecosia',
                                        confidence=0.5
                                    ))
                        except Exception as e:
                            continue
                    
                    if results:  # If we found results, stop trying other selectors
                        break
                        
                except Exception as e:
                    continue
            
            return results
            
        except Exception as e:
            logger.debug(f"Ecosia search failed: {e}")
            return []
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return url

async def test_search_engines():
    """Test all available search engines"""
    print("Testing search engines...")
    
    # Test free engines
    duckduckgo = DuckDuckGoSearch()
    results = await duckduckgo.search("Stanford University course search", 3)
    print(f"DuckDuckGo API: {len(results)} results")
    
    # Test HTML scraping (if available)
    web_search = SimpleWebSearch()
    results = await web_search.search("Stanford University course search", 3)
    print(f"Web scraping: {len(results)} results")
    
    # Test multi-engine (free only)
    multi = MultiSearchEngine()
    results = await multi.search_all("Stanford University course search", 5)
    print(f"Multi-engine (free): {len(results)} results")
    
    for i, result in enumerate(results[:3]):
        print(f"{i+1}. {result.title}")
        print(f"   URL: {result.url}")
        print(f"   Source: {result.source}")

if __name__ == "__main__":
    asyncio.run(test_search_engines()) 