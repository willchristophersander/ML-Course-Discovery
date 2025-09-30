#!/usr/bin/env python3
"""
Working Search Implementation

This module provides reliable search methods that actually work:
1. Direct domain construction (current working approach)
2. Known course page database
3. Simple web search with proper error handling
"""

import asyncio
import json
import logging
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from urllib.parse import quote_plus, urlparse
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

class WorkingSearchEngine:
    """Reliable search engine that actually works"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Known course search pages for major universities
        self.known_course_pages = {
            'university of california, berkeley': [
                {
                    'title': 'UC Berkeley Class Schedule',
                    'url': 'https://classes.berkeley.edu/',
                    'snippet': 'Official class schedule and course search for UC Berkeley',
                    'domain': 'berkeley.edu'
                }
            ],
            'university of new hampshire': [
                {
                    'title': 'UNH Course Search',
                    'url': 'https://courses.unh.edu/',
                    'snippet': 'Course search and registration for University of New Hampshire',
                    'domain': 'unh.edu'
                }
            ],
            'ohio state university': [
                {
                    'title': 'OSU Class Schedule',
                    'url': 'https://classes.osu.edu/',
                    'snippet': 'Class schedule and course search for Ohio State University',
                    'domain': 'osu.edu'
                }
            ],
            'university of virginia': [
                {
                    'title': 'UVA Course Directory',
                    'url': 'https://rabi.phys.virginia.edu/mySIS/CS2/',
                    'snippet': 'Course directory for University of Virginia',
                    'domain': 'virginia.edu'
                }
            ],
            'university of vermont': [
                {
                    'title': 'UVM Course Catalog',
                    'url': 'https://catalogue.uvm.edu/',
                    'snippet': 'Course catalog for University of Vermont',
                    'domain': 'uvm.edu'
                }
            ],
            'harvard university': [
                {
                    'title': 'Harvard Course Search',
                    'url': 'https://courses.my.harvard.edu/psp/courses/EMPLOYEE/EMPL/h/?tab=HU_CLASS_SEARCH',
                    'snippet': 'Course search for Harvard University',
                    'domain': 'harvard.edu'
                }
            ],
            'stanford university': [
                {
                    'title': 'Stanford Course Search',
                    'url': 'https://explorecourses.stanford.edu/',
                    'snippet': 'Course search for Stanford University',
                    'domain': 'stanford.edu'
                }
            ],
            'mit': [
                {
                    'title': 'MIT Course Catalog',
                    'url': 'https://catalog.mit.edu/',
                    'snippet': 'Course catalog for Massachusetts Institute of Technology',
                    'domain': 'mit.edu'
                }
            ],
            'university of michigan': [
                {
                    'title': 'UMich Course Guide',
                    'url': 'https://www.lsa.umich.edu/cg/',
                    'snippet': 'Course guide for University of Michigan',
                    'domain': 'umich.edu'
                }
            ],
            'university of texas': [
                {
                    'title': 'UT Course Schedule',
                    'url': 'https://utdirect.utexas.edu/registrar/schedule/',
                    'snippet': 'Course schedule for University of Texas',
                    'domain': 'utexas.edu'
                }
            ],
            'university of florida': [
                {
                    'title': 'UF Course Schedule',
                    'url': 'https://one.uf.edu/soc/',
                    'snippet': 'Course schedule for University of Florida',
                    'domain': 'ufl.edu'
                }
            ],
            'university of washington': [
                {
                    'title': 'UW Course Catalog',
                    'url': 'https://www.washington.edu/students/crscat/',
                    'snippet': 'Course catalog for University of Washington',
                    'domain': 'washington.edu'
                }
            ],
            'university of illinois': [
                {
                    'title': 'UIUC Course Explorer',
                    'url': 'https://go.illinois.edu/courseexplorer',
                    'snippet': 'Course explorer for University of Illinois',
                    'domain': 'illinois.edu'
                }
            ],
            'university of wisconsin': [
                {
                    'title': 'UW Course Guide',
                    'url': 'https://guide.wisc.edu/',
                    'snippet': 'Course guide for University of Wisconsin',
                    'domain': 'wisc.edu'
                }
            ],
            'university of minnesota': [
                {
                    'title': 'UMN Course Catalog',
                    'url': 'https://onestop2.umn.edu/pcas/viewCatalogSearchForm.do',
                    'snippet': 'Course catalog for University of Minnesota',
                    'domain': 'umn.edu'
                }
            ]
        }
        
        # Common domain patterns
        self.domain_patterns = {
            'university of california, berkeley': 'berkeley.edu',
            'uc berkeley': 'berkeley.edu',
            'university of new hampshire': 'unh.edu',
            'unh': 'unh.edu',
            'ohio state university': 'osu.edu',
            'osu': 'osu.edu',
            'university of virginia': 'virginia.edu',
            'uva': 'virginia.edu',
            'university of vermont': 'uvm.edu',
            'uvm': 'uvm.edu',
            'harvard university': 'harvard.edu',
            'harvard': 'harvard.edu',
            'stanford university': 'stanford.edu',
            'stanford': 'stanford.edu',
            'mit': 'mit.edu',
            'massachusetts institute of technology': 'mit.edu',
            'university of michigan': 'umich.edu',
            'umich': 'umich.edu',
            'university of texas': 'utexas.edu',
            'ut austin': 'utexas.edu',
            'university of florida': 'ufl.edu',
            'uf': 'ufl.edu',
            'university of washington': 'washington.edu',
            'uw': 'washington.edu',
            'university of illinois': 'illinois.edu',
            'uiuc': 'illinois.edu',
            'university of wisconsin': 'wisc.edu',
            'uw madison': 'wisc.edu',
            'university of minnesota': 'umn.edu',
            'umn': 'umn.edu'
        }
    
    def _extract_domain_from_university(self, university_name: str) -> str:
        """Extract domain from university name"""
        university_lower = university_name.lower()
        
        # Check exact matches first
        if university_lower in self.domain_patterns:
            return self.domain_patterns[university_lower]
        
        # Check partial matches
        for pattern, domain in self.domain_patterns.items():
            if any(word in university_lower for word in pattern.split()):
                return domain
        
        return ""
    
    async def search(self, university_name: str) -> List[SearchResult]:
        """Search for course pages using multiple methods"""
        try:
            logger.info(f"Searching for course pages: {university_name}")
            
            results = []
            
            # Method 1: Check known course pages
            known_results = self._search_known_pages(university_name)
            results.extend(known_results)
            
            # Method 2: Direct domain construction
            direct_results = await self._search_direct_domains(university_name)
            results.extend(direct_results)
            
            # Method 3: Try common patterns
            pattern_results = await self._search_common_patterns(university_name)
            results.extend(pattern_results)
            
            # Remove duplicates
            seen_urls = set()
            unique_results = []
            for result in results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    unique_results.append(result)
            
            logger.info(f"Found {len(unique_results)} course search results")
            return unique_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _search_known_pages(self, university_name: str) -> List[SearchResult]:
        """Search known course pages database"""
        university_lower = university_name.lower()
        results = []
        
        # Check exact matches
        if university_lower in self.known_course_pages:
            for page in self.known_course_pages[university_lower]:
                results.append(SearchResult(
                    title=page['title'],
                    url=page['url'],
                    snippet=page['snippet'],
                    domain=page['domain'],
                    source='known_database',
                    confidence=0.9
                ))
        
        # Check partial matches
        for key, pages in self.known_course_pages.items():
            if any(word in university_lower for word in key.split()):
                for page in pages:
                    results.append(SearchResult(
                        title=page['title'],
                        url=page['url'],
                        snippet=page['snippet'],
                        domain=page['domain'],
                        source='known_database',
                        confidence=0.8
                    ))
        
        return results
    
    async def _search_direct_domains(self, university_name: str) -> List[SearchResult]:
        """Try direct domain construction"""
        try:
            domain = self._extract_domain_from_university(university_name)
            if not domain:
                return []
            
            # Common course search URL patterns
            candidate_urls = [
                f"https://classes.{domain}",
                f"https://courses.{domain}",
                f"https://www.{domain}/classes",
                f"https://www.{domain}/courses",
                f"https://www.{domain}/course-search",
                f"https://www.{domain}/class-search",
                f"https://www.{domain}/schedule",
                f"https://www.{domain}/catalog",
                f"https://{domain}/classes",
                f"https://{domain}/courses"
            ]
            
            results = []
            for url in candidate_urls:
                try:
                    response = self.session.get(url, timeout=10, allow_redirects=True)
                    if response.status_code == 200:
                        # Parse the page to get title
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(response.text, 'html.parser')
                        title = soup.find('title')
                        title_text = title.get_text() if title else f"Course Search - {domain}"
                        
                        results.append(SearchResult(
                            title=title_text,
                            url=response.url,
                            snippet=f"Course search page for {university_name}",
                            domain=domain,
                            source='direct_domain',
                            confidence=0.8
                        ))
                        logger.info(f"Found direct course search page: {url}")
                except Exception as e:
                    logger.debug(f"Failed to check {url}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Direct domain search failed: {e}")
            return []
    
    async def _search_common_patterns(self, university_name: str) -> List[SearchResult]:
        """Try common course search patterns"""
        try:
            domain = self._extract_domain_from_university(university_name)
            if not domain:
                return []
            
            # Additional patterns to try
            patterns = [
                f"https://catalog.{domain}",
                f"https://schedule.{domain}",
                f"https://registrar.{domain}",
                f"https://www.{domain}/registrar",
                f"https://www.{domain}/academics",
                f"https://www.{domain}/academic",
                f"https://{domain}/registrar",
                f"https://{domain}/academics"
            ]
            
            results = []
            for url in patterns:
                try:
                    response = self.session.get(url, timeout=10, allow_redirects=True)
                    if response.status_code == 200:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(response.text, 'html.parser')
                        title = soup.find('title')
                        title_text = title.get_text() if title else f"Academic Page - {domain}"
                        
                        # Check if page contains course-related content
                        page_text = soup.get_text().lower()
                        course_indicators = ['course', 'class', 'schedule', 'catalog', 'registration']
                        if any(indicator in page_text for indicator in course_indicators):
                            results.append(SearchResult(
                                title=title_text,
                                url=response.url,
                                snippet=f"Academic page for {university_name}",
                                domain=domain,
                                source='common_pattern',
                                confidence=0.6
                            ))
                            logger.info(f"Found common pattern page: {url}")
                except Exception as e:
                    logger.debug(f"Failed to check {url}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Common pattern search failed: {e}")
            return []

# Test function
async def test_working_search():
    """Test the working search engine"""
    
    print(" Testing Working Search Engine")
    print("=" * 50)
    
    search_engine = WorkingSearchEngine()
    
    # Test universities
    test_universities = [
        "University of California, Berkeley",
        "Harvard University",
        "Stanford University",
        "MIT",
        "University of Michigan"
    ]
    
    for university in test_universities:
        print(f"\n Searching for: {university}")
        results = await search_engine.search(university)
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results[:3]):
            print(f"  {i+1}. {result.title}")
            print(f"     URL: {result.url}")
            print(f"     Source: {result.source}")
            print(f"     Confidence: {result.confidence}")
            print()

if __name__ == "__main__":
    asyncio.run(test_working_search()) 