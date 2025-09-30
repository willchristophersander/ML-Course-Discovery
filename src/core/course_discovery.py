#!/usr/bin/env python3
"""
Google Course Search Finder v2 - Enhanced with Fallback Strategies

This system combines multiple search strategies with AI models to find course search pages:
1. Google search with fallback strategies
2. AI model selects the best link from search results
3. AI model navigates to the course search page
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import requests
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import re
import time
import random

from utils.path_utils import get_models_path
from utils.training_registry import TrainingRunRegistry

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result"""
    title: str
    url: str
    snippet: str
    domain: str
    source: str  # 'google', 'direct', 'fallback'
    confidence: float = 0.0

@dataclass
class NavigationResult:
    """Result of AI navigation"""
    success: bool
    final_url: str
    confidence: float
    navigation_path: List[str]
    error_message: Optional[str] = None

class EnhancedSearchEngine:
    """Enhanced search engine with multiple strategies"""
    
    def __init__(self):
        self.session = self._create_session()
        self.known_course_search_patterns = [
            'classes.{domain}',
            'courses.{domain}',
            '{domain}/classes',
            '{domain}/courses',
            '{domain}/course-search',
            '{domain}/class-search',
            '{domain}/schedule',
            '{domain}/catalog'
        ]
        
    def _create_session(self):
        """Create a requests session with proper headers"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        return session
    
    def _extract_domain_from_university(self, university_name: str) -> str:
        """Extract domain from university name"""
        # Common university domain patterns
        domain_mappings = {
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
            'winthrop university': 'winthrop.edu',
            'coahoma community college': 'coahomacc.edu',
            'northland college': 'northland.edu',
            'northwestern college of iowa': 'nwciowa.edu',
            'columbus technical college': 'columbustech.edu',
            'fayetteville state university': 'uncfsu.edu',
            'utah state university': 'usu.edu',
            'state university of new york college at new paltz': 'newpaltz.edu',
            'metropolitan community college': 'mccneb.edu',
            'hope college': 'hope.edu',
            'university of charleston': 'charleston.edu',
            'trinity college': 'trincoll.edu',
            'south arkansas community college': 'southark.edu',
        }
        
        university_lower = university_name.lower()
        
        # Try exact match first
        if university_lower in domain_mappings:
            return domain_mappings[university_lower]
        
        # Try partial matches for multi-word universities
        for key, domain in domain_mappings.items():
            if len(key.split()) >= 2:
                # For multi-word keys, check if all words are present
                key_words = set(key.split())
                uni_words = set(university_lower.split())
                if key_words.issubset(uni_words) or uni_words.issubset(key_words):
                    return domain
        
        # Enhanced fallback: try to construct domain from university name
        words = university_name.lower().split()
        
        # Remove common words
        filtered_words = [w for w in words if w not in ['university', 'of', 'the', 'and', 'college', 'institute', 'state', 'technical', 'community']]
        
        if filtered_words:
            # Try different domain construction strategies
            candidates = []
            
            # Strategy 1: First word + .edu
            candidates.append(f"{filtered_words[0]}.edu")
            
            # Strategy 2: First two words combined + .edu
            if len(filtered_words) >= 2:
                candidates.append(f"{filtered_words[0]}{filtered_words[1]}.edu")
            
            # Strategy 3: Abbreviated form
            if len(filtered_words) >= 2:
                candidates.append(f"{filtered_words[0][0]}{filtered_words[1]}.edu")
            
            # Strategy 4: Common abbreviations
            if 'university' in words:
                if 'new' in words and 'york' in words:
                    candidates.append("nyu.edu")
                elif 'new' in words and 'hampshire' in words:
                    candidates.append("unh.edu")
                elif 'california' in words:
                    candidates.append("uc.edu")
            
            # Return the first candidate (we'll test all of them)
            return candidates[0] if candidates else None
        
        return None
    
    async def search_with_fallbacks(self, university_name: str) -> List[SearchResult]:
        """
        Search for course search pages using multiple strategies
        
        Args:
            university_name: Name of the university
            
        Returns:
            List of SearchResult objects
        """
        results = []
        
        # Strategy 1: Try Google search
        google_results = await self._search_google(university_name)
        results.extend(google_results)
        
        # Strategy 2: Try direct domain construction
        direct_results = await self._search_direct_domains(university_name)
        results.extend(direct_results)
        
        # Strategy 3: Try known patterns
        pattern_results = await self._search_known_patterns(university_name)
        results.extend(pattern_results)
        
        # Remove duplicates while preserving order
        seen_urls = set()
        unique_results = []
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results
    
    async def _search_google(self, university_name: str) -> List[SearchResult]:
        """Try web search with multiple engines and direct domain construction"""
        try:
            logger.info(f"Trying enhanced search for: {university_name}")
            
            results = []
            
            # Strategy 1: Web search using HTML scraping (concurrent)
            try:
                from .search_engines import SimpleWebSearch
                web_searcher = SimpleWebSearch()
                
                # Try different search queries concurrently
                search_queries = [
                    f'"{university_name}" course search',
                    f'"{university_name}" class schedule',
                    f'"{university_name}" course catalog',
                    f'"{university_name}" course registration',
                    f'"{university_name}" course finder'
                ]
                
                async def search_query(query: str) -> List[SearchResult]:
                    try:
                        logger.info(f"Searching web for: {query}")
                        search_results = await web_searcher.search(query, max_results=5)
                        logger.info(f"Found {len(search_results)} web search results")
                        return search_results
                    except Exception as e:
                        logger.debug(f"Web search failed for query '{query}': {e}")
                        return []
                
                # Search all queries concurrently
                import asyncio
                tasks = [search_query(query) for query in search_queries]
                query_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Combine all results
                for query_result in query_results:
                    if isinstance(query_result, list):
                        results.extend(query_result)
                        if len(results) >= 15:  # Limit web search results
                            break
                        
            except Exception as e:
                logger.debug(f"Web search failed: {e}")
            
            # Strategy 2: Direct domain construction (concurrent)
            domain = self._extract_domain_from_university(university_name)
            if domain:
                # Try common course search patterns concurrently
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
                
                # Test URLs concurrently
                async def test_url(url: str) -> Optional[SearchResult]:
                    try:
                        response = self.session.get(url, timeout=10, allow_redirects=True)
                        if response.status_code == 200:
                            # Parse the page to get title
                            soup = BeautifulSoup(response.text, 'html.parser')
                            title = soup.find('title')
                            title_text = title.get_text() if title else f"Course Search - {domain}"
                            
                            result = SearchResult(
                                title=title_text,
                                url=response.url,
                                snippet=f"Course search page for {university_name}",
                                domain=domain,
                                source='direct'
                            )
                            logger.info(f"Found direct course search page: {url}")
                            return result
                    except Exception as e:
                        logger.debug(f"Failed to check {url}: {e}")
                    return None
                
                # Test all URLs concurrently
                import asyncio
                tasks = [test_url(url) for url in candidate_urls]
                url_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Add successful results
                for result in url_results:
                    if isinstance(result, SearchResult):
                        results.append(result)
            
            # Strategy 3: Add known course search pages for major universities
            known_course_pages = self._get_known_course_pages(university_name)
            for page in known_course_pages:
                results.append(SearchResult(
                    title=page['title'],
                    url=page['url'],
                    snippet=page['snippet'],
                    domain=page['domain'],
                    source='known'
                ))
            
            # Remove duplicates while preserving order
            seen_urls = set()
            unique_results = []
            for result in results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    unique_results.append(result)
            
            logger.info(f"Found {len(unique_results)} search results using enhanced method")
            return unique_results
            
        except Exception as e:
            logger.error(f"Error in enhanced search: {e}")
            return []
    
    def _get_known_course_pages(self, university_name: str) -> List[Dict[str, str]]:
        """Get known course search pages for major universities"""
        university_lower = university_name.lower()
        
        known_pages = {
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
            ]
        }
        
        # Check for exact matches
        if university_lower in known_pages:
            return known_pages[university_lower]
        
        # Check for partial matches (more specific)
        for key, pages in known_pages.items():
            key_words = key.split()
            # For multi-word keys, require consecutive word matches or exact phrase
            if len(key_words) >= 2:
                # Check for exact phrase match first
                if key in university_lower:
                    return pages
                # Check for consecutive word matches (e.g., "university of" in both)
                key_phrase = ' '.join(key_words)
                if key_phrase in university_lower:
                    return pages
            # For single word keys, require exact match
            elif len(key_words) == 1 and key_words[0] in university_lower:
                return pages
        
        return []
    
    async def _search_direct_domains(self, university_name: str) -> List[SearchResult]:
        """Try direct domain construction"""
        try:
            logger.info(f"Trying direct domain search for: {university_name}")
            
            domain = self._extract_domain_from_university(university_name)
            if not domain:
                return []
            
            # Try common course search URLs
            candidate_urls = [
                f"https://classes.{domain}",
                f"https://courses.{domain}",
                f"https://www.{domain}/classes",
                f"https://www.{domain}/courses",
                f"https://www.{domain}/course-search",
                f"https://www.{domain}/class-search",
                f"https://www.{domain}/schedule",
                f"https://www.{domain}/catalog"
            ]
            
            results = []
            for url in candidate_urls:
                try:
                    # Check if URL exists
                    response = self.session.get(url, timeout=10, allow_redirects=True)
                    if response.status_code == 200:
                        # Extract title from page
                        soup = BeautifulSoup(response.text, 'html.parser')
                        title = soup.find('title')
                        title_text = title.get_text() if title else f"Course Search - {domain}"
                        
                        results.append(SearchResult(
                            title=title_text,
                            url=url,
                            snippet=f"Direct course search page for {university_name}",
                            domain=domain,
                            source='direct'
                        ))
                        logger.info(f"Found direct course search page: {url}")
                        
                except Exception as e:
                    logger.debug(f"Failed to access {url}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in direct domain search: {e}")
            return []
    
    async def _search_known_patterns(self, university_name: str) -> List[SearchResult]:
        """Try known course search patterns"""
        try:
            logger.info(f"Trying known patterns for: {university_name}")
            
            domain = self._extract_domain_from_university(university_name)
            if not domain:
                return []
            
            # Known course search patterns for common universities
            known_patterns = {
                'berkeley.edu': [
                    'https://classes.berkeley.edu/',
                    'https://catalog.berkeley.edu/'
                ],
                'unh.edu': [
                    'https://courses.unh.edu/',
                    'https://www.unh.edu/registrar/class-schedule'
                ],
                'osu.edu': [
                    'https://classes.osu.edu/',
                    'https://catalog.osu.edu/'
                ],
                'virginia.edu': [
                    'https://classes.virginia.edu/',
                    'https://catalog.virginia.edu/'
                ],
                'uvm.edu': [
                    'https://catalogue.uvm.edu/',
                    'https://www.uvm.edu/registrar/class-schedule'
                ]
            }
            
            if domain in known_patterns:
                results = []
                for url in known_patterns[domain]:
                    try:
                        response = self.session.get(url, timeout=10)
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, 'html.parser')
                            title = soup.find('title')
                            title_text = title.get_text() if title else f"Course Search - {domain}"
                            
                            results.append(SearchResult(
                                title=title_text,
                                url=url,
                                snippet=f"Known course search page for {university_name}",
                                domain=domain,
                                source='known_pattern'
                            ))
                            logger.info(f"Found known pattern page: {url}")
                            
                    except Exception as e:
                        logger.debug(f"Failed to access {url}: {e}")
                        continue
                
                return results
            
            return []
            
        except Exception as e:
            logger.error(f"Error in known patterns search: {e}")
            return []

class SearchResultClassifier:
    """AI model to classify and rank search results"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_file = get_models_path('search_result_classifier_v2.pkl')
        self._load_or_train_model()
    
    def _extract_features(self, search_result: SearchResult) -> Dict[str, Any]:
        """Extract features from a search result"""
        features = {}
        
        # Text features
        text = f"{search_result.title} {search_result.snippet}".lower()
        
        # Course search keywords
        course_keywords = ['course', 'class', 'schedule', 'catalog', 'search', 'register', 'enroll']
        features['course_keyword_count'] = sum(1 for keyword in course_keywords if keyword in text)
        
        # University keywords
        university_keywords = ['university', 'college', 'school', 'institution', 'academic']
        features['university_keyword_count'] = sum(1 for keyword in university_keywords if keyword in text)
        
        # URL features
        url_lower = search_result.url.lower()
        features['has_course_in_url'] = 'course' in url_lower
        features['has_class_in_url'] = 'class' in url_lower
        features['has_schedule_in_url'] = 'schedule' in url_lower
        features['has_catalog_in_url'] = 'catalog' in url_lower
        features['has_search_in_url'] = 'search' in url_lower
        
        # Domain features
        domain_lower = search_result.domain.lower()
        features['is_edu_domain'] = domain_lower.endswith('.edu')
        features['is_official_domain'] = any(keyword in domain_lower for keyword in ['university', 'college', 'edu'])
        
        # Title features
        title_lower = search_result.title.lower()
        features['title_has_course'] = 'course' in title_lower
        features['title_has_class'] = 'class' in title_lower
        features['title_has_schedule'] = 'schedule' in title_lower
        features['title_has_catalog'] = 'catalog' in title_lower
        
        # Snippet features
        snippet_lower = search_result.snippet.lower()
        features['snippet_has_course'] = 'course' in snippet_lower
        features['snippet_has_search'] = 'search' in snippet_lower
        features['snippet_has_enroll'] = 'enroll' in snippet_lower
        
        # Source features
        features['is_direct_source'] = search_result.source == 'direct'
        features['is_known_pattern'] = search_result.source == 'known_pattern'
        features['is_google_source'] = search_result.source == 'google'
        
        return features
    
    def _load_or_train_model(self):
        """Load existing model or train a new one"""
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']
                logger.info(" Search result classifier loaded")
            else:
                self._train_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._train_model()
    
    def _train_model(self):
        """Train the search result classifier"""
        logger.info("Training search result classifier...")
        
        # Create training data
        training_data = self._create_training_data()
        
        # Extract features
        X = []
        y = []
        
        for item in training_data:
            features = self._extract_features(item['result'])
            X.append(list(features.values()))
            y.append(item['label'])
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # Save model
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        
        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(" Search result classifier trained and saved")
        TrainingRunRegistry().log_run(
            model_name="search_result_classifier_v2",
            metadata={"examples": int(len(training_data))}
        )
    
    def _create_training_data(self) -> List[Dict[str, Any]]:
        """Create training data for the classifier"""
        training_data = []
        
        # Positive examples (good course search results)
        positive_examples = [
            {
                'result': SearchResult(
                    title="UC Berkeley Class Schedule",
                    url="https://classes.berkeley.edu/",
                    snippet="Search for classes at UC Berkeley. Find course schedules, registration information, and academic calendars.",
                    domain="classes.berkeley.edu",
                    source="direct"
                ),
                'label': 1
            },
            {
                'result': SearchResult(
                    title="UNH Course Search",
                    url="https://courses.unh.edu/",
                    snippet="Search for courses at the University of New Hampshire. Browse course catalogs and registration information.",
                    domain="courses.unh.edu",
                    source="direct"
                ),
                'label': 1
            },
            {
                'result': SearchResult(
                    title="OSU Class Search",
                    url="https://classes.osu.edu/",
                    snippet="Search for classes at Ohio State University. Find course schedules and registration details.",
                    domain="classes.osu.edu",
                    source="direct"
                ),
                'label': 1
            },
            {
                'result': SearchResult(
                    title="UVM Course Catalog",
                    url="https://catalogue.uvm.edu/",
                    snippet="Browse the complete course catalog for University of Vermont.",
                    domain="catalogue.uvm.edu",
                    source="known_pattern"
                ),
                'label': 1
            }
        ]
        
        # Negative examples (bad results)
        negative_examples = [
            {
                'result': SearchResult(
                    title="University General Information",
                    url="https://www.university.edu/about/",
                    snippet="Learn about our university's history, mission, and general information.",
                    domain="university.edu",
                    source="google"
                ),
                'label': 0
            },
            {
                'result': SearchResult(
                    title="Student Life",
                    url="https://www.university.edu/student-life/",
                    snippet="Information about student activities, housing, and campus life.",
                    domain="university.edu",
                    source="google"
                ),
                'label': 0
            },
            {
                'result': SearchResult(
                    title="Admissions",
                    url="https://www.university.edu/admissions/",
                    snippet="Apply to our university. Learn about admission requirements and deadlines.",
                    domain="university.edu",
                    source="google"
                ),
                'label': 0
            }
        ]
        
        training_data.extend(positive_examples)
        training_data.extend(negative_examples)
        
        return training_data
    
    def predict_best_result(self, search_results: List[SearchResult]) -> Tuple[SearchResult, float]:
        """
        Predict the best search result for course search
        
        Args:
            search_results: List of search results
            
        Returns:
            Tuple of (best_result, confidence)
        """
        if not search_results:
            return None, 0.0
        
        # Extract features for all results
        features_list = []
        for result in search_results:
            features = self._extract_features(result)
            features_list.append(list(features.values()))
        
        # Predict probabilities
        X_scaled = self.scaler.transform(features_list)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Find the result with highest probability of being a course search page
        best_idx = np.argmax([prob[1] for prob in probabilities])
        best_result = search_results[best_idx]
        confidence = probabilities[best_idx][1]
        
        return best_result, confidence

class PageNavigationClassifier:
    """AI model to navigate from a page to course search"""
    
    def __init__(self):
        self.model = None
        self.model_file = get_models_path('page_navigation_classifier_v2.pkl')
        self._load_or_train_model()
    
    def _extract_link_features(self, link_element: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from a link element"""
        features = {}
        
        # Text features
        text = link_element.get('text', '').lower()
        href = link_element.get('href', '').lower()
        
        # Course search keywords
        course_keywords = ['course', 'class', 'schedule', 'catalog', 'search', 'register', 'enroll']
        features['text_course_keywords'] = sum(1 for keyword in course_keywords if keyword in text)
        features['href_course_keywords'] = sum(1 for keyword in course_keywords if keyword in href)
        
        # Navigation keywords
        nav_keywords = ['search', 'find', 'browse', 'lookup', 'directory']
        features['text_nav_keywords'] = sum(1 for keyword in nav_keywords if keyword in text)
        features['href_nav_keywords'] = sum(1 for keyword in nav_keywords if keyword in href)
        
        # URL features
        features['has_course_in_href'] = 'course' in href
        features['has_class_in_href'] = 'class' in href
        features['has_schedule_in_href'] = 'schedule' in href
        features['has_catalog_in_href'] = 'catalog' in href
        features['has_search_in_href'] = 'search' in href
        
        # Text features
        features['text_has_course'] = 'course' in text
        features['text_has_class'] = 'class' in text
        features['text_has_schedule'] = 'schedule' in text
        features['text_has_catalog'] = 'catalog' in text
        features['text_has_search'] = 'search' in text
        
        # Length features
        features['text_length'] = len(text)
        features['href_length'] = len(href)
        
        return features
    
    def _load_or_train_model(self):
        """Load existing model or train a new one"""
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(" Page navigation classifier loaded")
            else:
                self._train_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._train_model()
    
    def _train_model(self):
        """Train the page navigation classifier"""
        logger.info("Training page navigation classifier...")
        
        # Create training data
        training_data = self._create_training_data()
        
        # Extract features
        X = []
        y = []
        
        for item in training_data:
            features = self._extract_link_features(item['link'])
            X.append(list(features.values()))
            y.append(item['label'])
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Save model
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(" Page navigation classifier trained and saved")
        TrainingRunRegistry().log_run(
            model_name="page_navigation_classifier_v2",
            metadata={"examples": int(len(training_data))}
        )
    
    def _create_training_data(self) -> List[Dict[str, Any]]:
        """Create training data for the navigation classifier"""
        training_data = []
        
        # Positive examples (good navigation links)
        positive_examples = [
            {
                'link': {
                    'text': 'Course Search',
                    'href': '/course-search'
                },
                'label': 1
            },
            {
                'link': {
                    'text': 'Class Schedule',
                    'href': '/class-schedule'
                },
                'label': 1
            },
            {
                'link': {
                    'text': 'Course Catalog',
                    'href': '/catalog'
                },
                'label': 1
            },
            {
                'link': {
                    'text': 'Search Classes',
                    'href': '/search-classes'
                },
                'label': 1
            },
            {
                'link': {
                    'text': 'Browse Courses',
                    'href': '/browse-courses'
                },
                'label': 1
            }
        ]
        
        # Negative examples (bad navigation links)
        negative_examples = [
            {
                'link': {
                    'text': 'About Us',
                    'href': '/about'
                },
                'label': 0
            },
            {
                'link': {
                    'text': 'Contact',
                    'href': '/contact'
                },
                'label': 0
            },
            {
                'link': {
                    'text': 'Admissions',
                    'href': '/admissions'
                },
                'label': 0
            },
            {
                'link': {
                    'text': 'Student Life',
                    'href': '/student-life'
                },
                'label': 0
            },
            {
                'link': {
                    'text': 'Library',
                    'href': '/library'
                },
                'label': 0
            }
        ]
        
        training_data.extend(positive_examples)
        training_data.extend(negative_examples)
        
        return training_data
    
    def predict_best_link(self, links: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
        """
        Predict the best link to click for course search navigation
        
        Args:
            links: List of link dictionaries with 'text' and 'href' keys
            
        Returns:
            Tuple of (best_link, confidence)
        """
        if not links:
            return None, 0.0
        
        # Extract features for all links
        features_list = []
        for link in links:
            features = self._extract_link_features(link)
            features_list.append(list(features.values()))
        
        # Predict probabilities
        probabilities = self.model.predict_proba(features_list)
        
        # Find the link with highest probability of leading to course search
        best_idx = np.argmax([prob[1] for prob in probabilities])
        best_link = links[best_idx]
        confidence = probabilities[best_idx][1]
        
        return best_link, confidence

class EnhancedCourseSearchFinder:
    """Enhanced course search finder with multiple strategies"""
    
    def __init__(self):
        self.search_engine = EnhancedSearchEngine()
        self.search_classifier = SearchResultClassifier()
        self.navigation_classifier = PageNavigationClassifier()
        
    async def find_course_search_page(self, university_name: str) -> NavigationResult:
        """
        Find course search page for a university using multiple strategies
        
        Args:
            university_name: Name of the university
            
        Returns:
            NavigationResult with the final URL and confidence
        """
        try:
            logger.info(f" Finding course search page for: {university_name}")
            
            # Step 1: Search with multiple strategies
            search_results = await self.search_engine.search_with_fallbacks(university_name)
            
            if not search_results:
                return NavigationResult(
                    success=False,
                    final_url="",
                    confidence=0.0,
                    navigation_path=[],
                    error_message="No search results found with any strategy"
                )
            
            logger.info(f"Found {len(search_results)} search results from {len(set(r.source for r in search_results))} sources")
            
            # Step 2: AI selects best search result
            best_result, search_confidence = self.search_classifier.predict_best_result(search_results)
            
            logger.info(f" Best search result: {best_result.title} ({search_confidence:.2f}) from {best_result.source}")
            
            # Step 3: Navigate to the selected page
            navigation_result = await self._navigate_to_course_search(best_result.url, university_name)
            
            # Combine confidences
            overall_confidence = (search_confidence + navigation_result.confidence) / 2
            
            return NavigationResult(
                success=navigation_result.success,
                final_url=navigation_result.final_url,
                confidence=overall_confidence,
                navigation_path=[best_result.url] + navigation_result.navigation_path,
                error_message=navigation_result.error_message
            )
            
        except Exception as e:
            logger.error(f"Error in find_course_search_page: {e}")
            return NavigationResult(
                success=False,
                final_url="",
                confidence=0.0,
                navigation_path=[],
                error_message=str(e)
            )
    
    async def _navigate_to_course_search(self, start_url: str, university_name: str) -> NavigationResult:
        """
        Navigate from a page to the course search functionality
        
        Args:
            start_url: URL to start navigation from
            university_name: Name of the university
            
        Returns:
            NavigationResult with navigation results
        """
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                try:
                    # Navigate to the starting page
                    await page.goto(start_url)
                    await page.wait_for_load_state('networkidle')
                    
                    # Check if we're already on a course search page
                    current_url = page.url
                    page_content = await page.content()
                    
                    if self._is_course_search_page(page_content, current_url):
                        logger.info(f" Already on course search page: {current_url}")
                        await browser.close()
                        return NavigationResult(
                            success=True,
                            final_url=current_url,
                            confidence=0.9,
                            navigation_path=[]
                        )
                    
                    navigation_path = [start_url]
                    max_navigation_steps = 3
                    
                    for step in range(max_navigation_steps):
                        # Extract all links from the page
                        links = await page.evaluate("""
                            () => {
                                const links = [];
                                const linkElements = document.querySelectorAll('a[href]');
                                
                                linkElements.forEach(link => {
                                    if (link.href && link.href.startsWith('http')) {
                                        links.push({
                                            text: link.textContent.trim(),
                                            href: link.href
                                        });
                                    }
                                });
                                
                                return links;
                            }
                        """)
                        
                        if not links:
                            break
                        
                        # AI selects the best link
                        best_link, confidence = self.navigation_classifier.predict_best_link(links)
                        
                        if best_link and confidence > 0.5:
                            logger.info(f" Step {step + 1}: Clicking '{best_link['text']}' (confidence: {confidence:.2f})")
                            
                            # Click the link
                            await page.click(f"a[href='{best_link['href']}']")
                            await page.wait_for_load_state('networkidle')
                            
                            navigation_path.append(best_link['href'])
                            
                            # Check if we've reached a course search page
                            current_url = page.url
                            page_content = await page.content()
                            
                            if self._is_course_search_page(page_content, current_url):
                                logger.info(f" Found course search page: {current_url}")
                                await browser.close()
                                return NavigationResult(
                                    success=True,
                                    final_url=current_url,
                                    confidence=confidence,
                                    navigation_path=navigation_path
                                )
                        else:
                            logger.info(f" No good navigation link found (best confidence: {confidence:.2f})")
                            break
                    
                    # If we get here, we didn't find a course search page
                    await browser.close()
                    return NavigationResult(
                        success=False,
                        final_url=page.url,
                        confidence=0.0,
                        navigation_path=navigation_path,
                        error_message="Could not find course search page after navigation"
                    )
                    
                except Exception as e:
                    await browser.close()
                    return NavigationResult(
                        success=False,
                        final_url="",
                        confidence=0.0,
                        navigation_path=[],
                        error_message=str(e)
                    )
                    
        except Exception as e:
            return NavigationResult(
                success=False,
                final_url="",
                confidence=0.0,
                navigation_path=[],
                error_message=str(e)
            )
    
    def _is_course_search_page(self, page_content: str, url: str) -> bool:
        """
        Check if a page is a course search page
        
        Args:
            page_content: HTML content of the page
            url: URL of the page
            
        Returns:
            True if it's a course search page
        """
        # Convert to lowercase for easier matching
        content_lower = page_content.lower()
        url_lower = url.lower()
        
        # Check for course search indicators
        course_indicators = [
            'course search',
            'class search',
            'course catalog',
            'class schedule',
            'course schedule',
            'search courses',
            'find classes',
            'course finder',
            'browse courses',
            'course listings'
        ]
        
        for indicator in course_indicators:
            if indicator in content_lower:
                return True
        
        # Check URL patterns
        url_patterns = [
            '/course-search',
            '/class-search',
            '/course-catalog',
            '/class-schedule',
            '/courses',
            '/classes',
            '/schedule',
            '/catalog'
        ]
        
        for pattern in url_patterns:
            if pattern in url_lower:
                return True
        
        return False

async def main():
    """Main function to test the enhanced system"""
    finder = EnhancedCourseSearchFinder()
    
    # Test universities
    universities = [
        "University of California, Berkeley",
        "University of New Hampshire",
        "Ohio State University",
        "University of Virginia",
        "University of Vermont"
    ]
    
    print(" Enhanced Google Course Search Finder with AI Navigation")
    print("=" * 70)
    
    for university in universities:
        print(f"\n Testing: {university}")
        print("-" * 50)
        
        result = await finder.find_course_search_page(university)
        
        if result.success:
            print(f" Success! Found course search page:")
            print(f"   URL: {result.final_url}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Navigation Path: {' -> '.join(result.navigation_path)}")
        else:
            print(f" Failed to find course search page:")
            print(f"   Error: {result.error_message}")
            print(f"   Final URL: {result.final_url}")
        
        print()

if __name__ == "__main__":
    asyncio.run(main()) 
