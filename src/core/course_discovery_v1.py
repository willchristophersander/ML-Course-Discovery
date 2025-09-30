#!/usr/bin/env python3
"""
Google Course Search Finder with AI Navigation

This system combines Google search with AI models to find course search pages:
1. Searches Google for "college name" + "course search"
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pickle
import re

from utils.path_utils import get_models_path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result from Google"""
    title: str
    url: str
    snippet: str
    domain: str
    confidence: float = 0.0

@dataclass
class NavigationResult:
    """Result of AI navigation"""
    success: bool
    final_url: str
    confidence: float
    navigation_path: List[str]
    error_message: Optional[str] = None

class GoogleSearchEngine:
    """Handles Google search functionality"""
    
    def __init__(self):
        self.session = self._create_session()
        
    def _create_session(self):
        """Create a requests session with proper headers"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        return session
    
    async def search_google(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Search Google for course search pages
        
        Args:
            query: Search query (e.g., "University of California Berkeley course search")
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            logger.info(f"Searching Google for: {query}")
            
            # For demonstration, we'll use a simple approach
            # In production, you'd want to use Google's API or a service like SerpAPI
            
            # Create search URL
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                try:
                    await page.goto(search_url)
                    await page.wait_for_load_state('networkidle')
                    
                    # Extract search results
                    results = await page.evaluate("""
                        () => {
                            const results = [];
                            const searchResults = document.querySelectorAll('.g');
                            
                            searchResults.forEach((result, index) => {
                                if (index >= 10) return; // Limit to 10 results
                                
                                const titleElement = result.querySelector('h3');
                                const linkElement = result.querySelector('a');
                                const snippetElement = result.querySelector('.VwiC3b');
                                
                                if (titleElement && linkElement) {
                                    results.push({
                                        title: titleElement.textContent || '',
                                        url: linkElement.href || '',
                                        snippet: snippetElement ? snippetElement.textContent || '' : '',
                                        domain: new URL(linkElement.href || '').hostname
                                    });
                                }
                            });
                            
                            return results;
                        }
                    """)
                    
                    await browser.close()
                    
                    # Convert to SearchResult objects
                    search_results = []
                    for result in results:
                        search_results.append(SearchResult(
                            title=result['title'],
                            url=result['url'],
                            snippet=result['snippet'],
                            domain=result['domain']
                        ))
                    
                    logger.info(f"Found {len(search_results)} search results")
                    return search_results
                    
                except Exception as e:
                    logger.error(f"Error during Google search: {e}")
                    await browser.close()
                    return []
                    
        except Exception as e:
            logger.error(f"Error in search_google: {e}")
            return []

class SearchResultClassifier:
    """AI model to classify and rank search results"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.model_file = get_models_path('search_result_classifier.pkl')
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
        
        return features
    
    def _load_or_train_model(self):
        """Load existing model or train a new one"""
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.vectorizer = model_data['vectorizer']
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
            'vectorizer': self.vectorizer,
            'scaler': self.scaler
        }
        
        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(" Search result classifier trained and saved")
    
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
                    domain="classes.berkeley.edu"
                ),
                'label': 1
            },
            {
                'result': SearchResult(
                    title="UNH Course Search",
                    url="https://courses.unh.edu/",
                    snippet="Search for courses at the University of New Hampshire. Browse course catalogs and registration information.",
                    domain="courses.unh.edu"
                ),
                'label': 1
            },
            {
                'result': SearchResult(
                    title="OSU Class Search",
                    url="https://classes.osu.edu/",
                    snippet="Search for classes at Ohio State University. Find course schedules and registration details.",
                    domain="classes.osu.edu"
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
                    domain="university.edu"
                ),
                'label': 0
            },
            {
                'result': SearchResult(
                    title="Student Life",
                    url="https://www.university.edu/student-life/",
                    snippet="Information about student activities, housing, and campus life.",
                    domain="university.edu"
                ),
                'label': 0
            },
            {
                'result': SearchResult(
                    title="Admissions",
                    url="https://www.university.edu/admissions/",
                    snippet="Apply to our university. Learn about admission requirements and deadlines.",
                    domain="university.edu"
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
        self.model_file = get_models_path('page_navigation_classifier.pkl')
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

class GoogleCourseSearchFinder:
    """Main class that combines Google search with AI navigation"""
    
    def __init__(self):
        self.search_engine = GoogleSearchEngine()
        self.search_classifier = SearchResultClassifier()
        self.navigation_classifier = PageNavigationClassifier()
        
    async def find_course_search_page(self, university_name: str) -> NavigationResult:
        """
        Find course search page for a university
        
        Args:
            university_name: Name of the university
            
        Returns:
            NavigationResult with the final URL and confidence
        """
        try:
            logger.info(f" Finding course search page for: {university_name}")
            
            # Step 1: Search Google
            query = f'"{university_name}" "course search"'
            search_results = await self.search_engine.search_google(query)
            
            if not search_results:
                return NavigationResult(
                    success=False,
                    final_url="",
                    confidence=0.0,
                    navigation_path=[],
                    error_message="No search results found"
                )
            
            # Step 2: AI selects best search result
            best_result, search_confidence = self.search_classifier.predict_best_result(search_results)
            
            logger.info(f" Best search result: {best_result.title} ({search_confidence:.2f})")
            
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
            'course finder'
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
            '/classes'
        ]
        
        for pattern in url_patterns:
            if pattern in url_lower:
                return True
        
        return False

async def main():
    """Main function to test the system"""
    finder = GoogleCourseSearchFinder()
    
    # Test universities
    universities = [
        "University of California, Berkeley",
        "University of New Hampshire",
        "Ohio State University"
    ]
    
    print(" Google Course Search Finder with AI Navigation")
    print("=" * 60)
    
    for university in universities:
        print(f"\n Testing: {university}")
        print("-" * 40)
        
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
