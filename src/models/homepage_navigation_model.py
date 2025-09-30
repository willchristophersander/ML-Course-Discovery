#!/usr/bin/env python3
"""
Homepage Navigation Model

This model learns to navigate from college homepages to course search pages
by analyzing link patterns and text content.
"""

import re
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

from utils.path_utils import get_models_path
from utils.training_registry import TrainingRunRegistry

logger = logging.getLogger(__name__)

@dataclass
class NavigationLink:
    """Represents a navigation link found on a homepage"""
    text: str
    url: str
    confidence: float
    link_type: str  # 'course_search', 'academics', 'students', 'general', etc.

@dataclass
class NavigationResult:
    """Result of homepage navigation attempt"""
    success: bool
    final_url: str
    confidence: float
    navigation_path: List[NavigationLink]
    error_message: Optional[str] = None

class HomepageNavigationModel:
    """Model for navigating from college homepages to course search pages"""
    
    def __init__(self, model_file: str = "homepage_navigation_model.pkl"):
        self.model_file = get_models_path(model_file)
        self.link_classifier = None
        self.vectorizer = None
        self.training_data = []
        self.is_trained = False
        
    def extract_links_from_html(self, html_content: str, base_url: str) -> List[NavigationLink]:
        """Extract and classify links from HTML content"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            text = link.get_text(strip=True)
            
            if href and text:
                # Resolve relative URLs
                full_url = urljoin(base_url, href)
                
                # Skip external links and non-HTTP links
                if not full_url.startswith('http'):
                    continue
                
                parsed_base = urlparse(base_url)
                parsed_link = urlparse(full_url)
                
                if parsed_base.netloc != parsed_link.netloc:
                    continue
                
                # Calculate initial confidence based on text content
                confidence = self._calculate_link_confidence(text, full_url)
                
                # Determine link type
                link_type = self._classify_link_type(text, full_url)
                
                links.append(NavigationLink(
                    text=text,
                    url=full_url,
                    confidence=confidence,
                    link_type=link_type
                ))
        
        return links
    
    def _calculate_link_confidence(self, text: str, url: str) -> float:
        """Calculate confidence score for a link based on text and URL"""
        text_lower = text.lower()
        url_lower = url.lower()
        
        confidence = 0.0
        
        # Course-related keywords (high confidence)
        course_keywords = [
            'course', 'class', 'schedule', 'catalog', 'registration',
            'enrollment', 'search', 'finder', 'directory'
        ]
        
        for keyword in course_keywords:
            if keyword in text_lower:
                confidence += 0.2
            if keyword in url_lower:
                confidence += 0.1
        
        # Academic-related keywords (medium confidence)
        academic_keywords = [
            'academics', 'academic', 'education', 'learning',
            'students', 'student', 'registrar'
        ]
        
        for keyword in academic_keywords:
            if keyword in text_lower:
                confidence += 0.15
            if keyword in url_lower:
                confidence += 0.05
        
        # General navigation keywords (low confidence)
        general_keywords = [
            'about', 'admissions', 'research', 'campus', 'athletics',
            'news', 'events', 'alumni', 'giving'
        ]
        
        for keyword in general_keywords:
            if keyword in text_lower:
                confidence -= 0.05
        
        # URL patterns
        if 'course' in url_lower or 'class' in url_lower:
            confidence += 0.3
        if 'academic' in url_lower or 'academics' in url_lower:
            confidence += 0.2
        if 'registrar' in url_lower:
            confidence += 0.25
        
        return max(0.0, min(1.0, confidence))
    
    def _classify_link_type(self, text: str, url: str) -> str:
        """Classify the type of navigation link"""
        text_lower = text.lower()
        url_lower = url.lower()
        
        # Course search indicators
        course_indicators = ['course', 'class', 'schedule', 'catalog', 'registration']
        if any(indicator in text_lower or indicator in url_lower for indicator in course_indicators):
            return 'course_search'
        
        # Academic indicators
        academic_indicators = ['academic', 'academics', 'education', 'learning']
        if any(indicator in text_lower or indicator in url_lower for indicator in academic_indicators):
            return 'academics'
        
        # Student indicators
        student_indicators = ['student', 'students', 'registrar']
        if any(indicator in text_lower or indicator in url_lower for indicator in student_indicators):
            return 'students'
        
        return 'general'
    
    def rank_links(self, links: List[NavigationLink]) -> List[NavigationLink]:
        """Rank links by likelihood of leading to course search page"""
        # Sort by confidence (highest first)
        ranked_links = sorted(links, key=lambda x: x.confidence, reverse=True)
        
        # Boost confidence for certain link types
        for link in ranked_links:
            if link.link_type == 'course_search':
                link.confidence = min(1.0, link.confidence + 0.3)
            elif link.link_type == 'academics':
                link.confidence = min(1.0, link.confidence + 0.2)
            elif link.link_type == 'students':
                link.confidence = min(1.0, link.confidence + 0.1)
        
        # Re-sort after boosting
        ranked_links = sorted(ranked_links, key=lambda x: x.confidence, reverse=True)
        
        return ranked_links
    
    def train_model(self, training_data: List[Dict]):
        """Train the navigation model with examples"""
        if not training_data:
            logger.warning("No training data provided")
            return
        
        # Extract features from training data
        X = []
        y = []
        
        for example in training_data:
            features = self._extract_link_features(example)
            X.append(features)
            y.append(1 if example.get('success', False) else 0)
        
        if len(X) < 2:
            logger.warning("Insufficient training data")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train classifier
        self.link_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.link_classifier.fit(X_train, y_train)
        
        # Train vectorizer for text features
        texts = [example.get('link_text', '') for example in training_data]
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.vectorizer.fit(texts)
        
        self.is_trained = True
        logger.info(f"Model trained with {len(training_data)} examples")
        TrainingRunRegistry().log_run(
            model_name="homepage_navigation_model",
            metadata={"examples": int(len(training_data))}
        )
    
    def _extract_link_features(self, link: Dict) -> List[float]:
        """Extract features from a link for the classifier"""
        features = []
        
        # Text length
        features.append(len(link.get('link_text', '')))
        
        # URL length
        features.append(len(link.get('url', '')))
        
        # Keyword presence
        keywords = ['course', 'class', 'schedule', 'catalog', 'registration', 'academic', 'student']
        text_lower = link.get('link_text', '').lower()
        url_lower = link.get('url', '').lower()
        
        for keyword in keywords:
            features.append(1.0 if keyword in text_lower else 0.0)
            features.append(1.0 if keyword in url_lower else 0.0)
        
        # Link type encoding
        link_types = ['course_search', 'academics', 'students', 'general']
        link_type = link.get('link_type', 'general')
        for lt in link_types:
            features.append(1.0 if link_type == lt else 0.0)
        
        return features
    
    def predict_link_success(self, link: NavigationLink) -> float:
        """Predict the likelihood of a link leading to a course search page"""
        if not self.is_trained:
            return link.confidence
        
        # Extract features
        link_dict = {
            'link_text': link.text,
            'url': link.url,
            'link_type': link.link_type
        }
        
        features = self._extract_link_features(link_dict)
        
        # Make prediction
        prediction = self.link_classifier.predict_proba([features])[0]
        return prediction[1]  # Probability of success
    
    def navigate_from_homepage(self, homepage_url: str, html_content: str, max_time_minutes: int = 10) -> NavigationResult:
        """
        Navigate from homepage to find course search page using depth-first search with backtracking
        
        Args:
            homepage_url: Starting URL
            html_content: Homepage HTML content
            max_time_minutes: Maximum time to search (default: 10 minutes)
            
        Returns:
            NavigationResult with the best path found
        """
        import time
        import requests
        from urllib.parse import urljoin
        
        start_time = time.time()
        max_time_seconds = max_time_minutes * 60
        
        # Track visited URLs to avoid cycles
        visited_urls = set()
        
        # Stack for depth-first search: (url, html_content, navigation_path, remaining_links)
        search_stack = []
        
        # Initialize with homepage
        initial_links = self.extract_links_from_html(html_content, homepage_url)
        ranked_links = self.rank_links(initial_links)
        
        if ranked_links:
            # Add all links to stack (will be processed in order of confidence)
            search_stack.append((homepage_url, html_content, [], ranked_links))
        
        best_result = NavigationResult(
            success=False,
            final_url=homepage_url,
            confidence=0.0,
            navigation_path=[],
            error_message="No course search page found"
        )
        
        while search_stack and (time.time() - start_time) < max_time_seconds:
            current_url, current_html, current_path, remaining_links = search_stack.pop()
            
            # Check if we've found a course search page
            if self._is_course_search_page(current_url, current_html):
                confidence = sum(link.confidence for link in current_path) / len(current_path) if current_path else 0.0
                return NavigationResult(
                    success=True,
                    final_url=current_url,
                    confidence=confidence,
                    navigation_path=current_path
                )
            
            # If no more links to explore from this page, backtrack
            if not remaining_links:
                continue
            
            # Get the next best link
            next_link = remaining_links.pop(0)
            
            # Skip if we've already visited this URL
            if next_link.url in visited_urls:
                continue
            
            # Add current state back to stack (for backtracking)
            if remaining_links:
                search_stack.append((current_url, current_html, current_path, remaining_links))
            
            # Try to fetch the next page
            try:
                logger.info(f"Exploring: {next_link.text} -> {next_link.url} (confidence: {next_link.confidence:.2f})")
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(next_link.url, headers=headers, timeout=10)
                response.raise_for_status()
                
                next_html = response.text
                visited_urls.add(next_link.url)
                
                # Create new navigation path
                new_path = current_path + [next_link]
                
                # Extract links from the new page
                next_links = self.extract_links_from_html(next_html, next_link.url)
                ranked_next_links = self.rank_links(next_links)
                
                # Add to search stack (will be processed in order of confidence)
                if ranked_next_links:
                    search_stack.append((next_link.url, next_html, new_path, ranked_next_links))
                
                # Update best result if this path has higher confidence
                path_confidence = sum(link.confidence for link in new_path) / len(new_path)
                if path_confidence > best_result.confidence:
                    best_result = NavigationResult(
                        success=False,
                        final_url=next_link.url,
                        confidence=path_confidence,
                        navigation_path=new_path,
                        error_message="Best path found so far (not a course search page)"
                    )
                
            except Exception as e:
                logger.warning(f"Error fetching {next_link.url}: {e}")
                # Continue with next link
                continue
        
        # Check if we timed out
        if (time.time() - start_time) >= max_time_seconds:
            best_result.error_message = f"Search timed out after {max_time_minutes} minutes"
        
        return best_result
    
    def _is_course_search_page(self, url: str, html_content: str) -> bool:
        """Check if a page is a course search page"""
        url_lower = url.lower()
        content_lower = html_content.lower()
        
        # URL-based indicators (strong signals)
        url_indicators = [
            'course', 'class', 'schedule', 'catalog', 'registration',
            'enrollment', 'search', 'finder', 'directory', 'courses',
            'classes', 'schedules', 'registrar', 'academic'
        ]
        
        url_score = 0
        for indicator in url_indicators:
            if indicator in url_lower:
                url_score += 1
        
        # Content-based indicators (strong signals)
        content_indicators = [
            'course search', 'class schedule', 'course catalog',
            'course registration', 'course enrollment', 'course finder',
            'search courses', 'find courses', 'course directory',
            'course listing', 'course offerings', 'course schedule',
            'academic calendar', 'course catalog', 'class search',
            'course finder', 'course search', 'course directory'
        ]
        
        content_score = 0
        for indicator in content_indicators:
            if indicator in content_lower:
                content_score += 1
        
        # Form-based indicators (medium signals)
        form_indicators = ['search form', 'course search', 'class search']
        form_score = 0
        for indicator in form_indicators:
            if indicator in content_lower:
                form_score += 1
        
        # Check for search functionality
        search_functionality = [
            'search', 'find', 'browse', 'filter', 'query',
            'input', 'form', 'submit', 'search button'
        ]
        
        functionality_score = 0
        for indicator in search_functionality:
            if indicator in content_lower:
                functionality_score += 1
        
        # Calculate total score
        total_score = url_score * 2 + content_score * 3 + form_score * 2 + functionality_score
        
        # Return True if we have strong indicators
        return total_score >= 3
    
    def save_model(self):
        """Save the trained model"""
        if not self.is_trained:
            logger.warning("No trained model to save")
            return
        
        model_data = {
            'link_classifier': self.link_classifier,
            'vectorizer': self.vectorizer,
            'training_data': self.training_data
        }
        
        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {self.model_file}")
    
    def load_model(self):
        """Load a trained model"""
        if not os.path.exists(self.model_file):
            logger.warning(f"Model file {self.model_file} not found")
            return
        
        with open(self.model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        self.link_classifier = model_data['link_classifier']
        self.vectorizer = model_data['vectorizer']
        self.training_data = model_data.get('training_data', [])
        self.is_trained = True
        
        logger.info(f"Model loaded from {self.model_file}")
    
    def add_training_example(self, homepage_url: str, successful_path: List[NavigationLink], 
                           all_links: List[NavigationLink]):
        """Add a training example to improve the model"""
        # Convert navigation links to training format
        for link in successful_path:
            example = {
                'link_text': link.text,
                'url': link.url,
                'link_type': link.link_type,
                'success': True
            }
            self.training_data.append(example)
        
        # Add negative examples from other links
        successful_urls = {link.url for link in successful_path}
        for link in all_links:
            if link.url not in successful_urls:
                example = {
                    'link_text': link.text,
                    'url': link.url,
                    'link_type': link.link_type,
                    'success': False
                }
                self.training_data.append(example)
        
        logger.info(f"Added training example with {len(successful_path)} successful links") 
