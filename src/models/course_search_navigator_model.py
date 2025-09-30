#!/usr/bin/env python3
"""
Course Search Navigator Model

This model is designed to find course search pages from initial college catalog pages
by analyzing the relationships between pages, link patterns, and content structure.
"""

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re
import json
import pickle
import logging
from urllib.parse import urljoin, urlparse
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from utils.path_utils import get_models_path
from utils.training_registry import TrainingRunRegistry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CourseSearchNavigatorModel:
    """
    Advanced model for navigating from college catalog pages to course search interfaces.
    
    This model uses a multi-modal approach:
    1. Content analysis (text, forms, structure)
    2. Link relationship analysis (graph-based)
    3. URL pattern analysis
    4. Navigation hierarchy analysis
    """
    
    def __init__(self, model_file='course_search_navigator_model.pkl'):
        self.model_file = get_models_path(model_file)
        self.content_classifier = None
        self.link_classifier = None
        self.url_classifier = None
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.feature_names: list[str] | None = None
        
        # Course search indicators
        self.course_search_indicators = [
            r'\bsearch\s+courses?\b',
            r'\bfind\s+courses?\b',
            r'\bcourse\s+search\b',
            r'\bcatalog\s+search\b',
            r'\bclass\s+search\b',
            r'\bschedule\s+of\s+classes\b',
            r'\bcourse\s+explorer\b',
            r'\bcourse\s+finder\b',
            r'\bcourse\s+browser\b',
            r'\bglobal\s+class\s+search\b',
            r'\badvanced\s+search\b',
            r'\bkeyword\s+search\b',
            r'\bexport\s+results\b',
            r'\bresults?\s*\(\d+\)\b',
            r'\bsort\s+by\b',
            r'\bfilter\s+by\b',
            r'\bdepartments?\b',
            r'\bsubjects?\b',
            r'\bterms?\b',
            r'\bsemesters?\b',
            r'\bcampus\b',
            r'\blocation\b',
            r'\binstructor\b',
            r'\bcredit\s+hours?\b',
            r'\bprerequisite[s]?\b',
            r'\bmeets\s+with\b',
            r'\bcross-listed\b',
            r'\bregistration\s+worksheet\b',
            r'\bcourse\s+demand\s+statistics\b',
            r'\bdiscussion\s+section\b',
            r'\bacademic\s+calendar\b',
            r'\btextbook\s+information\b',
        ]
        
        # URL patterns for course search
        self.course_search_url_patterns = [
            r'/course-search',
            r'/courses/search',
            r'/catalog/search',
            r'/schedule',
            r'/soc\.',
            r'/courses\.',
            r'/catalog\.',
            r'/classes\.',
            r'/courseexplorer',
            r'/course-search',
            r'/globalsearch',
            r'/coursefinder',
            r'/course-browser',
            r'/course-index',
            r'/course-database',
            r'/course-guide',
            r'/schedule-of-classes',
            r'/class-schedule',
            r'/course-listings',
            r'/course-offerings',
        ]
        
        # Link text patterns indicating course search
        self.course_search_link_patterns = [
            r'\bsearch\s+courses?\b',
            r'\bfind\s+courses?\b',
            r'\bcourse\s+search\b',
            r'\bcatalog\s+search\b',
            r'\bclass\s+search\b',
            r'\bschedule\b',
            r'\bcourses?\b',
            r'\bcatalog\b',
            r'\bclasses?\b',
            r'\bexplore\s+courses?\b',
            r'\bbrowse\s+courses?\b',
            r'\bcourse\s+finder\b',
            r'\bcourse\s+explorer\b',
            r'\bcourse\s+browser\b',
            r'\bglobal\s+search\b',
            r'\badvanced\s+search\b',
        ]
        
        # Form indicators for course search
        self.course_search_form_indicators = [
            'search',
            'find',
            'browse',
            'explore',
            'course',
            'class',
            'catalog',
            'schedule',
            'subject',
            'department',
            'term',
            'semester',
            'instructor',
            'credit',
            'prerequisite',
        ]

    def extract_content_features(self, html_content, url):
        """
        Extract features from page content that indicate course search functionality.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text().lower()
            
            features = {}
            
            # Course search pattern matching
            search_pattern_matches = sum(1 for pattern in self.course_search_indicators 
                                      if re.search(pattern, text))
            features['search_pattern_matches'] = search_pattern_matches
            
            # Form analysis
            forms = soup.find_all('form')
            inputs = soup.find_all('input')
            selects = soup.find_all('select')
            buttons = soup.find_all('button')
            
            features['total_forms'] = len(forms)
            features['total_inputs'] = len(inputs)
            features['total_selects'] = len(selects)
            features['total_buttons'] = len(buttons)
            
            # Search-specific form elements
            search_inputs = [inp for inp in inputs if inp.get('type') in ['text', 'search']]
            features['search_inputs'] = len(search_inputs)
            
            # Form placeholders and labels
            placeholders = []
            labels = []
            for inp in inputs:
                placeholder = inp.get('placeholder', '').lower()
                if placeholder:
                    placeholders.append(placeholder)
            
            for label in soup.find_all('label'):
                label_text = label.get_text().lower()
                if label_text:
                    labels.append(label_text)
            
            # Check for course search related placeholders/labels
            course_search_placeholders = sum(1 for p in placeholders 
                                          if any(indicator in p for indicator in self.course_search_form_indicators))
            course_search_labels = sum(1 for l in labels 
                                    if any(indicator in l for indicator in self.course_search_form_indicators))
            
            features['course_search_placeholders'] = course_search_placeholders
            features['course_search_labels'] = course_search_labels
            
            # Export functionality (strong indicator)
            export_patterns = [
                r'\bexport\s+all\s+results\b',
                r'\bexport\s+as\s+csv\b',
                r'\bexport\s+as\s+excel\b',
                r'\bdownload\s+results\b',
                r'\bexport\s+data\b',
            ]
            features['has_export_functionality'] = any(re.search(pattern, text) for pattern in export_patterns)
            
            # Results section indicators
            results_patterns = [
                r'\bresults?\s*\(\d+\)\b',
                r'\bresults?\s+found\b',
                r'\bno\s+results\b',
                r'\bsearch\s+results\b',
            ]
            features['has_results_section'] = any(re.search(pattern, text) for pattern in results_patterns)
            
            # Advanced filtering indicators
            filter_patterns = [
                r'\bsort\s+by\b',
                r'\bfilter\s+by\b',
                r'\bdepartments?\b',
                r'\bsubjects?\b',
                r'\bterms?\b',
                r'\bsemesters?\b',
                r'\bcampus\b',
                r'\blocation\b',
            ]
            features['has_advanced_filtering'] = any(re.search(pattern, text) for pattern in filter_patterns)
            
            # Content structure
            features['text_length'] = len(text)
            features['word_count'] = len(text.split())
            
            # Course-related keyword density
            course_keywords = ['course', 'class', 'credit', 'prerequisite', 'instructor', 'department', 'subject']
            course_keyword_count = sum(text.count(keyword) for keyword in course_keywords)
            features['course_keyword_density'] = course_keyword_count / max(len(text.split()), 1)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting content features: {e}")
            return {}

    def extract_link_features(self, html_content, base_url):
        """
        Extract features from links that might lead to course search pages.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = soup.find_all('a', href=True)
            
            features = {}
            
            # Analyze all links
            course_search_links = []
            potential_course_links = []
            
            for link in links:
                href = link.get('href', '')
                link_text = link.get_text().lower().strip()
                
                # Full URL
                full_url = urljoin(base_url, href)
                
                # Check URL patterns
                url_matches_pattern = any(re.search(pattern, full_url.lower()) 
                                        for pattern in self.course_search_url_patterns)
                
                # Check link text patterns
                text_matches_pattern = any(re.search(pattern, link_text) 
                                        for pattern in self.course_search_link_patterns)
                
                if url_matches_pattern or text_matches_pattern:
                    course_search_links.append({
                        'url': full_url,
                        'text': link_text,
                        'url_score': sum(1 for pattern in self.course_search_url_patterns 
                                       if re.search(pattern, full_url.lower())),
                        'text_score': sum(1 for pattern in self.course_search_link_patterns 
                                        if re.search(pattern, link_text))
                    })
                elif any(keyword in link_text for keyword in ['course', 'class', 'catalog', 'schedule']):
                    potential_course_links.append({
                        'url': full_url,
                        'text': link_text
                    })
            
            features['total_links'] = len(links)
            features['course_search_links'] = len(course_search_links)
            features['potential_course_links'] = len(potential_course_links)
            
            # Score the best course search link
            if course_search_links:
                best_link = max(course_search_links, key=lambda x: x['url_score'] + x['text_score'])
                features['best_course_search_score'] = best_link['url_score'] + best_link['text_score']
                features['best_course_search_url'] = best_link['url']
                features['best_course_search_text'] = best_link['text']
            else:
                features['best_course_search_score'] = 0
                features['best_course_search_url'] = ''
                features['best_course_search_text'] = ''
            
            # Convert string features to numeric or remove them
            features_to_remove = []
            for key in features:
                if isinstance(features[key], str):
                    features_to_remove.append(key)
            
            for key in features_to_remove:
                del features[key]
            
            # Navigation structure analysis
            nav_elements = soup.find_all(['nav', 'menu', 'ul', 'ol'])
            features['navigation_elements'] = len(nav_elements)
            
            # Check for navigation menus with course-related items
            course_nav_items = 0
            for nav in nav_elements:
                nav_text = nav.get_text().lower()
                if any(keyword in nav_text for keyword in ['course', 'class', 'catalog', 'schedule', 'academic']):
                    course_nav_items += 1
            
            features['course_navigation_items'] = course_nav_items
            
            return features, course_search_links, potential_course_links
            
        except Exception as e:
            logger.error(f"Error extracting link features: {e}")
            return {}, [], []

    def extract_url_features(self, url):
        """
        Extract features from the URL itself.
        """
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            query = parsed_url.query.lower()
            
            features = {}
            
            # URL pattern matching
            url_pattern_matches = sum(1 for pattern in self.course_search_url_patterns 
                                    if re.search(pattern, path))
            features['url_pattern_matches'] = url_pattern_matches
            
            # Path depth
            path_parts = [p for p in path.split('/') if p]
            features['path_depth'] = len(path_parts)
            
            # Query parameters
            features['has_query_params'] = bool(query)
            
            # Specific path indicators
            features['path_has_course'] = 'course' in path
            features['path_has_catalog'] = 'catalog' in path
            features['path_has_search'] = 'search' in path
            features['path_has_schedule'] = 'schedule' in path
            features['path_has_classes'] = 'class' in path
            
            # Subdomain analysis
            subdomain = parsed_url.netloc.split('.')[0] if '.' in parsed_url.netloc else ''
            features['subdomain_is_course'] = subdomain in ['courses', 'catalog', 'schedule', 'soc']
            features['subdomain_is_search'] = subdomain in ['search', 'find', 'explore']
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting URL features: {e}")
            return {}

    def extract_comprehensive_features(self, url, html_content):
        """
        Extract all features for a given URL and HTML content.
        """
        try:
            # Extract different types of features
            content_features = self.extract_content_features(html_content, url)
            link_features, course_search_links, potential_course_links = self.extract_link_features(html_content, url)
            url_features = self.extract_url_features(url)
            
            # Combine all features
            all_features = {}
            all_features.update(content_features)
            all_features.update(link_features)
            all_features.update(url_features)
            
            # Add derived features
            all_features['total_course_indicators'] = (
                content_features.get('search_pattern_matches', 0) +
                link_features.get('course_search_links', 0) +
                url_features.get('url_pattern_matches', 0)
            )
            
            all_features['is_likely_course_search'] = int(
                content_features.get('has_export_functionality', False) or
                content_features.get('has_results_section', False) or
                content_features.get('has_advanced_filtering', False) or
                link_features.get('best_course_search_score', 0) > 3
            )
            
            # Convert boolean features to integers
            for key in all_features:
                if isinstance(all_features[key], bool):
                    all_features[key] = int(all_features[key])
                elif all_features[key] is None:
                    all_features[key] = 0
            
            return all_features, course_search_links, potential_course_links
            
        except Exception as e:
            logger.error(f"Error extracting comprehensive features: {e}")
            return {}, [], []

    def train_model(self, training_data):
        """
        Train the course search navigator model.
        
        Args:
            training_data: List of dicts with 'url', 'html_content', 'is_course_search' keys
        """
        logger.info("Training course search navigator model...")
        
        # Extract features for all training examples
        features_list = []
        labels = []
        
        for item in training_data:
            try:
                features, _, _ = self.extract_comprehensive_features(item['url'], item['html_content'])
                if features:
                    features_list.append(features)
                    labels.append(1 if item['is_course_search'] else 0)
            except Exception as e:
                logger.warning(f"Error processing training item {item['url']}: {e}")
        
        if len(features_list) < 10:
            logger.error("Insufficient training data!")
            return False
        
        # Convert to numpy arrays and remember feature ordering
        feature_names = list(features_list[0].keys())
        self.feature_names = feature_names
        X = np.array([[f.get(name, 0) for name in feature_names] for f in features_list])
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        logger.info("Model Training Results:")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Test samples: {len(X_test)}")
        logger.info(f"Feature importance: {dict(zip(feature_names, self.model.feature_importances_))}")
        
        # Save model
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': feature_names,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }

        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {self.model_file}")
        TrainingRunRegistry().log_run(
            model_name="course_search_navigator_model",
            metadata={
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test))
            },
            inline_snapshot={"feature_names": feature_names}
        )
        return True
        
    def predict_course_search(self, url, html_content):
        """
        Predict if a page is a course search page.
        
        Returns:
            (is_course_search, confidence, details)
        """
        try:
            if self.model is None:
                logger.error("Model not trained! Please call train_model() first.")
                return False, 0.0, {"error": "Model not trained"}
            
            # Extract features
            features, course_search_links, potential_course_links = self.extract_comprehensive_features(url, html_content)
            
            if not features:
                return False, 0.0, {"error": "Could not extract features"}
            
            # Prepare feature vector
            feature_names = getattr(self.model, "feature_names_in_", None)
            if feature_names is not None:
                feature_vector = [features.get(name, 0) for name in feature_names]
            else:
                feature_names = getattr(self, "feature_names", None)
                if feature_names is None:
                    raise AttributeError(
                        "Navigator model is missing feature_names metadata."
                    )
                feature_vector = [features.get(name, 0) for name in feature_names]
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Make prediction
            prediction = self.model.predict(feature_vector_scaled)[0]
            confidence = self.model.predict_proba(feature_vector_scaled)[0][1]  # Probability of course search
            
            details = {
                'features': features,
                'course_search_links': course_search_links,
                'potential_course_links': potential_course_links,
                'feature_importance': dict(zip(feature_names, self.model.feature_importances_))
            }
            
            return bool(prediction), confidence, details
            
        except Exception as e:
            logger.error(f"Error predicting course search for {url}: {e}")
            return False, 0.0, {"error": str(e)}

    def find_course_search_links(self, url, html_content):
        """
        Find links that might lead to course search pages.
        
        Returns:
            List of candidate URLs with scores
        """
        try:
            _, course_search_links, potential_course_links = self.extract_link_features(html_content, url)
            
            candidates = []
            
            # Add course search links with high scores
            for link in course_search_links:
                score = link['url_score'] + link['text_score']
                candidates.append({
                    'url': link['url'],
                    'text': link['text'],
                    'score': score,
                    'type': 'course_search_link'
                })
            
            # Add potential course links with lower scores
            for link in potential_course_links:
                candidates.append({
                    'url': link['url'],
                    'text': link['text'],
                    'score': 1,  # Base score for potential links
                    'type': 'potential_course_link'
                })
            
            # Sort by score
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error finding course search links: {e}")
            return []

    def load_model(self):
        """Load the trained model."""
        try:
            with open(self.model_file, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data.get('feature_names')
            logger.info(f"Model loaded from {self.model_file}")
            return True
            
        except FileNotFoundError:
            logger.info(f"Model file {self.model_file} not found")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

def main():
    """Test the course search navigator model."""
    
    print(" TESTING COURSE SEARCH NAVIGATOR MODEL")
    print("=" * 60)
    
    # Initialize the model
    model = CourseSearchNavigatorModel()
    
    # Test URLs
    test_cases = [
        {
            'url': 'https://catalog.mccc.edu/courses?page=1&cq=',
            'description': 'MCCC Course search page',
            'expected': True
        },
        {
            'url': 'https://catalog.mccc.edu',
            'description': 'MCCC Main catalog page',
            'expected': False
        },
        {
            'url': 'https://globalsearch.cuny.edu/',
            'description': 'CUNY Global Class Search',
            'expected': True
        },
        {
            'url': 'https://classes.berkeley.edu/',
            'description': 'UC Berkeley Class Schedule',
            'expected': True
        }
    ]
    
    print(" TESTING FEATURE EXTRACTION:")
    print("-" * 40)
    
    for test_case in test_cases:
        url = test_case['url']
        description = test_case['description']
        
        print(f"\n Testing: {url}")
        print(f"Description: {description}")
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                features, course_search_links, potential_links = model.extract_comprehensive_features(url, response.text)
                
                print(f"Content Features:")
                print(f"  - Search pattern matches: {features.get('search_pattern_matches', 0)}")
                print(f"  - Has export functionality: {features.get('has_export_functionality', False)}")
                print(f"  - Has results section: {features.get('has_results_section', False)}")
                print(f"  - Has advanced filtering: {features.get('has_advanced_filtering', False)}")
                
                print(f"Link Features:")
                print(f"  - Course search links: {len(course_search_links)}")
                print(f"  - Potential course links: {len(potential_links)}")
                print(f"  - Best course search score: {features.get('best_course_search_score', 0)}")
                
                print(f"URL Features:")
                print(f"  - URL pattern matches: {features.get('url_pattern_matches', 0)}")
                print(f"  - Path has course: {features.get('path_has_course', False)}")
                print(f"  - Path has catalog: {features.get('path_has_catalog', False)}")
                
                if course_search_links:
                    print(f"  - Top course search link: {course_search_links[0]['url']}")
                
            else:
                print(f" HTTP {response.status_code}")
                
        except Exception as e:
            print(f" Error: {e}")
    
    print(f"\n FEATURE EXTRACTION COMPLETE")
    print("The model is ready for training with real data!")

if __name__ == "__main__":
    main() 
