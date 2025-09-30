#!/usr/bin/env python3
"""
Enhanced Course Search Page Classifier v2.0

This version includes improved feature extraction specifically designed
to better recognize course search pages with their unique characteristics.
"""

import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os

from utils.path_utils import get_models_path
from utils.training_registry import TrainingRunRegistry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedCourseSearchPageClassifier:
    """
    Enhanced machine learning classifier for determining if a web page is a course search interface.
    
    This enhanced version includes better feature extraction specifically for course search pages:
    - Export functionality (CSV, Excel)
    - Results sections and pagination
    - Advanced filtering options
    - Department/subject filters
    - Course search specific patterns
    """
    
    def __init__(self, model_file='enhanced_course_search_classifier_v2.pkl'):
        self.model_file = get_models_path(model_file)
        self.model = None
        self.vectorizer = None
        self.confidence_threshold = 0.6  # Lowered threshold for better recall
        
        # Enhanced course search specific patterns
        self.course_search_patterns = [
            r'\bsearch\s+courses?\b',
            r'\bfind\s+courses?\b',
            r'\bcourse\s+search\b',
            r'\bcatalog\s+search\b',
            r'\badvanced\s+search\b',
            r'\bkeyword\b',
            r'\bsubject\b',
            r'\bterm\b',
            r'\bcredit\s+hours?\b',
            r'\bprerequisite[s]?\b',
            r'\binstructor\b',
            r'\bmeets\s+with\b',
            r'\bcross-listed\b',
            r'\bregistration\s+worksheet\b',
            r'\bcourse\s+demand\s+statistics\b',
            r'\bdiscussion\s+section\b',
            r'\bclass\s+schedule\b',
            r'\bacademic\s+calendar\b',
            r'\btextbook\s+information\b',
            # Enhanced patterns for course search
            r'\bexport\s+all\s+results\b',
            r'\bexport\s+as\s+csv\b',
            r'\bresults?\s*\(\d+\)\b',
            r'\bsort\s+by\b',
            r'\bfilter\s+by\b',
            r'\bdepartments?\b',
            r'\bsubjects?\b',
            r'\bterms?\b',
            r'\bsemesters?\b',
            r'\bcourse\s+explorer\b',
            r'\bclass\s+search\b',
            r'\bschedule\s+of\s+classes\b',
            r'\bcourse\s+listings?\b',
            r'\bcourse\s+offerings?\b',
            r'\bglobal\s+class\s+search\b',
            r'\bcourse\s+finder\b',
            r'\bcourse\s+browser\b',
            r'\bcourse\s+index\b',
            r'\bcourse\s+database\b',
            r'\bcourse\s+guide\b',
        ]
        
        # Enhanced form and input patterns
        self.form_patterns = [
            r'<input[^>]*type=["\'](text|search|email|tel)["\'][^>]*>',
            r'<select[^>]*>',
            r'<textarea[^>]*>',
            r'<button[^>]*>',
            r'<form[^>]*>',
        ]
        
        # Enhanced URL patterns indicating course search
        self.url_patterns = [
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
        ]
        
        # Enhanced non-course search indicators
        self.non_search_patterns = [
            r'\babout\s+us\b',
            r'\badmissions\b',
            r'\bcontact\s+us\b',
            r'\bstudent\s+life\b',
            r'\balumni\b',
            r'\bfaculty\s+and\s+staff\b',
            r'\bresearch\b',
            r'\bnews\b',
            r'\bevents\b',
            r'\bdonate\b',
            r'\bgive\b',
            r'\bapply\b',
            r'\bvisit\b',
            r'\btour\b',
            r'\bvirtual\s+tour\b',
            r'\bcampus\s+map\b',
            r'\bdirections\b',
            r'\bparking\b',
            r'\btransportation\b',
            r'\bhousing\b',
            r'\bdining\b',
            r'\bhealth\s+services\b',
            r'\bcounseling\b',
            r'\bcareer\s+services\b',
            r'\blibrary\b',
            r'\bathletics\b',
            r'\bsports\b',
            r'\bclubs\b',
            r'\borganizations\b',
        ]
        
        # Course search specific features
        self.course_search_features = [
            r'\bexport\s+all\s+results\b',
            r'\bexport\s+as\s+csv\b',
            r'\bexport\s+as\s+excel\b',
            r'\bresults?\s*\(\d+\)\b',
            r'\bsort\s+by\b',
            r'\bfilter\s+by\b',
            r'\bdepartments?\b',
            r'\bsubjects?\b',
            r'\bterms?\b',
            r'\bsemesters?\b',
            r'\bcourse\s+explorer\b',
            r'\bclass\s+search\b',
            r'\bschedule\s+of\s+classes\b',
            r'\bcourse\s+listings?\b',
            r'\bcourse\s+offerings?\b',
            r'\bglobal\s+class\s+search\b',
            r'\bcourse\s+finder\b',
            r'\bcourse\s+browser\b',
            r'\bcourse\s+index\b',
            r'\bcourse\s+database\b',
            r'\bcourse\s+guide\b',
            r'\bsearch\s+courses?\b',
            r'\bfind\s+courses?\b',
            r'\bcourse\s+search\b',
            r'\bcatalog\s+search\b',
            r'\badvanced\s+search\b',
        ]

    def extract_enhanced_features(self, content, url):
        """
        Extract enhanced features specifically designed for course search pages.
        """
        try:
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text().lower()
            
            features = {}
            
            # Basic URL features
            features['url_has_course_search'] = any(re.search(pattern, url.lower()) for pattern in self.url_patterns)
            features['url_has_courses'] = 'courses' in url.lower()
            features['url_has_classes'] = 'classes' in url.lower()
            features['url_has_schedule'] = 'schedule' in url.lower()
            features['url_has_catalog'] = 'catalog' in url.lower()
            features['url_has_query_params'] = '?' in url
            
            # Enhanced course search pattern matching
            search_pattern_matches = sum(1 for pattern in self.course_search_patterns if re.search(pattern, text))
            features['search_pattern_matches'] = search_pattern_matches
            
            # Course search specific features
            course_search_feature_matches = sum(1 for pattern in self.course_search_features if re.search(pattern, text))
            features['course_search_feature_matches'] = course_search_feature_matches
            
            # Export functionality (strong indicator of course search)
            export_patterns = [
                r'\bexport\s+all\s+results\b',
                r'\bexport\s+as\s+csv\b',
                r'\bexport\s+as\s+excel\b',
                r'\bdownload\s+results\b',
                r'\bexport\s+data\b',
            ]
            features['has_export_functionality'] = any(re.search(pattern, text) for pattern in export_patterns)
            
            # Results section (strong indicator of course search)
            results_patterns = [
                r'\bresults?\s*\(\d+\)\b',
                r'\bresults?\s+found\b',
                r'\bno\s+results\b',
                r'\bsearch\s+results\b',
            ]
            features['has_results_section'] = any(re.search(pattern, text) for pattern in results_patterns)
            
            # Advanced filtering (strong indicator of course search)
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
            
            # Form elements
            forms = soup.find_all('form')
            inputs = soup.find_all('input')
            selects = soup.find_all('select')
            buttons = soup.find_all('button')
            
            features['form_elements'] = len(forms) + len(inputs) + len(selects) + len(buttons)
            features['has_search_form'] = len(forms) > 0
            features['has_search_input'] = len([inp for inp in inputs if inp.get('type') in ['text', 'search']]) > 0
            features['has_select_dropdowns'] = len(selects) > 0
            
            # Search functionality indicators
            search_indicators = [
                r'\bsearch\b',
                r'\bfind\b',
                r'\bbrowse\b',
                r'\bexplore\b',
                r'\blookup\b',
            ]
            features['search_functionality'] = sum(1 for pattern in search_indicators if re.search(pattern, text))
            
            # Course-related keywords
            course_keywords = [
                r'\bcourse\b',
                r'\bclass\b',
                r'\bcredit\b',
                r'\bprerequisite\b',
                r'\binstructor\b',
                r'\bdepartment\b',
                r'\bsubject\b',
                r'\bterm\b',
                r'\bsemester\b',
            ]
            features['course_keyword_matches'] = sum(1 for pattern in course_keywords if re.search(pattern, text))
            
            # Non-search indicators (negative features)
            non_search_matches = sum(1 for pattern in self.non_search_patterns if re.search(pattern, text))
            features['non_search_pattern_matches'] = non_search_matches
            
            # Content structure
            features['text_length'] = len(text)
            features['word_count'] = len(text.split())
            
            # Enhanced scoring
            features['course_search_score'] = (
                features['search_pattern_matches'] * 2 +
                features['course_search_feature_matches'] * 3 +
                features['has_export_functionality'] * 5 +
                features['has_results_section'] * 4 +
                features['has_advanced_filtering'] * 3 +
                features['search_functionality'] * 2 +
                features['course_keyword_matches'] * 1
            )
            
            # Negative scoring
            features['non_search_penalty'] = features['non_search_pattern_matches'] * -1
            
            # Final enhanced score
            features['enhanced_score'] = features['course_search_score'] + features['non_search_penalty']
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting enhanced features: {e}")
            return {}

    def predict_course_search_page(self, url):
        """
        Predict if a URL is a course search page using enhanced features.
        """
        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                return False, 0.0, {"error": f"HTTP {response.status_code}"}
            
            # Extract enhanced features
            features = self.extract_enhanced_features(response.text, url)
            
            if not features:
                return False, 0.0, {"error": "Could not extract features"}
            
            # Use enhanced scoring for prediction
            enhanced_score = features.get('enhanced_score', 0)
            course_search_score = features.get('course_search_score', 0)
            
            # Enhanced decision logic
            is_course_search = False
            confidence = 0.0
            
            # Strong indicators of course search
            if features.get('has_export_functionality', False):
                is_course_search = True
                confidence = 0.9
            elif features.get('has_results_section', False):
                is_course_search = True
                confidence = 0.8
            elif features.get('has_advanced_filtering', False):
                is_course_search = True
                confidence = 0.7
            elif enhanced_score >= 10:
                is_course_search = True
                confidence = min(0.8, enhanced_score / 20.0)
            elif course_search_score >= 8:
                is_course_search = True
                confidence = min(0.7, course_search_score / 15.0)
            elif features.get('search_pattern_matches', 0) >= 3:
                is_course_search = True
                confidence = 0.6
            elif features.get('course_search_feature_matches', 0) >= 2:
                is_course_search = True
                confidence = 0.5
            else:
                # Check for strong negative indicators
                if features.get('non_search_pattern_matches', 0) >= 3:
                    is_course_search = False
                    confidence = 0.8
                else:
                    is_course_search = False
                    confidence = 0.4
            
            # Ensure confidence is within bounds
            confidence = max(0.0, min(1.0, confidence))
            
            return is_course_search, confidence, {
                'method': 'enhanced_course_search_classifier_v2',
                'features': features,
                'enhanced_score': enhanced_score,
                'course_search_score': course_search_score
            }
            
        except Exception as e:
            logger.error(f"Error predicting course search page: {e}")
            return False, 0.0, {"error": str(e)}

    def load_or_train(self, course_search_urls, non_course_search_urls):
        """
        Load existing model or train a new one with enhanced features.
        """
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Enhanced model loaded successfully")
                return True
            else:
                logger.info("No existing enhanced model found. Training new model...")
                return self.train_model(course_search_urls, non_course_search_urls)
        except Exception as e:
            logger.error(f"Error loading enhanced model: {e}")
            return False

    def train_model(self, course_search_urls, non_course_search_urls):
        """
        Train the enhanced model with comprehensive data.
        """
        try:
            logger.info("Training enhanced course search classifier...")
            
            # Extract features from all URLs
            features_list = []
            labels = []
            
            # Process course search URLs
            for url in course_search_urls:
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        features = self.extract_enhanced_features(response.text, url)
                        if features:
                            features_list.append(features)
                            labels.append(1)  # Course search
                except Exception as e:
                    logger.debug(f"Error processing course search URL {url}: {e}")
            
            # Process non-course search URLs
            for url in non_course_search_urls:
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        features = self.extract_enhanced_features(response.text, url)
                        if features:
                            features_list.append(features)
                            labels.append(0)  # Not course search
                except Exception as e:
                    logger.debug(f"Error processing non-course search URL {url}: {e}")
            
            if len(features_list) < 10:
                logger.warning("Insufficient training data for enhanced model")
                return False
            
            # Convert to numpy arrays
            X = np.array([[f.get('enhanced_score', 0), f.get('course_search_score', 0), 
                          f.get('search_pattern_matches', 0), f.get('course_search_feature_matches', 0),
                          f.get('has_export_functionality', False), f.get('has_results_section', False),
                          f.get('has_advanced_filtering', False), f.get('search_functionality', 0),
                          f.get('course_keyword_matches', 0), f.get('non_search_pattern_matches', 0)] 
                         for f in features_list])
            y = np.array(labels)
            
            # Train Random Forest classifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            
            # Save the model
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.model, f)
            
            logger.info(f"Enhanced model trained successfully with {len(features_list)} examples")
            TrainingRunRegistry().log_run(
                model_name="enhanced_course_search_classifier_v2",
                data_sources=course_search_urls + non_course_search_urls,
                metadata={"examples": int(len(features_list))}
            )
            return True
            
        except Exception as e:
            logger.error(f"Error training enhanced model: {e}")
            return False

    def is_model_ready(self):
        """Check if the enhanced model is ready."""
        return self.model is not None or os.path.exists(self.model_file)

def main():
    """Test the enhanced course search classifier."""
    
    print(" TESTING ENHANCED COURSE SEARCH CLASSIFIER V2.0")
    print("=" * 60)
    
    # Initialize the enhanced classifier
    classifier = EnhancedCourseSearchPageClassifier()
    
    # Test URLs
    test_urls = [
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
    
    print(" TESTING ENHANCED CLASSIFIER:")
    print("-" * 40)
    
    correct_predictions = 0
    for test_case in test_urls:
        url = test_case['url']
        description = test_case['description']
        expected = test_case['expected']
        
        print(f"\n Testing: {url}")
        print(f"Description: {description}")
        print(f"Expected: {'Course Search' if expected else 'Not Course Search'}")
        
        try:
            is_valid, confidence, details = classifier.predict_course_search_page(url)
            
            validation_correct = (is_valid == expected)
            status = " CORRECT" if validation_correct else " INCORRECT"
            
            if validation_correct:
                correct_predictions += 1
            
            print(f"Result: {status}")
            print(f"Prediction: {'Course Search' if is_valid else 'Not Course Search'}")
            print(f"Confidence: {confidence:.1%}")
            
            if 'features' in details:
                features = details['features']
                print(f"Enhanced Score: {features.get('enhanced_score', 0)}")
                print(f"Course Search Score: {features.get('course_search_score', 0)}")
                print(f"Export Functionality: {features.get('has_export_functionality', False)}")
                print(f"Results Section: {features.get('has_results_section', False)}")
                print(f"Advanced Filtering: {features.get('has_advanced_filtering', False)}")
            
        except Exception as e:
            print(f" Error: {e}")
    
    accuracy = correct_predictions / len(test_urls)
    print(f"\n ENHANCED CLASSIFIER ACCURACY: {accuracy:.1%} ({correct_predictions}/{len(test_urls)})")
    
    if accuracy >= 0.8:
        print(" Excellent! Enhanced classifier performs very well!")
    elif accuracy >= 0.6:
        print(" Good! Enhanced classifier shows improvement!")
    else:
        print("  Enhanced classifier needs more improvement.")

if __name__ == "__main__":
    main() 
