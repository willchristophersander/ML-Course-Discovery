#!/usr/bin/env python3
"""
Simple Course Search Classifier

A robust classifier for distinguishing between course search pages and catalog pages.
Designed to work well with small datasets and provide interpretable results.
"""

import requests
from bs4 import BeautifulSoup
import numpy as np
import json
import pickle
import logging
import re
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from utils.path_utils import get_data_path, get_models_path
from utils.training_registry import TrainingRunRegistry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCourseSearchClassifier:
    """
    A simple but effective classifier for course search pages.
    """
    
    def __init__(self, model_file='simple_course_search_classifier.pkl'):
        self.model_file = get_models_path(model_file)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
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
            # Standard course search patterns
            r'/course-search',
            r'/courses/search',
            r'/catalog/search',
            r'/schedule',
            r'/soc\.',
            r'/courses\.',
            r'/catalog\.',
            r'/classes\.',
            r'/courseexplorer',
            r'/globalsearch',
            r'/coursefinder',
            r'/course-browser',
            r'/course-index',
            r'/course-database',
            r'/course-guide',
            r'/class-search',
            r'/classesearch',
            r'/course-finder',
            
            # Complex URL patterns (like IUP example)
            r'/content\.php\?.*catoid=',
            r'/content\.php\?.*navoid=',
            r'/search\.php\?.*',
            r'/catalog\.php\?.*',
            r'/courses\.php\?.*',
            r'/search\.asp\?.*',
            r'/catalog\.asp\?.*',
            r'/courses\.asp\?.*',
            r'/search\.aspx\?.*',
            r'/catalog\.aspx\?.*',
            r'/courses\.aspx\?.*',
            
            # Query parameter patterns
            r'\?.*search=',
            r'\?.*catalog=',
            r'\?.*course=',
            r'\?.*class=',
            r'\?.*subject=',
            r'\?.*department=',
            r'\?.*term=',
            r'\?.*semester=',
            r'\?.*year=',
            r'\?.*instructor=',
            r'\?.*keyword=',
            r'\?.*title=',
            r'\?.*number=',
            r'\?.*credits=',
            
            # Path-based patterns
            r'/course-search/',
            r'/course-search-results/',
            r'/search-courses/',
            r'/find-courses/',
            r'/browse-courses/',
            r'/course-finder/',
            r'/course-browser/',
            r'/course-index/',
            r'/course-database/',
            r'/course-guide/',
            r'/class-search/',
            r'/classesearch/',
            r'/courseexplorer/',
            r'/globalsearch/',
            r'/coursefinder/',
            
            # Subdomain patterns
            r'courses\.',
            r'catalog\.',
            r'schedule\.',
            r'classes\.',
            r'search\.',
            r'course\.',
        ]
        
        # Catalog page indicators
        self.catalog_indicators = [
            r'\bacademic\s+catalog\b',
            r'\bcourse\s+catalog\b',
            r'\buniversity\s+catalog\b',
            r'\bcollege\s+catalog\b',
            r'\bdegree\s+requirements\b',
            r'\bgraduation\s+requirements\b',
            r'\bacademic\s+programs\b',
            r'\bprograms\s+of\s+study\b',
            r'\bcourse\s+descriptions\b',
            r'\bcourse\s+offerings\b',
            r'\bcourse\s+listings\b',
            r'\bcourse\s+information\b',
            r'\bcourse\s+details\b',
            r'\bcourse\s+overview\b',
            r'\bcourse\s+summary\b',
            r'\bcourse\s+outline\b',
            r'\bcourse\s+syllabus\b',
            r'\bcourse\s+curriculum\b',
            r'\bcourse\s+plan\b',
            r'\bcourse\s+schedule\b',
            r'\bcourse\s+calendar\b',
            r'\bcourse\s+year\b',
            r'\bcourse\s+semester\b',
            r'\bcourse\s+quarter\b',
            r'\bcourse\s+term\b',
            r'\bcourse\s+session\b',
            r'\bcourse\s+period\b',
            r'\bcourse\s+cycle\b',
            r'\bcourse\s+rotation\b',
            r'\bcourse\s+sequence\b',
            r'\bcourse\s+progression\b',
            r'\bcourse\s+pathway\b',
            r'\bcourse\s+track\b',
            r'\bcourse\s+concentration\b',
            r'\bcourse\s+specialization\b',
            r'\bcourse\s+focus\b',
            r'\bcourse\s+emphasis\b',
            r'\bcourse\s+area\b',
            r'\bcourse\s+field\b',
            r'\bcourse\s+discipline\b',
            r'\bcourse\s+subject\b',
            r'\bcourse\s+department\b',
            r'\bcourse\s+school\b',
            r'\bcourse\s+college\b',
            r'\bcourse\s+division\b',
            r'\bcourse\s+unit\b',
            r'\bcourse\s+section\b',
            r'\bcourse\s+component\b',
            r'\bcourse\s+element\b',
            r'\bcourse\s+part\b',
            r'\bcourse\s+module\b',
            r'\bcourse\s+block\b',
            r'\bcourse\s+segment\b',
            r'\bcourse\s+portion\b',
            r'\bcourse\s+fraction\b',
            r'\bcourse\s+piece\b',
            r'\bcourse\s+bit\b',
            r'\bcourse\s+chunk\b',
            r'\bcourse\s+slice\b',
        ]

    def extract_simple_features(self, url, html_content):
        """
        Extract simple but effective features for classification.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text().lower()
            
            features = {}
            
            # Course search pattern matching
            search_pattern_matches = sum(1 for pattern in self.course_search_indicators 
                                      if re.search(pattern, text))
            features['search_pattern_matches'] = search_pattern_matches
            
            # Catalog pattern matching
            catalog_pattern_matches = sum(1 for pattern in self.catalog_indicators 
                                       if re.search(pattern, text))
            features['catalog_pattern_matches'] = catalog_pattern_matches
            
            # URL pattern matching
            url_pattern_matches = sum(1 for pattern in self.course_search_url_patterns 
                                    if re.search(pattern, url.lower()))
            features['url_pattern_matches'] = url_pattern_matches
            
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
            
            # Export functionality (strong indicator of course search)
            export_patterns = [
                r'\bexport\s+all\s+results\b',
                r'\bexport\s+as\s+csv\b',
                r'\bexport\s+as\s+excel\b',
                r'\bdownload\s+results\b',
                r'\bexport\s+data\b',
            ]
            features['has_export_functionality'] = int(any(re.search(pattern, text) for pattern in export_patterns))
            
            # Results section (strong indicator of course search)
            results_patterns = [
                r'\bresults?\s*\(\d+\)\b',
                r'\bcourses?\s+found\b',
                r'\bclasses?\s+found\b',
                r'\bsearch\s+results\b',
                r'\bcourse\s+results\b',
                r'\bclass\s+results\b',
                r'\bfound\s+\d+\s+courses?\b',
                r'\bfound\s+\d+\s+classes?\b',
                r'\bshowing\s+\d+\s+of\s+\d+\s+courses?\b',
                r'\bshowing\s+\d+\s+of\s+\d+\s+classes?\b',
            ]
            features['has_results_section'] = int(any(re.search(pattern, text) for pattern in results_patterns))
            
            # Advanced search functionality
            advanced_search_patterns = [
                r'\badvanced\s+search\b',
                r'\bsearch\s+options\b',
                r'\bsearch\s+criteria\b',
                r'\bsearch\s+filters\b',
                r'\bfilter\s+by\b',
                r'\bsort\s+by\b',
                r'\bsearch\s+parameters\b',
                r'\bsearch\s+settings\b',
            ]
            features['has_advanced_search'] = int(any(re.search(pattern, text) for pattern in advanced_search_patterns))
            
            # Course search specific elements
            course_search_elements = [
                'subject', 'department', 'instructor', 'course number', 'course title',
                'credits', 'prerequisites', 'meeting time', 'meeting location',
                'term', 'semester', 'year', 'campus', 'location', 'section',
                'course code', 'course name', 'class number', 'class title'
            ]
            features['course_search_elements'] = sum(1 for element in course_search_elements if element in text)
            
            # Navigation and breadcrumb patterns (common in course search)
            navigation_patterns = [
                r'\bhome\s*>\s*catalog\b',
                r'\bhome\s*>\s*courses\b',
                r'\bhome\s*>\s*search\b',
                r'\bcatalog\s*>\s*search\b',
                r'\bcourses\s*>\s*search\b',
                r'\bsearch\s*>\s*results\b',
                r'\bcatalog\s*>\s*course\s+search\b',
                r'\bcourses\s*>\s*course\s+search\b',
            ]
            features['has_navigation_patterns'] = int(any(re.search(pattern, text) for pattern in navigation_patterns))
            
            # Table structure (common in course search results)
            tables = soup.find_all('table')
            features['total_tables'] = len(tables)
            
            # List structure (common in course search results)
            lists = soup.find_all(['ul', 'ol'])
            features['total_lists'] = len(lists)
            
            # Link density (course search pages often have many links)
            links = soup.find_all('a')
            features['total_links'] = len(links)
            
            # Content structure analysis
            content_sections = soup.find_all(['div', 'section', 'article'])
            features['content_sections'] = len(content_sections)
            
            # Form complexity (course search forms are often complex)
            form_inputs = []
            for form in forms:
                form_inputs.extend(form.find_all('input'))
                form_inputs.extend(form.find_all('select'))
                form_inputs.extend(form.find_all('textarea'))
            features['total_form_elements'] = len(form_inputs)
            
            # Search form specificity
            search_form_indicators = [
                'search', 'find', 'browse', 'filter', 'sort', 'export',
                'subject', 'department', 'instructor', 'course', 'class',
                'term', 'semester', 'year', 'campus', 'location'
            ]
            form_text = ' '.join([str(form) for form in forms]).lower()
            features['search_form_specificity'] = sum(1 for indicator in search_form_indicators if indicator in form_text)
            
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
            features['has_advanced_filtering'] = int(any(re.search(pattern, text) for pattern in filter_patterns))
            
            # Content structure
            features['text_length'] = len(text)
            features['word_count'] = len(text.split())
            
            # Course-related keyword density
            course_keywords = ['course', 'class', 'credit', 'prerequisite', 'instructor', 'department', 'subject']
            course_keyword_count = sum(text.count(keyword) for keyword in course_keywords)
            features['course_keyword_density'] = course_keyword_count / max(len(text.split()), 1)
            
            # URL features
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            features['path_depth'] = len([p for p in path.split('/') if p])
            features['has_query_params'] = int(bool(parsed_url.query))
            features['path_has_course'] = int('course' in path)
            features['path_has_catalog'] = int('catalog' in path)
            features['path_has_search'] = int('search' in path)
            features['path_has_schedule'] = int('schedule' in path)
            features['path_has_classes'] = int('class' in path)
            
            # Subdomain analysis
            subdomain = parsed_url.netloc.split('.')[0] if '.' in parsed_url.netloc else ''
            features['subdomain_is_course'] = int(subdomain in ['courses', 'catalog', 'schedule', 'soc'])
            features['subdomain_is_search'] = int(subdomain in ['search', 'find', 'explore'])
            
            # Derived features
            features['total_course_indicators'] = (
                features['search_pattern_matches'] +
                features['url_pattern_matches'] +
                features['has_export_functionality'] +
                features['has_results_section'] +
                features['has_advanced_filtering']
            )
            
            features['is_likely_course_search'] = int(
                features['has_export_functionality'] or
                features['has_results_section'] or
                features['has_advanced_filtering'] or
                features['search_pattern_matches'] >= 3 or
                features['url_pattern_matches'] >= 2
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}

    def train_model(self, training_data):
        """
        Train the simple course search classifier.
        """
        logger.info("Training simple course search classifier...")
        
        # Extract features for all training examples
        features_list = []
        labels = []
        
        for item in training_data:
            try:
                features = self.extract_simple_features(item['url'], item['html_content'])
                if features:
                    features_list.append(features)
                    labels.append(1 if item['is_course_search'] else 0)
            except Exception as e:
                logger.warning(f"Error processing {item['url']}: {e}")
        
        if len(features_list) < 5:
            logger.error("Insufficient training data!")
            return False
        
        # Convert to numpy arrays
        self.feature_names = list(features_list[0].keys())
        X = np.array([[f.get(name, 0) for name in self.feature_names] for f in features_list])
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training with {len(X_train)} examples")
        logger.info(f"Testing with {len(X_test)} examples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model accuracy: {accuracy:.3f}")
        logger.info(f"Feature importance: {dict(zip(self.feature_names, self.model.feature_importances_))}")
        
        # Save model
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'accuracy': accuracy
        }
        
        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {self.model_file}")
        TrainingRunRegistry().log_run(
            model_name="simple_course_search_classifier",
            metadata={
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "accuracy": float(accuracy)
            },
            inline_snapshot={"training_data": training_data}
        )
        return True

    def predict_course_search(self, url, html_content):
        """
        Predict if a page is a course search page.
        """
        if self.model is None:
            logger.error("Model not trained! Please call train_model() first.")
            return False, 0.0, {"error": "Model not trained"}
        
        try:
            features = self.extract_simple_features(url, html_content)
            if not features:
                return False, 0.0, {"error": "Could not extract features"}
            
            # Prepare feature vector
            feature_vector = [features.get(name, 0) for name in self.feature_names]
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Make prediction
            prediction = self.model.predict(feature_vector_scaled)[0]
            confidence = self.model.predict_proba(feature_vector_scaled)[0][1]  # Probability of course search
            
            details = {
                'features': features,
                'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
            }
            
            return bool(prediction), confidence, details
            
        except Exception as e:
            logger.error(f"Error predicting course search for {url}: {e}")
            return False, 0.0, {"error": str(e)}

    def load_model(self):
        """Load the trained model."""
        try:
            with open(self.model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            
            logger.info(f"Model loaded from {self.model_file}")
            logger.info(f"Training accuracy: {model_data.get('accuracy', 'unknown')}")
            return True
            
        except FileNotFoundError:
            logger.info(f"Model file {self.model_file} not found")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

def main():
    """Test the simple course search classifier."""
    
    print(" TESTING SIMPLE COURSE SEARCH CLASSIFIER")
    print("=" * 60)
    
    # Load training data
    try:
        training_data_path = get_data_path('course_search_training_data.json')
        with open(training_data_path, 'r') as f:
            training_data = json.load(f)
        print(f"Loaded {len(training_data)} training examples from {training_data_path}")
    except FileNotFoundError:
        print(" Training data not found! Run analyze_course_search_data.py first.")
        return
    
    # Initialize classifier
    classifier = SimpleCourseSearchClassifier()
    
    # Train model
    print("\n Training model...")
    if classifier.train_model(training_data):
        print(" Model training completed!")
        
        # Test on specific examples
        print("\n Testing on examples...")
        test_cases = [
            ('https://globalsearch.cuny.edu/', 'CUNY Global Class Search', True),
            ('https://classes.berkeley.edu/', 'UC Berkeley Class Schedule', True),
            ('https://catalog.unc.edu/', 'UNC Chapel Hill Catalog', False),
            ('https://catalog.utexas.edu/', 'UT Austin Catalog', False),
        ]
        
        correct = 0
        total = 0
        
        for url, description, expected in test_cases:
            # Find corresponding training data
            for item in training_data:
                if item['url'] == url:
                    is_course_search, confidence, details = classifier.predict_course_search(
                        url, item['html_content']
                    )
                    
                    correct_prediction = (is_course_search == expected)
                    if correct_prediction:
                        correct += 1
                    total += 1
                    
                    status = "" if correct_prediction else ""
                    print(f"{status} {description}")
                    print(f"  Expected: {'Course Search' if expected else 'Catalog'}")
                    print(f"  Predicted: {'Course Search' if is_course_search else 'Catalog'}")
                    print(f"  Confidence: {confidence:.3f}")
                    
                    if 'features' in details:
                        features = details['features']
                        print(f"  Search patterns: {features.get('search_pattern_matches', 0)}")
                        print(f"  URL patterns: {features.get('url_pattern_matches', 0)}")
                        print(f"  Export functionality: {features.get('has_export_functionality', 0)}")
                        print(f"  Results section: {features.get('has_results_section', 0)}")
                        print(f"  Advanced filtering: {features.get('has_advanced_filtering', 0)}")
                    break
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n Test Accuracy: {accuracy:.3f} ({correct}/{total})")
        
        if accuracy >= 0.8:
            print(" Excellent! Model performs very well!")
        elif accuracy >= 0.6:
            print(" Good! Model shows promise!")
        else:
            print("  Model needs more training data or tuning.")
    
    else:
        print(" Model training failed!")

if __name__ == "__main__":
    main() 
