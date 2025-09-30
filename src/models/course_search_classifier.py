import requests
import pickle
import hashlib
import json
import logging
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import re

from utils.path_utils import get_data_path, get_models_path
from utils.training_registry import TrainingRunRegistry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CourseSearchClassifier:
    """
    A dedicated model for identifying course search interfaces and course catalog pages.
    This model is specifically trained to distinguish between:
    - Course search interfaces (forms, search boxes, course listings)
    - Course catalog pages (detailed course descriptions, requirements)
    - General university pages (non-course related content)
    """
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.training_data_hash = None
        self.model_file = get_models_path('course_search_classifier.pkl')
        self.training_data_file = get_data_path('course_search_training_data.json')
        
    def extract_search_features(self, url, html_content):
        """
        Extract features specific to course search interfaces and catalog pages.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text().lower()
            
            features = {}
            
            # Course search interface indicators
            features['has_search_form'] = len(soup.find_all('form')) > 0
            features['has_search_input'] = len(soup.find_all('input', {'type': ['text', 'search']})) > 0
            features['has_search_button'] = len(soup.find_all('button', string=re.compile(r'search', re.I))) > 0
            features['has_course_search_text'] = any(term in text for term in ['search courses', 'find courses', 'course search'])
            features['has_catalog_text'] = any(term in text for term in ['course catalog', 'academic catalog', 'course catalog'])
            
            # Course catalog page indicators
            features['has_course_descriptions'] = len(re.findall(r'course\s+description|course\s+overview', text)) > 0
            features['has_credit_info'] = len(re.findall(r'credit|credits|credit\s+hours', text)) > 0
            features['has_prerequisites'] = len(re.findall(r'prerequisite|prerequisites|prereq', text)) > 0
            features['has_course_codes'] = len(re.findall(r'[A-Z]{2,4}\s*\d{3,4}', text)) > 0
            features['has_department_names'] = len(re.findall(r'department\s+of|school\s+of', text)) > 0
            
            # Navigation and structure indicators
            features['has_course_links'] = len(soup.find_all('a', href=re.compile(r'course|catalog'))) > 0
            features['has_course_tables'] = len(soup.find_all('table')) > 0
            features['has_course_lists'] = len(soup.find_all(['ul', 'ol'])) > 0
            
            # URL-based indicators
            features['url_has_course'] = 'course' in url.lower()
            features['url_has_catalog'] = 'catalog' in url.lower()
            features['url_has_search'] = 'search' in url.lower()
            
            # Content density indicators
            features['text_length'] = len(text)
            features['word_count'] = len(text.split())
            features['course_keyword_density'] = text.count('course') / max(len(text.split()), 1)
            features['catalog_keyword_density'] = text.count('catalog') / max(len(text.split()), 1)
            
            # Form and interaction indicators
            features['has_dropdown'] = len(soup.find_all('select')) > 0
            features['has_checkbox'] = len(soup.find_all('input', {'type': 'checkbox'})) > 0
            features['has_radio'] = len(soup.find_all('input', {'type': 'radio'})) > 0
            
            # Page structure indicators
            features['has_sidebar'] = len(soup.find_all(['aside', 'nav'])) > 0
            features['has_footer'] = len(soup.find_all('footer')) > 0
            features['has_header'] = len(soup.find_all('header')) > 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {url}: {e}")
            return {}
    
    def prepare_training_data(self):
        """
        Prepare training data for course search classification.
        """
        try:
            with open(self.training_data_file, 'r') as f:
                data = json.load(f)
            
            features_list = []
            labels = []
            
            # Process course search interfaces (label: 2)
            for item in data.get('course_search_interfaces', []):
                features = self.extract_search_features(item['url'], item.get('html', ''))
                if features:
                    features_list.append(list(features.values()))
                    labels.append(2)  # Course search interface
            
            # Process course catalog pages (label: 1)
            for item in data.get('course_catalog_pages', []):
                features = self.extract_search_features(item['url'], item.get('html', ''))
                if features:
                    features_list.append(list(features.values()))
                    labels.append(1)  # Course catalog page
            
            # Process general pages (label: 0)
            for item in data.get('general_pages', []):
                features = self.extract_search_features(item['url'], item.get('html', ''))
                if features:
                    features_list.append(list(features.values()))
                    labels.append(0)  # General page
            
            return np.array(features_list), np.array(labels)
            
        except FileNotFoundError:
            logger.warning(f"Training data file {self.training_data_file} not found. Creating sample data.")
            return self._create_sample_training_data()
    
    def _create_sample_training_data(self):
        """
        Create sample training data for initial model training.
        """
        # Sample features for different types of pages
        sample_data = {
            'course_search_interfaces': [
                {'url': 'https://example.edu/course-search', 'html': '<form><input type="text" placeholder="Search courses"><button>Search</button></form>'},
                {'url': 'https://example.edu/courses/search', 'html': '<div>Search Courses<input type="text"></div>'},
                {'url': 'https://example.edu/catalog/search', 'html': '<form>Find Courses<input type="search"></form>'}
            ],
            'course_catalog_pages': [
                {'url': 'https://example.edu/catalog', 'html': '<div>Course Catalog<h2>CS 101 - Introduction to Computer Science</h2><p>3 credits. Prerequisites: None</p></div>'},
                {'url': 'https://example.edu/courses', 'html': '<div>Course Descriptions<h3>MATH 201</h3><p>Calculus I. 4 credits.</p></div>'},
                {'url': 'https://example.edu/academics/catalog', 'html': '<div>Academic Catalog<h2>Department of Physics</h2><p>PHYS 101: 4 credits</p></div>'}
            ],
            'general_pages': [
                {'url': 'https://example.edu/about', 'html': '<div>About Our University<p>Founded in 1900</p></div>'},
                {'url': 'https://example.edu/admissions', 'html': '<div>Admissions<h2>How to Apply</h2><p>Application process</p></div>'},
                {'url': 'https://example.edu/contact', 'html': '<div>Contact Us<p>Phone: 555-1234</p></div>'}
            ]
        }
        
        # Save sample data
        with open(self.training_data_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        return self.prepare_training_data()
    
    def train_model(self):
        """
        Train the course search classifier model.
        """
        logger.info("Training course search classifier...")
        
        # Prepare training data
        X, y = self.prepare_training_data()
        
        if len(X) == 0:
            logger.error("No training data available!")
            return False
        
        # Create and train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Save model
        self._save_model()

        logger.info(f"Model trained with {len(X)} samples")
        TrainingRunRegistry().log_run(
            model_name="course_search_classifier",
            data_sources=[self.training_data_file],
            metadata={"samples": int(len(X))}
        )
        return True
    
    def predict_course_search(self, url, html_content):
        """
        Predict if a page is a course search interface (2), course catalog page (1), or general page (0).
        """
        if self.model is None:
            logger.error("Model not trained! Please call train_model() first.")
            return 0, 0.0, {}
        
        try:
            features = self.extract_search_features(url, html_content)
            if not features:
                return 0, 0.0, {'error': 'Could not extract features'}
            
            # Convert features to array
            X = np.array([list(features.values())])
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            confidence = max(probabilities)
            
            # Map prediction to label
            label_map = {0: 'general_page', 1: 'course_catalog', 2: 'course_search_interface'}
            predicted_label = label_map.get(prediction, 'unknown')
            
            details = {
                'prediction': prediction,
                'label': predicted_label,
                'confidence': confidence,
                'probabilities': {
                    'general_page': probabilities[0],
                    'course_catalog': probabilities[1],
                    'course_search_interface': probabilities[2]
                },
                'features': features
            }
            
            return prediction, confidence, details
            
        except Exception as e:
            logger.error(f"Error predicting course search for {url}: {e}")
            return 0, 0.0, {'error': str(e)}
    
    def _save_model(self):
        """Save the trained model."""
        try:
            model_data = {
                'model': self.model,
                'training_data_hash': self.training_data_hash
            }
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {self.model_file}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def _load_model(self):
        """Load the trained model."""
        try:
            with open(self.model_file, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.training_data_hash = model_data.get('training_data_hash')
            logger.info(f"Model loaded from {self.model_file}")
            return True
        except FileNotFoundError:
            logger.info(f"Model file {self.model_file} not found")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def load_or_train(self):
        """
        Load existing model or train a new one if needed.
        """
        # Try to load existing model
        if self._load_model():
            return True
        
        # Train new model if loading failed
        logger.info("No existing model found. Training new model...")
        return self.train_model()
    
    def is_model_ready(self):
        """Check if the model is loaded and ready for prediction."""
        return self.model is not None

def main():
    """Test the course search classifier."""
    classifier = CourseSearchClassifier()
    
    # Load or train model
    if classifier.load_or_train():
        print(" Model ready for prediction")
        
        # Test with sample URLs
        test_cases = [
            ("https://example.edu/course-search", "<form><input type='text' placeholder='Search courses'><button>Search</button></form>"),
            ("https://example.edu/catalog", "<div>Course Catalog<h2>CS 101</h2><p>3 credits</p></div>"),
            ("https://example.edu/about", "<div>About Our University<p>Founded in 1900</p></div>")
        ]
        
        for url, html in test_cases:
            prediction, confidence, details = classifier.predict_course_search(url, html)
            print(f"\nURL: {url}")
            print(f"Prediction: {details.get('label', 'unknown')}")
            print(f"Confidence: {confidence:.3f}")
            print(f"Details: {details}")
    else:
        print(" Failed to load or train model")

if __name__ == "__main__":
    main() 
