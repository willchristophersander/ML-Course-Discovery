#!/usr/bin/env python3
"""
Course Page Classifier

This module trains a machine learning model to determine whether a web page
contains college courses or not. It leverages the distinctive patterns and
styles of college course pages for accurate classification.
"""

import json
import pickle
import hashlib
import logging
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from bs4 import BeautifulSoup
import requests
from typing import Dict, List, Tuple, Optional

from utils.path_utils import get_models_path
from utils.training_registry import TrainingRunRegistry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoursePageClassifier:
    """
    Machine learning classifier for determining if a web page contains college courses.
    
    This classifier is trained on distinctive patterns found in college course pages:
    - Course code patterns (MATH 101, CS 201)
    - Course description styles
    - Credit hour mentions
    - Prerequisite structures
    - Academic term references
    - Department/school organization
    """
    
    def __init__(self, model_file='course_page_classifier.pkl'):
        self.model_file = get_models_path(model_file)
        self.model = None
        self.vectorizer = None
        self.training_data_hash = None
        self.confidence_threshold = 0.7
        
        # Course-specific features
        self.course_patterns = [
            r'\b[A-Z]{2,4}\s*\d{3,4}\b',  # Course codes like "MATH 101"
            r'\b\d{3,4}\s*[-–]\s*[A-Za-z\s]+\b',  # "101 - Introduction to..."
            r'\bcredit[s]?\b.*\d+',  # Credits mentioned
            r'\bprerequisite[s]?\b',  # Prerequisites
            r'\bcourse\s+description[s]?\b',  # Course descriptions
            r'\bclass\s+schedule\b',  # Class schedule
            r'\bacademic\s+catalog\b',  # Academic catalog
            r'\bcredit\s+hour[s]?\b',  # Credit hours
            r'\bcredit\s+unit[s]?\b',  # Credit units
            r'\bprerequisite[s]?\s*:\s*[A-Z]',  # Prerequisite: MATH 101
            r'\bco-requisite[s]?\b',  # Co-requisites
            r'\bmeets\s+with\s+[A-Z]',  # Meets with MATH 101
            r'\bcross-listed\s+with\s+[A-Z]',  # Cross-listed
            r'\blecture\s+[0-9]+\s+hour[s]?\b',  # Lecture 3 hours
            r'\blab\s+[0-9]+\s+hour[s]?\b',  # Lab 2 hours
            r'\bcontact\s+hour[s]?\b',  # Contact hours
            r'\bgrade\s+of\s+[A-Z]',  # Grade of C or better
            r'\bpassing\s+grade\b',  # Passing grade
            r'\bminimum\s+grade\b',  # Minimum grade
            r'\bgrade\s+point\s+average\b',  # GPA
        ]
        
        # Academic term patterns
        self.term_patterns = [
            r'\bfall\s+[0-9]{4}\b',  # Fall 2024
            r'\bspring\s+[0-9]{4}\b',  # Spring 2024
            r'\bsummer\s+[0-9]{4}\b',  # Summer 2024
            r'\bwinter\s+[0-9]{4}\b',  # Winter 2024
            r'\bsemester\b',  # Semester
            r'\bquarter\b',  # Quarter
            r'\btrimester\b',  # Trimester
            r'\bacademic\s+year\b',  # Academic year
        ]
        
        # Department/school patterns
        self.department_patterns = [
            r'\bdepartment\s+of\s+[A-Z]',  # Department of Mathematics
            r'\bschool\s+of\s+[A-Z]',  # School of Engineering
            r'\bcollege\s+of\s+[A-Z]',  # College of Arts and Sciences
            r'\bfaculty\s+of\s+[A-Z]',  # Faculty of Science
            r'\bdivision\s+of\s+[A-Z]',  # Division of Humanities
        ]
        
        # Course content indicators
        self.content_indicators = [
            'course objectives', 'learning outcomes', 'course goals',
            'course content', 'topics covered', 'course materials',
            'textbook', 'required reading', 'recommended reading',
            'course requirements', 'course policies', 'attendance policy',
            'grading policy', 'course evaluation', 'course assessment',
            'final exam', 'midterm exam', 'course project', 'term paper',
            'course syllabus', 'course outline', 'course schedule',
            'office hours', 'instructor contact', 'teaching assistant',
            'course website', 'blackboard', 'canvas', 'moodle',
            'course announcements', 'course updates', 'course changes'
        ]
        
        # Try to load existing model automatically
        self._load_model()
    
    def is_model_ready(self):
        """Check if the model is trained and ready to use."""
        return self.model is not None
    
    def extract_course_features(self, text_content: str) -> Dict[str, any]:
        """
        Extract course-specific features from page content.
        """
        text_lower = text_content.lower()
        
        features = {}
        
        # Count course pattern matches
        course_matches = 0
        for pattern in self.course_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            course_matches += len(matches)
        features['course_pattern_matches'] = course_matches
        
        # Count term pattern matches
        term_matches = 0
        for pattern in self.term_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            term_matches += len(matches)
        features['term_pattern_matches'] = term_matches
        
        # Count department pattern matches
        dept_matches = 0
        for pattern in self.department_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            dept_matches += len(matches)
        features['department_pattern_matches'] = dept_matches
        
        # Count content indicator matches
        content_matches = 0
        for indicator in self.content_indicators:
            if indicator in text_lower:
                content_matches += 1
        features['content_indicator_matches'] = content_matches
        
        # Course code density
        course_codes = re.findall(r'\b[A-Z]{2,4}\s*\d{3,4}\b', text_content, re.IGNORECASE)
        features['course_code_count'] = len(course_codes)
        features['course_code_density'] = len(course_codes) / max(len(text_content.split()), 1)
        
        # Credit mentions
        credit_mentions = len(re.findall(r'\bcredit[s]?\b', text_lower))
        features['credit_mentions'] = credit_mentions
        
        # Prerequisite mentions
        prereq_mentions = len(re.findall(r'\bprerequisite[s]?\b', text_lower))
        features['prereq_mentions'] = prereq_mentions
        
        # Course-related keyword density
        course_keywords = [
            'course', 'courses', 'class', 'classes', 'credit', 'credits',
            'prerequisite', 'prerequisites', 'syllabus', 'curriculum',
            'undergraduate', 'graduate', 'elective', 'required', 'core',
            'semester', 'quarter', 'trimester', 'academic', 'department',
            'school', 'college', 'faculty', 'instructor', 'professor',
            'textbook', 'assignment', 'exam', 'test', 'quiz', 'grade'
        ]
        
        keyword_count = sum(text_lower.count(keyword) for keyword in course_keywords)
        features['course_keyword_density'] = keyword_count / max(len(text_content.split()), 1)
        
        # Text structure features
        features['text_length'] = len(text_content)
        features['word_count'] = len(text_content.split())
        features['avg_word_length'] = np.mean([len(word) for word in text_content.split()]) if text_content.split() else 0
        
        # HTML structure features (if available)
        features['has_tables'] = 1 if '<table' in text_content.lower() else 0
        features['has_lists'] = 1 if '<ul' in text_content.lower() or '<ol' in text_content.lower() else 0
        features['has_forms'] = 1 if '<form' in text_content.lower() else 0
        
        return features
    
    def prepare_training_data(self, positive_urls: List[str], negative_urls: List[str]) -> Tuple[List[str], List[int]]:
        """
        Prepare training data by fetching and processing URLs.
        """
        texts = []
        labels = []
        
        # Process positive examples (course pages)
        logger.info(f"Processing {len(positive_urls)} positive examples...")
        for url in positive_urls:
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text()
                    texts.append(text)
                    labels.append(1)
                    logger.info(f" Added positive example: {url}")
                else:
                    logger.warning(f" Failed to fetch {url}: {response.status_code}")
            except Exception as e:
                logger.warning(f" Error fetching {url}: {e}")
        
        # Process negative examples (non-course pages)
        logger.info(f"Processing {len(negative_urls)} negative examples...")
        for url in negative_urls:
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text()
                    texts.append(text)
                    labels.append(0)
                    logger.info(f" Added negative example: {url}")
                else:
                    logger.warning(f" Failed to fetch {url}: {response.status_code}")
            except Exception as e:
                logger.warning(f" Error fetching {url}: {e}")
        
        return texts, labels
    
    def train_model(self, positive_urls: List[str], negative_urls: List[str]):
        """
        Train the course page classifier.
        """
        logger.info(" Training Course Page Classifier")
        logger.info("=" * 50)
        
        # Prepare training data
        texts, labels = self.prepare_training_data(positive_urls, negative_urls)
        
        if len(texts) < 10:
            logger.error(" Insufficient training data. Need at least 10 examples.")
            return None
        
        # Create feature matrix
        logger.info(" Creating feature matrix...")
        feature_matrix = []
        for text in texts:
            features = self.extract_course_features(text)
            feature_matrix.append(list(features.values()))
        
        # Convert to numpy array
        X = np.array(feature_matrix)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        logger.info(" Training Random Forest Classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f" Model Accuracy: {accuracy:.3f}")
        logger.info("\n Classification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Save model
        self._save_model()
        
        # Calculate training data hash
        self.training_data_hash = hashlib.md5(
            json.dumps({'positive': positive_urls, 'negative': negative_urls}, sort_keys=True).encode()
        ).hexdigest()
        
        logger.info(" Model training completed!")
        TrainingRunRegistry().log_run(
            model_name="course_page_classifier",
            data_sources=positive_urls + negative_urls,
            metadata={
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "accuracy": float(accuracy)
            }
        )
        return accuracy
    
    def predict_course_page(self, url: str) -> Tuple[bool, float, Dict]:
        """
        Predict whether a URL contains course information.
        Returns: (is_course_page, confidence, details)
        """
        if self.model is None:
            logger.error(" Model not trained! Please call train_model() first.")
            return False, 0.0, {"error": "Model not trained"}
        
        try:
            # Fetch the page
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                return False, 0.0, {"error": f"HTTP {response.status_code}"}
            
            # Parse content
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            
            # Extract features
            features = self.extract_course_features(text)
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(feature_vector)[0]
            confidence = self.model.predict_proba(feature_vector)[0][1]  # Probability of being course page
            
            # Determine result
            is_course_page = confidence >= self.confidence_threshold
            
            details = {
                'features': features,
                'prediction': prediction,
                'confidence': confidence,
                'threshold': self.confidence_threshold
            }
            
            return is_course_page, confidence, details
            
        except Exception as e:
            logger.error(f" Error predicting {url}: {e}")
            return False, 0.0, {"error": str(e)}
    
    def _save_model(self):
        """Save the trained model to disk."""
        if self.model is not None:
            model_data = {
                'model': self.model,
                'training_data_hash': self.training_data_hash
            }
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f" Model saved to {self.model_file}")
    
    def _load_model(self):
        """Load the trained model from disk."""
        try:
            with open(self.model_file, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.training_data_hash = model_data.get('training_data_hash')
            logger.info(f" Model loaded from {self.model_file}")
            return True
        except Exception as e:
            logger.warning(f" Could not load model: {e}")
            return False
    
    def load_or_train(self, positive_urls: List[str], negative_urls: List[str], force_retrain=False):
        """
        Load existing model or train a new one if needed.
        
        Args:
            positive_urls: List of URLs for positive examples
            negative_urls: List of URLs for negative examples
            force_retrain: Force retraining even if model exists
        """
        # If model is already loaded and we're not forcing retrain, use it
        if self.model is not None and not force_retrain:
            logger.info(" Using existing loaded model")
            return True
        
        # Try to load existing model
        if self._load_model() and not force_retrain:
            # Check if training data has changed
            current_hash = hashlib.md5(
                json.dumps({'positive': positive_urls, 'negative': negative_urls}, sort_keys=True).encode()
            ).hexdigest()
            
            if current_hash == self.training_data_hash:
                logger.info(" Using existing model (training data unchanged)")
                return True
            else:
                logger.info(" Training data changed, retraining model...")
        
        # Train new model
        logger.info(" Training new model...")
        return self.train_model(positive_urls, negative_urls)

def create_training_dataset():
    """
    Create a training dataset with known course and non-course pages.
    """
    # Known course catalog pages (positive examples)
    positive_urls = [
        "https://catalog.berkeley.edu",
        "https://catalog.mit.edu", 
        "https://courses.illinois.edu",
        "https://catalog.unc.edu",
        "https://catalog.ufl.edu",
        "https://catalog.umn.edu",
        "https://catalog.kauai.hawaii.edu",
        "https://catalog.apsu.edu",
        "https://catalog.carrollu.edu",
        "https://catalog.govst.edu",
        "https://catalog.marist.edu",
        "https://catalog.tuskegee.edu",
        "https://catalog.ucsf.edu",
        "https://catalog.bscc.edu",
        "https://catalog.nwscc.edu",
        "https://catalog.iowacentral.edu",
        "https://catalog.pgcc.edu",
        "https://catalog.pccc.edu",
        "https://catalog.sanjuancollege.edu",
        "https://catalog.sunybroome.edu"
    ]
    
    # Known non-course pages (negative examples)
    negative_urls = [
        "https://www.google.com",
        "https://www.wikipedia.org", 
        "https://www.github.com",
        "https://www.youtube.com",
        "https://www.amazon.com",
        "https://www.facebook.com",
        "https://www.twitter.com",
        "https://www.linkedin.com",
        "https://www.reddit.com",
        "https://www.stackoverflow.com",
        "https://www.weather.com",
        "https://www.cnn.com",
        "https://www.bbc.com",
        "https://www.nytimes.com",
        "https://www.espn.com",
        "https://www.netflix.com",
        "https://www.spotify.com",
        "https://www.instagram.com",
        "https://www.pinterest.com",
        "https://www.tumblr.com"
    ]
    
    return positive_urls, negative_urls

if __name__ == "__main__":
    # Create classifier
    classifier = CoursePageClassifier()
    
    # Get training data
    positive_urls, negative_urls = create_training_dataset()
    
    # Train or load model
    classifier.load_or_train(positive_urls, negative_urls)
    
    # Test the classifier
    test_urls = [
        "https://catalog.berkeley.edu",  # Should be course page
        "https://www.google.com",       # Should not be course page
        "https://catalog.mit.edu",      # Should be course page
        "https://www.wikipedia.org"     # Should not be course page
    ]
    
    print("\n Testing Course Page Classifier")
    print("=" * 50)
    
    for url in test_urls:
        is_course, confidence, details = classifier.predict_course_page(url)
        status = "" if is_course else ""
        print(f"{status} {url}: {confidence:.3f} confidence")
    
    print("\n Course Page Classifier ready for use!") 
