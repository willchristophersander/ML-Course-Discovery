#!/usr/bin/env python3
"""
Training Script for Course Search Navigator Model

This script helps collect training data and train the course search navigator model.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import random
from urllib.parse import urljoin, urlparse
import logging
import re  # Added missing import for re
import sys
import os

# Ensure the models package is importable when executed as a script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.course_search_navigator_model import CourseSearchNavigatorModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CourseSearchNavigatorTrainer:
    """
    Trainer for the course search navigator model.
    """
    
    def __init__(self):
        self.model = CourseSearchNavigatorModel()
        self.training_data = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def collect_training_data(self, universities_data):
        """
        Collect training data from university websites.
        
        Args:
            universities_data: List of dicts with university information
        """
        logger.info("Collecting training data from university websites...")
        
        for university in universities_data:
            try:
                domain = university.get('domain', '')
                if not domain:
                    continue
                
                logger.info(f"Processing {university.get('name', 'Unknown')} ({domain})")
                
                # Generate candidate URLs
                candidate_urls = self._generate_candidate_urls(domain)
                
                # Test each candidate URL
                for url in candidate_urls:
                    try:
                        response = self.session.get(url, timeout=30)
                        if response.status_code == 200:
                            # Determine if this is a course search page
                            is_course_search = self._is_likely_course_search(url, response.text)
                            
                            training_item = {
                                'url': url,
                                'html_content': response.text,
                                'is_course_search': is_course_search,
                                'university': university.get('name', 'Unknown'),
                                'domain': domain
                            }
                            
                            self.training_data.append(training_item)
                            logger.info(f"  Added: {url} (Course Search: {is_course_search})")
                            
                            # Add some delay to be respectful
                            time.sleep(random.uniform(1, 3))
                            
                    except Exception as e:
                        logger.debug(f"Error processing {url}: {e}")
                
            except Exception as e:
                logger.error(f"Error processing university {university.get('name', 'Unknown')}: {e}")
    
    def _generate_candidate_urls(self, domain):
        """
        Generate candidate URLs for a domain.
        """
        # Clean domain
        if domain.startswith('http'):
            domain = urlparse(domain).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Common course search URL patterns
        patterns = [
            f'https://{domain}/catalog',
            f'https://{domain}/courses',
            f'https://{domain}/academics/catalog',
            f'https://{domain}/academics/courses',
            f'https://{domain}/academic/catalog',
            f'https://{domain}/academic/courses',
            f'https://{domain}/bulletin',
            f'https://{domain}/academics/bulletin',
            f'https://{domain}/academic/bulletin',
            f'https://{domain}/schedule',
            f'https://{domain}/academics/schedule',
            f'https://{domain}/academic/schedule',
            f'https://catalog.{domain}',
            f'https://courses.{domain}',
            f'https://bulletin.{domain}',
            f'https://schedule.{domain}',
            f'https://{domain}/course-catalog',
            f'https://{domain}/course-catalogs',
            f'https://{domain}/course-listings',
            f'https://{domain}/course-offerings',
            f'https://{domain}/course-search',
            f'https://{domain}/courses/search',
            f'https://{domain}/catalog/search',
            f'https://{domain}/schedule-of-classes',
            f'https://{domain}/class-schedule',
            f'https://{domain}/course-explorer',
            f'https://{domain}/course-finder',
            f'https://{domain}/course-browser',
            f'https://{domain}/course-index',
            f'https://{domain}/course-database',
            f'https://{domain}/course-guide',
            f'https://{domain}/global-class-search',
            f'https://{domain}/globalsearch',
            f'https://{domain}/coursefinder',
            f'https://{domain}/courseexplorer',
            f'https://{domain}/coursebrowser',
            f'https://{domain}/courseindex',
            f'https://{domain}/coursedatabase',
            f'https://{domain}/courseguide',
        ]
        
        return patterns
    
    def _is_likely_course_search(self, url, html_content):
        """
        Determine if a page is likely a course search page based on heuristics.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text().lower()
            
            # Strong indicators of course search
            strong_indicators = [
                r'\bexport\s+all\s+results\b',
                r'\bexport\s+as\s+csv\b',
                r'\bresults?\s*\(\d+\)\b',
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
            ]
            
            # Check for strong indicators
            for pattern in strong_indicators:
                if re.search(pattern, text):
                    return True
            
            # Check for search forms with course-related placeholders
            forms = soup.find_all('form')
            for form in forms:
                inputs = form.find_all('input')
                for inp in inputs:
                    placeholder = inp.get('placeholder', '').lower()
                    if any(term in placeholder for term in ['course', 'class', 'catalog', 'search', 'find']):
                        return True
            
            # Check URL patterns
            url_patterns = [
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
            ]
            
            for pattern in url_patterns:
                if re.search(pattern, url.lower()):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error determining if {url} is course search: {e}")
            return False
    
    def save_training_data(self, filename='course_search_navigator_training_data.json'):
        """
        Save training data to file.
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.training_data, f, indent=2)
            logger.info(f"Training data saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
    
    def load_training_data(self, filename='course_search_navigator_training_data.json'):
        """
        Load training data from file.
        """
        try:
            with open(filename, 'r') as f:
                self.training_data = json.load(f)
            logger.info(f"Training data loaded from {filename}")
            return True
        except FileNotFoundError:
            logger.warning(f"Training data file {filename} not found")
            return False
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return False
    
    def train_model(self):
        """
        Train the course search navigator model.
        """
        if not self.training_data:
            logger.error("No training data available!")
            return False
        
        logger.info(f"Training model with {len(self.training_data)} examples...")
        
        # Count positive and negative examples
        positive_count = sum(1 for item in self.training_data if item['is_course_search'])
        negative_count = len(self.training_data) - positive_count
        
        logger.info(f"Positive examples: {positive_count}")
        logger.info(f"Negative examples: {negative_count}")
        
        # Train the model
        success = self.model.train_model(self.training_data)
        
        if success:
            logger.info("Model training completed successfully!")
        else:
            logger.error("Model training failed!")
        
        return success
    
    def evaluate_model(self, test_urls):
        """
        Evaluate the trained model on test URLs.
        """
        logger.info("Evaluating model on test URLs...")
        
        correct_predictions = 0
        total_predictions = 0
        
        for test_case in test_urls:
            url = test_case['url']
            expected = test_case['expected']
            
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    is_course_search, confidence, details = self.model.predict_course_search(url, response.text)
                    
                    correct = (is_course_search == expected)
                    if correct:
                        correct_predictions += 1
                    total_predictions += 1
                    
                    status = "" if correct else ""
                    logger.info(f"{status} {url}: Predicted={is_course_search}, Expected={expected}, Confidence={confidence:.3f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {url}: {e}")
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            logger.info(f"Model accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
            return accuracy
        else:
            logger.error("No predictions made!")
            return 0.0

def main():
    """Main training script."""
    
    print(" COURSE SEARCH NAVIGATOR MODEL TRAINING")
    print("=" * 60)
    
    trainer = CourseSearchNavigatorTrainer()
    
    # Load universities data
    try:
        with open('world_universities_and_domains.json', 'r') as f:
            universities_data = json.load(f)
        print(f"Loaded {len(universities_data)} universities")
    except FileNotFoundError:
        print(" Universities data file not found!")
        return
    
    # Check if we should collect new data or use existing
    print("\n TRAINING DATA OPTIONS:")
    print("1. Collect new training data from university websites")
    print("2. Use existing training data (if available)")
    print("3. Load existing training data and continue collecting")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\n Collecting new training data...")
        # Limit to first 50 universities for initial training
        limited_universities = universities_data[:50]
        trainer.collect_training_data(limited_universities)
        trainer.save_training_data()
        
    elif choice == "2":
        if not trainer.load_training_data():
            print(" No existing training data found!")
            return
        
    elif choice == "3":
        trainer.load_training_data()
        print("\n Continuing to collect training data...")
        # Add more universities
        additional_universities = universities_data[50:100]
        trainer.collect_training_data(additional_universities)
        trainer.save_training_data()
    
    else:
        print(" Invalid choice!")
        return
    
    # Train the model
    print("\n Training the model...")
    if trainer.train_model():
        print(" Model training completed!")
        
        # Test the model
        print("\n Testing the model...")
        test_urls = [
            {'url': 'https://catalog.mccc.edu/courses?page=1&cq=', 'expected': True},
            {'url': 'https://catalog.mccc.edu', 'expected': False},
            {'url': 'https://globalsearch.cuny.edu/', 'expected': True},
            {'url': 'https://classes.berkeley.edu/', 'expected': True},
            {'url': 'https://catalog.berkeley.edu', 'expected': False},
        ]
        
        accuracy = trainer.evaluate_model(test_urls)
        
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
