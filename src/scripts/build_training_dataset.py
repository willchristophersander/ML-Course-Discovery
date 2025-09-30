#!/usr/bin/env python3
"""
Build Training Dataset from Validated Examples

This script takes validated course search pages from the random college validation
and automatically builds a training dataset by extracting features from the pages.
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests
from bs4 import BeautifulSoup
import re

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.course_search_classifier import CourseSearchClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingDatasetBuilder:
    """Builds training dataset from validated course search pages"""
    
    def __init__(self, validation_results_file: str = None):
        self.classifier = CourseSearchClassifier()
        self.validation_results_file = validation_results_file
        self.training_data = {
            'course_search_interfaces': [],
            'course_catalog_pages': [],
            'general_pages': []
        }
    
    def load_validation_results(self, results_file: str) -> List[Dict]:
        """Load validation results from file"""
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            # Get detailed results
            detailed_results = data.get('detailed_results', [])
            logger.info(f"Loaded {len(detailed_results)} validation results from {results_file}")
            return detailed_results
            
        except Exception as e:
            logger.error(f"Error loading validation results: {e}")
            return []
    
    async def fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch HTML content from a URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def extract_page_features(self, url: str, html_content: str) -> Dict[str, Any]:
        """Extract features from a page using the classifier's feature extraction"""
        try:
            features = self.classifier.extract_search_features(url, html_content)
            return features
        except Exception as e:
            logger.error(f"Error extracting features from {url}: {e}")
            return {}
    
    def determine_page_type(self, features: Dict[str, Any], validation_confidence: float) -> str:
        """Determine if a page is a course search interface, catalog page, or general page"""
        
        # High validation confidence suggests it's a course search interface
        if validation_confidence > 0.1:
            return 'course_search_interfaces'
        
        # Check features to determine type
        has_search_form = features.get('has_search_form', False)
        has_course_descriptions = features.get('has_course_descriptions', False)
        has_catalog_text = features.get('has_catalog_text', False)
        
        if has_search_form or has_catalog_text:
            return 'course_search_interfaces'
        elif has_course_descriptions:
            return 'course_catalog_pages'
        else:
            return 'general_pages'
    
    async def build_training_examples(self, validation_results: List[Dict]) -> Dict[str, List]:
        """Build training examples from validation results"""
        logger.info(f"Building training examples from {len(validation_results)} validation results")
        
        training_examples = {
            'course_search_interfaces': [],
            'course_catalog_pages': [],
            'general_pages': []
        }
        
        for result in validation_results:
            university_name = result['university_name']
            discovered_url = result['discovered_url']
            validation_confidence = result.get('validation_confidence', 0.0)
            
            # Skip if no URL was discovered
            if not discovered_url:
                continue
            
            logger.info(f"Processing: {university_name} - {discovered_url}")
            
            # Fetch page content
            html_content = await self.fetch_page_content(discovered_url)
            if not html_content:
                continue
            
            # Extract features
            features = self.extract_page_features(discovered_url, html_content)
            
            # Determine page type
            page_type = self.determine_page_type(features, validation_confidence)
            
            # Create training example
            example = {
                'url': discovered_url,
                'html': html_content,
                'university_name': university_name,
                'validation_confidence': validation_confidence,
                'has_collegetransfer_courses': result.get('has_collegetransfer_courses', False),
                'features': features,
                'source': 'random_college_validation'
            }
            
            training_examples[page_type].append(example)
            logger.info(f"Added to {page_type}: {university_name}")
        
        return training_examples
    
    def merge_with_existing_training_data(self, new_examples: Dict[str, List], existing_file: str = None):
        """Merge new examples with existing training data"""
        try:
            # Determine the correct data directory path
            if existing_file is None:
                # Try different possible locations - prioritize current_training_data.json
                possible_paths = [
                    "src/data/current_training_data.json",
                    "data/current_training_data.json",
                    "../data/current_training_data.json",
                    "src/data/training_data.json",
                    "../data/training_data.json",
                    "data/training_data.json"
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        existing_file = path
                        break
                else:
                    # Default to src/data if none exist
                    existing_file = "src/data/training_data.json"
            
            # Load existing training data
            if os.path.exists(existing_file):
                with open(existing_file, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {
                    'course_search_interfaces': [],
                    'course_catalog_pages': [],
                    'general_pages': []
                }
            
            # Track URLs to avoid duplicates
            seen_urls = {
                'course_search_interfaces': set(),
                'course_catalog_pages': set(),
                'general_pages': set()
            }
            
            # Add existing URLs to seen set
            for page_type in existing_data:
                if page_type in seen_urls:  # Skip metadata
                    for example in existing_data[page_type]:
                        if 'url' in example:
                            seen_urls[page_type].add(example['url'])
            
            # Merge new examples (avoiding duplicates)
            new_examples_added = 0
            for page_type in new_examples:
                if page_type in seen_urls:  # Skip metadata
                    for example in new_examples[page_type]:
                        if 'url' in example and example['url'] not in seen_urls[page_type]:
                            existing_data[page_type].append(example)
                            seen_urls[page_type].add(example['url'])
                            new_examples_added += 1
            
            # Save updated data to current_training_data.json (growing the file)
            data_dir = os.path.dirname(existing_file)
            current_file = os.path.join(data_dir, "current_training_data.json")
            
            # Ensure directory exists
            os.makedirs(data_dir, exist_ok=True)
            
            with open(current_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            logger.info(f"Updated training data saved to: {current_file}")
            logger.info(f"Added {new_examples_added} new examples")
            
            # Print summary
            total_examples = sum(len(existing_data[page_type]) for page_type in existing_data if page_type != 'metadata')
            logger.info(f"Total training examples: {total_examples}")
            for page_type, examples in existing_data.items():
                if page_type != 'metadata':
                    logger.info(f"  {page_type}: {len(examples)} examples")
            
            return current_file
            
        except Exception as e:
            logger.error(f"Error merging training data: {e}")
            return None
    
    async def process_validation_file(self, validation_file: str):
        """Process a validation results file and build training dataset"""
        logger.info(f"Processing validation file: {validation_file}")
        
        # Load validation results
        validation_results = self.load_validation_results(validation_file)
        
        if not validation_results:
            logger.error("No validation results found")
            return
        
        # Build training examples
        training_examples = await self.build_training_examples(validation_results)
        
        # Merge with existing training data
        merged_file = self.merge_with_existing_training_data(training_examples)
        
        if merged_file:
            logger.info(f" Training dataset successfully built: {merged_file}")
        else:
            logger.error(" Failed to build training dataset")

async def main():
    """Main function to build training dataset"""
    
    print(" Training Dataset Builder")
    print("Building training dataset from validated course search pages")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize builder
    builder = TrainingDatasetBuilder()
    
    # Find the most recent validation results file
    validation_dir = "results/validation_results"
    if os.path.exists(validation_dir):
        validation_files = [f for f in os.listdir(validation_dir) if f.startswith('random_college_validation_')]
        if validation_files:
            # Sort by timestamp and get the most recent
            validation_files.sort(reverse=True)
            latest_file = os.path.join(validation_dir, validation_files[0])
            
            print(f" Processing latest validation file: {latest_file}")
            await builder.process_validation_file(latest_file)
        else:
            print(" No validation files found. Run random_college_validation.py first.")
    else:
        print(" Validation results directory not found. Run random_college_validation.py first.")
    
    print(f"\n Training dataset builder completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main()) 