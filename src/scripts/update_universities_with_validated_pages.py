#!/usr/bin/env python3
"""
Update Universities with Validated Course Search Pages

This script takes validation results from the random college validation process
and updates the world_universities_and_domains.json file with validated course
search pages. This creates a self-improving system where each validation adds
to the knowledge base.
"""

import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UniversitiesUpdater:
    """Updates universities data with validated course search pages"""
    
    def __init__(self, universities_file: str = None):
        if universities_file is None:
            # Try different possible locations
            possible_paths = [
                "src/data/world_universities_and_domains.json",
                "../data/world_universities_and_domains.json",
                "data/world_universities_and_domains.json"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    universities_file = path
                    break
            else:
                # Default to src/data if none exist
                universities_file = "src/data/world_universities_and_domains.json"
        
        self.universities_file = universities_file
        
        # Create backup directory if it doesn't exist
        backup_dir = os.path.join(os.path.dirname(universities_file), 'backups')
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create backup filename in the backups directory
        backup_filename = f"world_universities_and_domains.json.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.backup_file = os.path.join(backup_dir, backup_filename)
    
    def load_universities(self) -> List[Dict]:
        """Load universities from JSON file"""
        try:
            with open(self.universities_file, 'r') as f:
                universities = json.load(f)
            logger.info(f"Loaded {len(universities)} universities from {self.universities_file}")
            return universities
        except Exception as e:
            logger.error(f"Error loading universities: {e}")
            return []
    
    def save_universities(self, universities: List[Dict]):
        """Save universities to JSON file with backup"""
        try:
            # Create backup
            with open(self.universities_file, 'r') as f:
                original_data = f.read()
            
            with open(self.backup_file, 'w') as f:
                f.write(original_data)
            
            logger.info(f"Created backup: {self.backup_file}")
            
            # Save updated data
            with open(self.universities_file, 'w') as f:
                json.dump(universities, f, indent=2)
            
            logger.info(f"Updated {self.universities_file} with validated course search pages")
            
        except Exception as e:
            logger.error(f"Error saving universities: {e}")
    
    def find_university_by_name(self, universities: List[Dict], university_name: str) -> Optional[int]:
        """Find university index by name (with fuzzy matching)"""
        # Exact match first
        for i, university in enumerate(universities):
            if university['name'].lower() == university_name.lower():
                return i
        
        # Fuzzy matching for common variations
        for i, university in enumerate(universities):
            name = university['name'].lower()
            search_name = university_name.lower()
            
            # Check for common variations
            if (search_name in name or name in search_name or
                search_name.replace('university', 'univ') in name or
                search_name.replace('univ', 'university') in name):
                return i
        
        return None
    
    def update_university_with_course_search(self, universities: List[Dict], 
                                          university_name: str, 
                                          course_search_url: str,
                                          validation_confidence: float) -> bool:
        """Update a university with validated course search page"""
        try:
            # Find the university
            university_index = self.find_university_by_name(universities, university_name)
            
            if university_index is None:
                logger.warning(f"University not found: {university_name}")
                return False
            
            university = universities[university_index]
            
            # Check if course_catalog field already exists and has a value
            if university.get('course_catalog') and university['course_catalog'] != course_search_url:
                logger.info(f"University {university_name} already has course_catalog: {university['course_catalog']}")
                logger.info(f"New validated URL: {course_search_url} (confidence: {validation_confidence:.2f})")
                
                # Only update if new confidence is higher or if current is null
                if validation_confidence > 0.0:  # Any success is sufficient
                    university['course_catalog'] = course_search_url
                    university['course_catalog_confidence'] = validation_confidence
                    university['course_catalog_validated'] = True
                    university['course_catalog_updated'] = datetime.now().isoformat()
                    logger.info(f"Updated course_catalog for {university_name}")
                    return True
                else:
                    logger.info(f"Skipping update due to low confidence: {validation_confidence:.2f}")
                    return False
            else:
                # Add course_catalog field
                university['course_catalog'] = course_search_url
                university['course_catalog_confidence'] = validation_confidence
                university['course_catalog_validated'] = True
                university['course_catalog_updated'] = datetime.now().isoformat()
                logger.info(f"Added course_catalog for {university_name}: {course_search_url}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating university {university_name}: {e}")
            return False
    
    def process_validation_results(self, validation_file: str) -> Dict[str, Any]:
        """Process validation results and update universities"""
        try:
            # Load validation results
            with open(validation_file, 'r') as f:
                validation_data = json.load(f)
            
            detailed_results = validation_data.get('detailed_results', [])
            logger.info(f"Processing {len(detailed_results)} validation results")
            
            # Load universities
            universities = self.load_universities()
            if not universities:
                return {'error': 'Failed to load universities'}
            
            # Track updates
            updates_made = 0
            high_confidence_updates = 0
            skipped_updates = 0
            
            # Process each validation result
            for result in detailed_results:
                university_name = result['university_name']
                discovered_url = result['discovered_url']
                validation_success = result.get('validation_success', False)
                validation_confidence = result.get('validation_confidence', 0.0)
                
                # Only update if validation was successful (even one course found is sufficient)
                if (validation_success and discovered_url and 
                    validation_confidence > 0.0):  # Any success is sufficient
                    
                    if self.update_university_with_course_search(
                        universities, university_name, discovered_url, validation_confidence
                    ):
                        updates_made += 1
                        if validation_confidence > 0.5:
                            high_confidence_updates += 1
                    else:
                        skipped_updates += 1
                else:
                    skipped_updates += 1
            
            # Save updated universities
            self.save_universities(universities)
            
            return {
                'total_results': len(detailed_results),
                'updates_made': updates_made,
                'high_confidence_updates': high_confidence_updates,
                'skipped_updates': skipped_updates,
                'backup_file': self.backup_file
            }
            
        except Exception as e:
            logger.error(f"Error processing validation results: {e}")
            return {'error': str(e)}
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print summary of updates"""
        if 'error' in summary:
            print(f" Error: {summary['error']}")
            return
        
        print(f"\n Universities Update Summary:")
        print(f"Total validation results: {summary['total_results']}")
        print(f"Updates made: {summary['updates_made']}")
        print(f"High confidence updates (>0.5): {summary['high_confidence_updates']}")
        print(f"Skipped updates: {summary['skipped_updates']}")
        print(f"Backup created: {summary['backup_file']}")
        
        if summary['updates_made'] > 0:
            print(f" Successfully updated {summary['updates_made']} universities with validated course search pages")
        else:
            print(" No updates made - check validation confidence thresholds")

def main():
    """Main function to update universities with validated course search pages"""
    
    print(" Universities Update System")
    print("Updating world_universities_and_domains.json with validated course search pages")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize updater
    updater = UniversitiesUpdater()
    
    # Find the most recent validation results file
    validation_dir = "results/validation_results"
    if os.path.exists(validation_dir):
        validation_files = [f for f in os.listdir(validation_dir) if f.startswith('random_college_validation_')]
        if validation_files:
            # Sort by timestamp and get the most recent
            validation_files.sort(reverse=True)
            latest_file = os.path.join(validation_dir, validation_files[0])
            
            print(f" Processing latest validation file: {latest_file}")
            summary = updater.process_validation_results(latest_file)
            updater.print_summary(summary)
        else:
            print(" No validation files found. Run random_college_validation.py first.")
    else:
        print(" Validation results directory not found. Run random_college_validation.py first.")
    
    print(f"\n Universities update completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 