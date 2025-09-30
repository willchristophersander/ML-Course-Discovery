#!/usr/bin/env python3
"""
Random College Validation Script

This script randomly selects colleges from the world universities data,
discovers their course search pages using existing models, and validates
them using CollegeTransfer.net data to build a high-quality training dataset.

The validation process:
1. Randomly select US colleges from world universities data
2. Use existing ML models to discover course search pages
3. Look up college courses on CollegeTransfer.net
4. Test if discovered course search pages can find those courses
5. Build training dataset from validated examples
"""

import asyncio
import json
import logging
import random
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.integrated_system import IntegratedCourseDiscoverySystem
from core.college_transfer_search import search_university_courses
from core.course_discovery import EnhancedCourseSearchFinder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of college course search validation"""
    university_name: str
    university_domain: str
    discovered_url: str
    discovery_success: bool
    discovery_confidence: float
    collegetransfer_courses_found: int
    total_collegetransfer_courses: int
    validation_success: bool
    validation_confidence: float
    error_message: Optional[str] = None
    discovery_time: float = 0.0
    validation_time: float = 0.0
    collegetransfer_courses: List[Dict] = None

class RandomCollegeValidator:
    """Validates course search pages for randomly selected colleges"""
    
    def __init__(self, universities_file: str = "../data/world_universities_and_domains.json"):
        self.universities_file = universities_file
        self.course_finder = EnhancedCourseSearchFinder()
        self.universities = self._load_universities()
        self.us_universities = self._filter_us_universities()
        
    def _load_universities(self) -> List[Dict]:
        """Load universities from JSON file"""
        try:
            with open(self.universities_file, 'r') as f:
                universities = json.load(f)
            logger.info(f"Loaded {len(universities)} universities from {self.universities_file}")
            return universities
        except Exception as e:
            logger.error(f"Error loading universities: {e}")
            return []
    
    def _filter_us_universities(self) -> List[Dict]:
        """Filter to only US universities with domains"""
        us_universities = [
            u for u in self.universities 
            if u.get('alpha_two_code') == 'US' and u.get('domains')
        ]
        logger.info(f"Found {len(us_universities)} US universities with domains")
        return us_universities
    
    def select_random_colleges(self, count: int = 10) -> List[Dict]:
        """Randomly select colleges for validation"""
        if count > len(self.us_universities):
            count = len(self.us_universities)
        
        selected = random.sample(self.us_universities, count)
        logger.info(f"Selected {len(selected)} random colleges for validation")
        return selected
    
    async def validate_college(self, university: Dict) -> ValidationResult:
        """Validate a single college's course search page"""
        university_name = university['name']
        domain = university['domains'][0] if university['domains'] else None
        
        logger.info(f"Validating: {university_name} ({domain})")
        
        # Step 1: Discover course search page
        discovery_start = time.time()
        try:
            navigation_result = await self.course_finder.find_course_search_page(university_name)
            discovery_time = time.time() - discovery_start
            
            if not navigation_result.success:
                return ValidationResult(
                    university_name=university_name,
                    university_domain=domain,
                    discovered_url="",
                    discovery_success=False,
                    discovery_confidence=0.0,
                    collegetransfer_courses_found=0,
                    total_collegetransfer_courses=0,
                    validation_success=False,
                    validation_confidence=0.0,
                    error_message="Course search page discovery failed",
                    discovery_time=discovery_time
                )
            
            discovered_url = navigation_result.best_url
            discovery_confidence = navigation_result.confidence
            
        except Exception as e:
            discovery_time = time.time() - discovery_start
            return ValidationResult(
                university_name=university_name,
                university_domain=domain,
                discovered_url="",
                discovery_success=False,
                discovery_confidence=0.0,
                collegetransfer_courses_found=0,
                total_collegetransfer_courses=0,
                validation_success=False,
                validation_confidence=0.0,
                error_message=f"Discovery error: {str(e)}",
                discovery_time=discovery_time
            )
        
        # Step 2: Get courses from CollegeTransfer.net
        validation_start = time.time()
        try:
            collegetransfer_result = await search_university_courses(university_name)
            validation_time = time.time() - validation_start
            
            if not collegetransfer_result or not collegetransfer_result.get('courses'):
                return ValidationResult(
                    university_name=university_name,
                    university_domain=domain,
                    discovered_url=discovered_url,
                    discovery_success=True,
                    discovery_confidence=discovery_confidence,
                    collegetransfer_courses_found=0,
                    total_collegetransfer_courses=0,
                    validation_success=False,
                    validation_confidence=0.0,
                    error_message="No courses found on CollegeTransfer.net",
                    discovery_time=discovery_time,
                    validation_time=validation_time
                )
            
            courses = collegetransfer_result['courses']
            total_courses = len(courses)
            
            # Step 3: Test if discovered page can find these courses
            courses_found = await self._test_course_search_page(discovered_url, courses)
            # Even one course found is sufficient for validation
            validation_success = courses_found > 0
            validation_confidence = 1.0 if validation_success else 0.0  # Binary success/failure
            
            return ValidationResult(
                university_name=university_name,
                university_domain=domain,
                discovered_url=discovered_url,
                discovery_success=True,
                discovery_confidence=discovery_confidence,
                collegetransfer_courses_found=courses_found,
                total_collegetransfer_courses=total_courses,
                validation_success=validation_success,
                validation_confidence=validation_confidence,
                discovery_time=discovery_time,
                validation_time=validation_time,
                collegetransfer_courses=courses
            )
            
        except Exception as e:
            validation_time = time.time() - validation_start
            return ValidationResult(
                university_name=university_name,
                university_domain=domain,
                discovered_url=discovered_url,
                discovery_success=True,
                discovery_confidence=discovery_confidence,
                collegetransfer_courses_found=0,
                total_collegetransfer_courses=0,
                validation_success=False,
                validation_confidence=0.0,
                error_message=f"Validation error: {str(e)}",
                discovery_time=discovery_time,
                validation_time=validation_time
            )
    
    async def _test_course_search_page(self, url: str, courses: List[Dict]) -> int:
        """Test if a course search page can find specific courses"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Get the page content
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            page_text = soup.get_text().lower()
            
            courses_found = 0
            
            # Test each course
            for course in courses:
                course_id = course.get('course_id', '').lower()
                course_title = course.get('course_title', '').lower()
                
                # Try different search strategies
                found = False
                
                # Strategy 1: Exact course ID match
                if course_id in page_text:
                    found = True
                
                # Strategy 2: Exact course title match
                elif course_title in page_text:
                    found = True
                
                # Strategy 3: Partial course ID (just the prefix like "ACC")
                elif course_id and ' ' in course_id:
                    course_prefix = course_id.split(' ')[0]  # Get "ACC" from "ACC 111"
                    if course_prefix in page_text:
                        found = True
                
                # Strategy 4: Partial course title (first few words)
                elif course_title and len(course_title.split()) > 1:
                    title_words = course_title.split()[:3]  # First 3 words
                    title_partial = ' '.join(title_words).lower()
                    if title_partial in page_text:
                        found = True
                
                if found:
                    courses_found += 1
                    logger.debug(f"Found course: {course_id} - {course_title}")
            
            return courses_found
            
        except Exception as e:
            logger.error(f"Error testing course search page {url}: {e}")
            return 0
    
    async def run_validation(self, num_colleges: int = 10) -> List[ValidationResult]:
        """Run validation on randomly selected colleges"""
        logger.info(f"Starting validation of {num_colleges} random colleges")
        
        selected_colleges = self.select_random_colleges(num_colleges)
        results = []
        
        for i, college in enumerate(selected_colleges, 1):
            logger.info(f"Processing college {i}/{len(selected_colleges)}: {college['name']}")
            
            result = await self.validate_college(college)
            results.append(result)
            
            # Add delay between requests
            await asyncio.sleep(2)
        
        return results
    
    def analyze_results(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Analyze validation results"""
        total_colleges = len(results)
        successful_discoveries = sum(1 for r in results if r.discovery_success)
        successful_validations = sum(1 for r in results if r.validation_success)
        
        avg_discovery_confidence = sum(r.discovery_confidence for r in results if r.discovery_success) / max(successful_discoveries, 1)
        avg_validation_confidence = sum(r.validation_confidence for r in results if r.validation_success) / max(successful_validations, 1)
        
        total_courses = sum(r.total_collegetransfer_courses for r in results)
        total_courses_found = sum(r.collegetransfer_courses_found for r in results)
        
        return {
            'total_colleges': total_colleges,
            'successful_discoveries': successful_discoveries,
            'successful_validations': successful_validations,
            'discovery_success_rate': successful_discoveries / total_colleges if total_colleges > 0 else 0,
            'validation_success_rate': successful_validations / total_colleges if total_colleges > 0 else 0,
            'avg_discovery_confidence': avg_discovery_confidence,
            'avg_validation_confidence': avg_validation_confidence,
            'total_courses': total_courses,
            'total_courses_found': total_courses_found,
            'course_finding_rate': total_courses_found / total_courses if total_courses > 0 else 0,
            'high_quality_examples': [
                {
                    'university_name': r.university_name,
                    'discovered_url': r.discovered_url,
                    'discovery_confidence': r.discovery_confidence,
                    'validation_confidence': r.validation_confidence,
                    'courses_found': f"{r.collegetransfer_courses_found}/{r.total_collegetransfer_courses}"
                }
                for r in results 
                if r.validation_success  # Any successful validation is high quality
            ]
        }
    
    def save_results(self, results: List[ValidationResult], analysis: Dict[str, Any]):
        """Save validation results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'results/validation_results/random_college_validation_{timestamp}.json'
        
        # Ensure results directory exists
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis,
                'detailed_results': [
                    {
                        'university_name': r.university_name,
                        'university_domain': r.university_domain,
                        'discovered_url': r.discovered_url,
                        'discovery_success': r.discovery_success,
                        'discovery_confidence': r.discovery_confidence,
                        'validation_success': r.validation_success,
                        'validation_confidence': r.validation_confidence,
                        'collegetransfer_courses_found': r.collegetransfer_courses_found,
                        'total_collegetransfer_courses': r.total_collegetransfer_courses,
                        'error_message': r.error_message,
                        'discovery_time': r.discovery_time,
                        'validation_time': r.validation_time,
                        'collegetransfer_courses': r.collegetransfer_courses
                    }
                    for r in results
                ]
            }, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        return results_file

async def main():
    """Main function to run random college validation"""
    
    print(" Random College Validation System")
    print("Discovering and validating course search pages for random colleges")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize validator
    validator = RandomCollegeValidator()
    
    # Run validation
    results = await validator.run_validation(num_colleges=10)
    
    # Analyze results
    analysis = validator.analyze_results(results)
    
    # Print analysis
    print(f"\n Validation Results:")
    print(f"Total colleges tested: {analysis['total_colleges']}")
    print(f"Successful discoveries: {analysis['successful_discoveries']} ({analysis['discovery_success_rate']:.1%})")
    print(f"Successful validations: {analysis['successful_validations']} ({analysis['validation_success_rate']:.1%})")
    print(f"Average discovery confidence: {analysis['avg_discovery_confidence']:.2f}")
    print(f"Average validation confidence: {analysis['avg_validation_confidence']:.2f}")
    print(f"Course finding rate: {analysis['course_finding_rate']:.1%}")
    
    if analysis['high_quality_examples']:
        print(f"\n High-quality examples found:")
        for example in analysis['high_quality_examples']:
            print(f"  • {example['university_name']}: {example['discovered_url']}")
            print(f"    Discovery confidence: {example['discovery_confidence']:.2f}")
            print(f"    Validation confidence: {example['validation_confidence']:.2f}")
            print(f"    Courses found: {example['courses_found']}")
    
    # Save results
    results_file = validator.save_results(results, analysis)
    
    print(f"\n Results saved to: {results_file}")
    print(f"\n Random college validation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main()) 