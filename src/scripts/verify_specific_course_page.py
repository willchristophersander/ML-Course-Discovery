#!/usr/bin/env python3
"""
Verify Specific Course Search Page

This script allows you to verify a specific course search page by:
1. Taking a university name and course search URL
2. Getting course data from CollegeTransfer.net for that university
3. Testing if the course search page can find those courses
4. Providing detailed verification results

Usage:
    python verify_specific_course_page.py "University Name" "https://course-search-url.com"
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import time
import requests
from bs4 import BeautifulSoup

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.college_transfer_search import search_university_courses

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpecificCoursePageVerifier:
    """Verifies a specific course search page using CollegeTransfer.net"""
    
    def __init__(self):
        pass
    
    async def verify_course_search_page(self, university_name: str, course_search_url: str) -> Dict[str, Any]:
        """Verify a specific course search page"""
        logger.info(f"Verifying course search page for: {university_name}")
        logger.info(f"Course search URL: {course_search_url}")
        
        verification_start = time.time()
        
        try:
            # Step 1: Get courses from CollegeTransfer.net
            print(f" Getting courses from CollegeTransfer.net for: {university_name}")
            collegetransfer_result = await search_university_courses(university_name)
            
            if not collegetransfer_result or not collegetransfer_result:
                return {
                    'university_name': university_name,
                    'course_search_url': course_search_url,
                    'verification_success': False,
                    'verification_confidence': 0.0,
                    'collegetransfer_courses_found': 0,
                    'total_collegetransfer_courses': 0,
                    'error_message': "No courses found on CollegeTransfer.net",
                    'verification_time': time.time() - verification_start,
                    'collegetransfer_courses': [],
                    'found_courses': []
                }
            
            courses = collegetransfer_result  # The function returns the courses list directly
            total_courses = len(courses)
            
            print(f" Found {total_courses} courses on CollegeTransfer.net")
            
            # Step 2: Test if course search page can find these courses
            print(f" Testing course search page: {course_search_url}")
            courses_found, found_courses = await self._test_course_search_page(course_search_url, courses)
            verification_confidence = courses_found / total_courses if total_courses > 0 else 0.0
            
            return {
                'university_name': university_name,
                'course_search_url': course_search_url,
                'verification_success': courses_found > 0,  # Even one course found is success
                'verification_confidence': 1.0 if courses_found > 0 else 0.0,  # Binary success/failure
                'collegetransfer_courses_found': courses_found,
                'total_collegetransfer_courses': total_courses,
                'verification_time': time.time() - verification_start,
                'collegetransfer_courses': courses,
                'found_courses': found_courses,
                'error_message': None
            }
            
        except Exception as e:
            return {
                'university_name': university_name,
                'course_search_url': course_search_url,
                'verification_success': False,
                'verification_confidence': 0.0,
                'collegetransfer_courses_found': 0,
                'total_collegetransfer_courses': 0,
                'error_message': f"Verification error: {str(e)}",
                'verification_time': time.time() - verification_start,
                'collegetransfer_courses': [],
                'found_courses': []
            }
    
    async def _test_course_search_page(self, url: str, courses: List[Dict]) -> tuple[int, List[Dict]]:
        """Test if a course search page can find specific courses"""
        try:
            print(f" Fetching page content from: {url}")
            
            # Get the page content
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            page_text = soup.get_text().lower()
            
            print(f" Page content length: {len(page_text)} characters")
            
            courses_found = 0
            found_courses = []
            
            # Test each course
            for course in courses:
                course_id = course.get('course_id', '').lower()
                course_title = course.get('course_title', '').lower()
                
                # Try different search strategies
                found = False
                found_by = None
                
                # Strategy 1: Exact course ID match
                if course_id in page_text:
                    found = True
                    found_by = 'exact_course_id'
                
                # Strategy 2: Exact course title match
                elif course_title in page_text:
                    found = True
                    found_by = 'exact_course_title'
                
                # Strategy 3: Partial course ID (just the prefix like "ACC")
                elif course_id and ' ' in course_id:
                    course_prefix = course_id.split(' ')[0]  # Get "ACC" from "ACC 111"
                    if course_prefix in page_text:
                        found = True
                        found_by = 'course_prefix'
                
                # Strategy 4: Partial course title (first few words)
                elif course_title and len(course_title.split()) > 1:
                    title_words = course_title.split()[:3]  # First 3 words
                    title_partial = ' '.join(title_words).lower()
                    if title_partial in page_text:
                        found = True
                        found_by = 'partial_title'
                
                if found:
                    courses_found += 1
                    found_courses.append({
                        'course_id': course.get('course_id', ''),
                        'course_title': course.get('course_title', ''),
                        'credits': course.get('credits', ''),
                        'description': course.get('description', ''),
                        'found_by': found_by
                    })
                    print(f" Found course: {course.get('course_id', '')} - {course.get('course_title', '')} (by {found_by})")
                else:
                    print(f" Not found: {course.get('course_id', '')} - {course.get('course_title', '')}")
            
            return courses_found, found_courses
            
        except Exception as e:
            logger.error(f"Error testing course search page {url}: {e}")
            return 0, []
    
    def print_verification_results(self, results: Dict[str, Any]):
        """Print detailed verification results"""
        print(f"\n Verification Results for {results['university_name']}")
        print(f"Course Search URL: {results['course_search_url']}")
        print(f"Verification Success: {' Yes' if results['verification_success'] else ' No'}")
        print(f"Verification Confidence: {results['verification_confidence']:.2f}")
        print(f"Courses Found: {results['collegetransfer_courses_found']}/{results['total_collegetransfer_courses']}")
        print(f"Verification Time: {results['verification_time']:.2f} seconds")
        
        if results['error_message']:
            print(f"Error: {results['error_message']}")
        
        if results['found_courses']:
            print(f"\n Found Courses:")
            for i, course in enumerate(results['found_courses'][:5], 1):  # Show first 5
                print(f"  {i}. {course['course_id']} - {course['course_title']}")
                print(f"     Credits: {course['credits']}")
                print(f"     Found by: {course['found_by']}")
                print()
            
            if len(results['found_courses']) > 5:
                print(f"  ... and {len(results['found_courses']) - 5} more courses")
        
        # Show sample CollegeTransfer.net courses
        if results['collegetransfer_courses']:
            print(f"\n Sample CollegeTransfer.net Courses:")
            for i, course in enumerate(results['collegetransfer_courses'][:3], 1):
                print(f"  {i}. {course.get('course_id', '')} - {course.get('course_title', '')}")
                print(f"     Credits: {course.get('credits', '')}")
                print()
            
            if len(results['collegetransfer_courses']) > 3:
                print(f"  ... and {len(results['collegetransfer_courses']) - 3} more courses")
    
    def save_verification_results(self, results: Dict[str, Any]):
        """Save verification results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_university_name = results['university_name'].replace(' ', '_').replace(',', '').replace('.', '')
        results_file = f'results/validation_results/specific_verification_{safe_university_name}_{timestamp}.json'
        
        # Ensure results directory exists
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'verification_results': results
            }, f, indent=2)
        
        logger.info(f"Verification results saved to: {results_file}")
        return results_file

async def main():
    """Main function to verify a specific course search page"""
    
    if len(sys.argv) != 3:
        print("Usage: python verify_specific_course_page.py 'University Name' 'https://course-search-url.com'")
        print("Example: python verify_specific_course_page.py 'University of Vermont' 'https://catalogue.uvm.edu/'")
        sys.exit(1)
    
    university_name = sys.argv[1]
    course_search_url = sys.argv[2]
    
    print(" Specific Course Search Page Verification")
    print(f"University: {university_name}")
    print(f"Course Search URL: {course_search_url}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize verifier
    verifier = SpecificCoursePageVerifier()
    
    # Run verification
    results = await verifier.verify_course_search_page(university_name, course_search_url)
    
    # Print results
    verifier.print_verification_results(results)
    
    # Save results
    results_file = verifier.save_verification_results(results)
    
    print(f"\n Results saved to: {results_file}")
    print(f"\n Verification completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main()) 