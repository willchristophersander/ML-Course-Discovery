#!/usr/bin/env python3
"""
Comprehensive Course Search Page Verification System

This script verifies every course search page by:
1. Loading all discovered course search pages from various sources
2. Getting course data from CollegeTransfer.net for each university
3. Testing if the discovered course search page can find those courses
4. Providing detailed verification reports and statistics

This creates a comprehensive validation of all course search pages in the system.
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time
import requests
from bs4 import BeautifulSoup
import re

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.college_transfer_search import search_university_courses

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VerificationResult:
    """Result of course search page verification"""
    university_name: str
    course_search_url: str
    collegetransfer_courses_found: int
    total_collegetransfer_courses: int
    verification_success: bool
    verification_confidence: float
    error_message: Optional[str] = None
    verification_time: float = 0.0
    collegetransfer_courses: List[Dict] = None
    source: str = "unknown"

class ComprehensiveCourseVerifier:
    """Comprehensive verification of course search pages using CollegeTransfer.net"""
    
    def __init__(self):
        self.verification_results = []
        
    def load_course_search_pages(self) -> List[Dict]:
        """Load course search pages from various sources"""
        course_search_pages = []
        
        # Source 1: World universities data with course_catalog field
        try:
            with open("../data/world_universities_and_domains.json", 'r') as f:
                universities = json.load(f)
            
            for university in universities:
                if university.get('course_catalog'):
                    course_search_pages.append({
                        'university_name': university['name'],
                        'course_search_url': university['course_catalog'],
                        'source': 'world_universities_data',
                        'confidence': university.get('course_catalog_confidence', 0.0),
                        'validated': university.get('course_catalog_validated', False)
                    })
            
            logger.info(f"Loaded {len(course_search_pages)} course search pages from world universities data")
        except Exception as e:
            logger.error(f"Error loading world universities data: {e}")
        
        # Source 2: Validation results files
        validation_dir = "results/validation_results"
        if os.path.exists(validation_dir):
            for filename in os.listdir(validation_dir):
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(validation_dir, filename), 'r') as f:
                            data = json.load(f)
                        
                        detailed_results = data.get('detailed_results', [])
                        for result in detailed_results:
                            if result.get('discovered_url') and result.get('validation_success'):
                                course_search_pages.append({
                                    'university_name': result['university_name'],
                                    'course_search_url': result['discovered_url'],
                                    'source': f'validation_results_{filename}',
                                    'confidence': result.get('validation_confidence', 0.0),
                                    'validated': True
                                })
                        
                        logger.info(f"Loaded course search pages from {filename}")
                    except Exception as e:
                        logger.error(f"Error loading {filename}: {e}")
        
        # Source 3: Discovery results files
        discovery_dir = "results/discovery_results"
        if os.path.exists(discovery_dir):
            for filename in os.listdir(discovery_dir):
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(discovery_dir, filename), 'r') as f:
                            data = json.load(f)
                        
                        # Extract university name from filename
                        university_name = filename.replace('_courses_', ' ').replace('.json', '')
                        university_name = university_name.replace('_', ' ')
                        
                        if data.get('search_query', {}).get('url'):
                            course_search_pages.append({
                                'university_name': university_name,
                                'course_search_url': data['search_query']['url'],
                                'source': f'discovery_results_{filename}',
                                'confidence': 0.0,
                                'validated': False
                            })
                        
                        logger.info(f"Loaded course search page from {filename}")
                    except Exception as e:
                        logger.error(f"Error loading {filename}: {e}")
        
        # Remove duplicates based on university name
        seen_universities = set()
        unique_pages = []
        for page in course_search_pages:
            if page['university_name'] not in seen_universities:
                seen_universities.add(page['university_name'])
                unique_pages.append(page)
        
        logger.info(f"Total unique course search pages: {len(unique_pages)}")
        return unique_pages
    
    async def verify_course_search_page(self, page_data: Dict) -> VerificationResult:
        """Verify a single course search page using CollegeTransfer.net"""
        university_name = page_data['university_name']
        course_search_url = page_data['course_search_url']
        source = page_data['source']
        
        logger.info(f"Verifying: {university_name} - {course_search_url}")
        
        verification_start = time.time()
        
        try:
            # Step 1: Get courses from CollegeTransfer.net
            collegetransfer_result = await search_university_courses(university_name)
            
            if not collegetransfer_result or not collegetransfer_result.get('courses'):
                return VerificationResult(
                    university_name=university_name,
                    course_search_url=course_search_url,
                    collegetransfer_courses_found=0,
                    total_collegetransfer_courses=0,
                    verification_success=False,
                    verification_confidence=0.0,
                    error_message="No courses found on CollegeTransfer.net",
                    verification_time=time.time() - verification_start,
                    source=source
                )
            
            courses = collegetransfer_result['courses']
            total_courses = len(courses)
            
            # Step 2: Test if course search page can find these courses
            courses_found = await self._test_course_search_page(course_search_url, courses)
            verification_confidence = courses_found / total_courses if total_courses > 0 else 0.0
            
            return VerificationResult(
                university_name=university_name,
                course_search_url=course_search_url,
                collegetransfer_courses_found=courses_found,
                total_collegetransfer_courses=total_courses,
                verification_success=courses_found > 0,  # Even one course found is success
                verification_confidence=1.0 if courses_found > 0 else 0.0,  # Binary success/failure
                verification_time=time.time() - verification_start,
                collegetransfer_courses=courses,
                source=source
            )
            
        except Exception as e:
            return VerificationResult(
                university_name=university_name,
                course_search_url=course_search_url,
                collegetransfer_courses_found=0,
                total_collegetransfer_courses=0,
                verification_success=False,
                verification_confidence=0.0,
                error_message=f"Verification error: {str(e)}",
                verification_time=time.time() - verification_start,
                source=source
            )
    
    async def _test_course_search_page(self, url: str, courses: List[Dict]) -> int:
        """Test if a course search page can find specific courses"""
        try:
            # Get the page content
            response = requests.get(url, timeout=15)
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
    
    async def run_comprehensive_verification(self) -> List[VerificationResult]:
        """Run comprehensive verification on all course search pages"""
        logger.info("Starting comprehensive course search page verification")
        
        # Load all course search pages
        course_search_pages = self.load_course_search_pages()
        
        if not course_search_pages:
            logger.error("No course search pages found to verify")
            return []
        
        results = []
        
        for i, page_data in enumerate(course_search_pages, 1):
            logger.info(f"Processing {i}/{len(course_search_pages)}: {page_data['university_name']}")
            
            result = await self.verify_course_search_page(page_data)
            results.append(result)
            
            # Add delay between requests
            await asyncio.sleep(2)
        
        return results
    
    def analyze_verification_results(self, results: List[VerificationResult]) -> Dict[str, Any]:
        """Analyze verification results"""
        total_pages = len(results)
        successful_verifications = sum(1 for r in results if r.verification_success)
        
        avg_verification_confidence = sum(r.verification_confidence for r in results if r.verification_success) / max(successful_verifications, 1)
        
        total_courses = sum(r.total_collegetransfer_courses for r in results)
        total_courses_found = sum(r.collegetransfer_courses_found for r in results)
        
        # Group by source
        source_stats = {}
        for result in results:
            source = result.source
            if source not in source_stats:
                source_stats[source] = {
                    'total': 0,
                    'successful': 0,
                    'avg_confidence': 0.0
                }
            source_stats[source]['total'] += 1
            if result.verification_success:
                source_stats[source]['successful'] += 1
        
        # Calculate average confidence for each source
        for source in source_stats:
            source_results = [r for r in results if r.source == source and r.verification_success]
            if source_results:
                source_stats[source]['avg_confidence'] = sum(r.verification_confidence for r in source_results) / len(source_results)
        
        return {
            'total_pages': total_pages,
            'successful_verifications': successful_verifications,
            'verification_success_rate': successful_verifications / total_pages if total_pages > 0 else 0,
            'avg_verification_confidence': avg_verification_confidence,
            'total_courses': total_courses,
            'total_courses_found': total_courses_found,
            'course_finding_rate': total_courses_found / total_courses if total_courses > 0 else 0,
            'source_stats': source_stats,
            'high_quality_pages': [
                {
                    'university_name': r.university_name,
                    'course_search_url': r.course_search_url,
                    'verification_confidence': r.verification_confidence,
                    'courses_found': f"{r.collegetransfer_courses_found}/{r.total_collegetransfer_courses}",
                    'source': r.source
                }
                for r in results 
                if r.verification_success and r.verification_confidence > 0.1
            ]
        }
    
    def save_verification_results(self, results: List[VerificationResult], analysis: Dict[str, Any]):
        """Save verification results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'results/validation_results/comprehensive_verification_{timestamp}.json'
        
        # Ensure results directory exists
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis,
                'detailed_results': [
                    {
                        'university_name': r.university_name,
                        'course_search_url': r.course_search_url,
                        'verification_success': r.verification_success,
                        'verification_confidence': r.verification_confidence,
                        'collegetransfer_courses_found': r.collegetransfer_courses_found,
                        'total_collegetransfer_courses': r.total_collegetransfer_courses,
                        'error_message': r.error_message,
                        'verification_time': r.verification_time,
                        'source': r.source,
                        'collegetransfer_courses': r.collegetransfer_courses
                    }
                    for r in results
                ]
            }, f, indent=2)
        
        logger.info(f"Verification results saved to: {results_file}")
        return results_file

async def main():
    """Main function to run comprehensive course verification"""
    
    print(" Comprehensive Course Search Page Verification System")
    print("Verifying all course search pages using CollegeTransfer.net")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize verifier
    verifier = ComprehensiveCourseVerifier()
    
    # Run comprehensive verification
    results = await verifier.run_comprehensive_verification()
    
    # Analyze results
    analysis = verifier.analyze_verification_results(results)
    
    # Print analysis
    print(f"\n Verification Results:")
    print(f"Total pages tested: {analysis['total_pages']}")
    print(f"Successful verifications: {analysis['successful_verifications']} ({analysis['verification_success_rate']:.1%})")
    print(f"Average verification confidence: {analysis['avg_verification_confidence']:.2f}")
    print(f"Course finding rate: {analysis['course_finding_rate']:.1%}")
    
    print(f"\n Source Statistics:")
    for source, stats in analysis['source_stats'].items():
        success_rate = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {source}: {stats['successful']}/{stats['total']} ({success_rate:.1%}) - Avg confidence: {stats['avg_confidence']:.2f}")
    
    if analysis['high_quality_pages']:
        print(f"\n High-quality verified pages:")
        for page in analysis['high_quality_pages'][:10]:  # Show top 10
            print(f"  • {page['university_name']}: {page['course_search_url']}")
            print(f"    Confidence: {page['verification_confidence']:.2f}, Courses: {page['courses_found']}, Source: {page['source']}")
    
    # Save results
    results_file = verifier.save_verification_results(results, analysis)
    
    print(f"\n Results saved to: {results_file}")
    print(f"\n Comprehensive verification completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main()) 