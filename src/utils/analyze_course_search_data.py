#!/usr/bin/env python3
"""
Analyze Course Search Data

This script analyzes the provided URLs to categorize them as course search pages vs catalog pages,
and prepares training data for the course search navigator model.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import random
import re
from urllib.parse import urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CourseSearchDataAnalyzer:
    """
    Analyzes URLs to categorize them as course search pages vs catalog pages.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
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
            r'/class-search',
            r'/classesearch',
            r'/courseexplorer',
            r'/course-finder',
            r'/course-browser',
            r'/course-index',
            r'/course-database',
            r'/course-guide',
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
            r'\bcourse\s+segment\b',
            r'\bcourse\s+division\b',
            r'\bcourse\s+section\b',
            r'\bcourse\s+part\b',
            r'\bcourse\s+component\b',
            r'\bcourse\s+element\b',
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

    def analyze_url(self, url, expected_type=None):
        """
        Analyze a URL to determine if it's a course search page or catalog page.
        
        Args:
            url: The URL to analyze
            expected_type: 'course_search', 'catalog', or None for auto-detection
            
        Returns:
            dict with analysis results
        """
        try:
            logger.info(f"Analyzing: {url}")
            
            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                return {
                    'url': url,
                    'status_code': response.status_code,
                    'error': f"HTTP {response.status_code}",
                    'is_course_search': False,
                    'is_catalog': False,
                    'confidence': 0.0
                }
            
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text().lower()
            
            # Extract features
            features = self._extract_features(url, text, soup)
            
            # Determine type based on features
            if expected_type:
                # Use expected type for training data
                is_course_search = (expected_type == 'course_search')
                is_catalog = (expected_type == 'catalog')
                confidence = 0.9  # High confidence for labeled data
            else:
                # Auto-detect based on features
                is_course_search, is_catalog, confidence = self._classify_page(features)
            
            return {
                'url': url,
                'status_code': response.status_code,
                'is_course_search': is_course_search,
                'is_catalog': is_catalog,
                'confidence': confidence,
                'features': features,
                'html_content': response.text,
                'expected_type': expected_type
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {url}: {e}")
            return {
                'url': url,
                'error': str(e),
                'is_course_search': False,
                'is_catalog': False,
                'confidence': 0.0
            }

    def _extract_features(self, url, text, soup):
        """
        Extract features from the page content.
        """
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
        
        # Content structure
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        
        # Course-related keyword density
        course_keywords = ['course', 'class', 'credit', 'prerequisite', 'instructor', 'department', 'subject']
        course_keyword_count = sum(text.count(keyword) for keyword in course_keywords)
        features['course_keyword_density'] = course_keyword_count / max(len(text.split()), 1)
        
        return features

    def _classify_page(self, features):
        """
        Classify a page based on extracted features.
        """
        # Strong indicators of course search
        if features.get('has_export_functionality', False):
            return True, False, 0.9
        elif features.get('has_results_section', False):
            return True, False, 0.8
        elif features.get('has_advanced_filtering', False):
            return True, False, 0.7
        elif features.get('search_pattern_matches', 0) >= 3:
            return True, False, 0.6
        elif features.get('url_pattern_matches', 0) >= 2:
            return True, False, 0.5
        
        # Strong indicators of catalog
        elif features.get('catalog_pattern_matches', 0) >= 3:
            return False, True, 0.7
        
        # Weak indicators
        elif features.get('search_pattern_matches', 0) >= 1:
            return True, False, 0.4
        elif features.get('catalog_pattern_matches', 0) >= 1:
            return False, True, 0.4
        else:
            return False, False, 0.3

def main():
    """Analyze the provided URLs and prepare training data."""
    
    print(" ANALYZING COURSE SEARCH DATA")
    print("=" * 60)
    
    # Define the URLs with their expected types
    urls_with_types = [
        # Course Search Pages
        ('https://globalsearch.cuny.edu/', 'course_search'),
        ('https://classes.berkeley.edu/', 'course_search'),
        ('https://classes.osu.edu/', 'course_search'),
        ('https://go.illinois.edu/courseexplorer', 'course_search'),
        ('https://courses.unh.edu/', 'course_search'),
        ('https://coursesearch.uchicago.edu/', 'course_search'),
        ('https://courses.my.harvard.edu/psp/courses/EMPLOYEE/EMPL/h/?tab=HU_CLASS_SEARCH', 'course_search'),
        ('https://saprd.my.uh.edu/psc/saprd/EMPLOYEE/HRMS/c/COMMUNITY_ACCESS.CLASS_SEARCH.GBL', 'course_search'),
        ('https://www.collegetransfer.net/Search/Search-for-Courses', 'course_search'),
        
        # Catalog Pages
        ('https://catalogue.uvm.edu/undergraduate/', 'catalog'),
        ('https://registrar.utah.edu/Catalog-schedules.php', 'catalog'),
        ('https://catalog.unc.edu/', 'catalog'),
        ('https://catalog.utexas.edu/', 'catalog'),
        ('https://catalogs.uky.edu/', 'catalog'),
        ('https://catalog.cofc.edu/', 'catalog'),
        ('https://catalog.uni.edu/', 'catalog'),
        ('https://www.seattlecolleges.edu/academics/academic-catalog', 'catalog'),
        ('https://catalog.calpoly.edu/', 'catalog'),
        ('https://cso.collegesource.com/', 'catalog'),
        
        # Online Learning Platforms (catalog-like)
        ('https://oyc.yale.edu/', 'catalog'),
        ('https://www.edx.org/', 'catalog'),
        ('https://www.coursera.org/browse', 'catalog'),
        ('https://www.futurelearn.com/', 'catalog'),
        ('https://www.classcentral.com/', 'catalog'),
    ]
    
    analyzer = CourseSearchDataAnalyzer()
    results = []
    
    print(f"\n Analyzing {len(urls_with_types)} URLs...")
    print("-" * 40)
    
    for url, expected_type in urls_with_types:
        print(f"\n Analyzing: {url}")
        print(f"Expected type: {expected_type}")
        
        result = analyzer.analyze_url(url, expected_type)
        results.append(result)
        
        if 'error' not in result:
            print(f" Status: {result['status_code']}")
            print(f" Course Search: {result['is_course_search']}")
            print(f" Catalog: {result['is_catalog']}")
            print(f" Confidence: {result['confidence']:.3f}")
            
            if 'features' in result:
                features = result['features']
                print(f" Search patterns: {features.get('search_pattern_matches', 0)}")
                print(f" Catalog patterns: {features.get('catalog_pattern_matches', 0)}")
                print(f" URL patterns: {features.get('url_pattern_matches', 0)}")
                print(f" Export functionality: {features.get('has_export_functionality', False)}")
                print(f" Results section: {features.get('has_results_section', False)}")
                print(f" Advanced filtering: {features.get('has_advanced_filtering', False)}")
        else:
            print(f" Error: {result['error']}")
        
        # Add delay to be respectful
        time.sleep(random.uniform(1, 2))
    
    # Prepare training data
    training_data = []
    course_search_count = 0
    catalog_count = 0
    
    for result in results:
        if 'error' not in result and 'html_content' in result:
            training_item = {
                'url': result['url'],
                'html_content': result['html_content'],
                'is_course_search': result['is_course_search'],
                'is_catalog': result['is_catalog'],
                'expected_type': result.get('expected_type', 'unknown'),
                'confidence': result['confidence'],
                'features': result.get('features', {})
            }
            
            training_data.append(training_item)
            
            if result['is_course_search']:
                course_search_count += 1
            elif result['is_catalog']:
                catalog_count += 1
    
    # Save results
    with open('course_search_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save training data
    with open('course_search_training_data.json', 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"\n ANALYSIS SUMMARY")
    print("=" * 40)
    print(f" Successfully analyzed: {len(training_data)} URLs")
    print(f" Course search pages: {course_search_count}")
    print(f" Catalog pages: {catalog_count}")
    print(f" Results saved to: course_search_analysis_results.json")
    print(f" Training data saved to: course_search_training_data.json")
    
    # Show some examples
    print(f"\n SAMPLE COURSE SEARCH PAGES:")
    for result in results:
        if result.get('is_course_search', False) and 'error' not in result:
            print(f"  • {result['url']}")
    
    print(f"\n SAMPLE CATALOG PAGES:")
    for result in results:
        if result.get('is_catalog', False) and 'error' not in result:
            print(f"  • {result['url']}")
    
    print(f"\n Ready to train the course search navigator model!")
    print("Run: python train_course_search_navigator.py")

if __name__ == "__main__":
    main() 