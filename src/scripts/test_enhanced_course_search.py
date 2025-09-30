#!/usr/bin/env python3
"""
Test Enhanced Course Search Discovery

This script tests the enhanced course search discovery capabilities,
specifically focusing on the IUP example:
- Catalog page: https://www.iup.edu/registrar/catalog/index.html
- Course search page: https://catalog.iup.edu/content.php?catoid=14&navoid=1816
"""

import sys
import os
import requests
from bs4 import BeautifulSoup
import logging
from pathlib import Path

import pytest

# Ensure project src is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

finder_module = pytest.importorskip(
    "ml_course_catalog_finder",
    reason="Legacy ml_course_catalog_finder module not present in current codebase",
)

MLCourseCatalogFinder = finder_module.MLCourseCatalogFinder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_iup_example():
    """Test the IUP example to see if we can discover the course search page."""
    
    print(" TESTING ENHANCED COURSE SEARCH DISCOVERY")
    print("=" * 60)
    
    # Initialize the finder
    finder = MLCourseCatalogFinder()
    
    # Test URLs
    catalog_url = "https://www.iup.edu/registrar/catalog/index.html"
    expected_search_url = "https://catalog.iup.edu/content.php?catoid=14&navoid=1816"
    
    print(f" Catalog URL: {catalog_url}")
    print(f" Expected Course Search URL: {expected_search_url}")
    print()
    
    # Test the enhanced link discovery
    print(" Testing enhanced course search link discovery...")
    discovered_url = finder.check_for_course_search_links(catalog_url)
    
    if discovered_url:
        print(f" DISCOVERED: {discovered_url}")
        
        # Test if it matches the expected URL
        if discovered_url == expected_search_url:
            print(" PERFECT MATCH! Found the exact expected course search page.")
        else:
            print("  Found a different course search page than expected.")
            
        # Validate the discovered URL
        print("\n Validating discovered URL...")
        is_valid, confidence, details = finder.validate_course_search_with_classifier(discovered_url)
        print(f"Validation result: {is_valid} (confidence: {confidence:.3f})")
        print(f"Details: {details}")
        
    else:
        print(" No course search page discovered.")
        
        # Let's analyze what we found on the catalog page
        print("\n Analyzing catalog page content...")
        try:
            response = requests.get(catalog_url, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for any links that might be course search related
                links = soup.find_all('a', href=True)
                course_related_links = []
                
                for link in links:
                    link_text = link.get_text().lower().strip()
                    href = link.get('href', '').lower()
                    
                    # Check for course-related patterns
                    course_patterns = [
                        'course', 'catalog', 'search', 'find', 'browse', 'schedule',
                        'content.php', 'search.php', 'catalog.php'
                    ]
                    
                    for pattern in course_patterns:
                        if pattern in link_text or pattern in href:
                            course_related_links.append({
                                'text': link_text,
                                'href': href,
                                'full_url': link.get('href')
                            })
                            break
                
                print(f"Found {len(course_related_links)} course-related links:")
                for i, link in enumerate(course_related_links[:10], 1):  # Show first 10
                    print(f"  {i}. {link['text']} -> {link['full_url']}")
                    
        except Exception as e:
            print(f"Error analyzing catalog page: {e}")

def test_enhanced_classifier():
    """Test the enhanced course search classifier with the IUP example."""
    
    print("\n TESTING ENHANCED COURSE SEARCH CLASSIFIER")
    print("=" * 60)
    
    # Initialize the finder
    finder = MLCourseCatalogFinder()
    
    # Test URLs
    test_urls = [
        ("https://catalog.iup.edu/content.php?catoid=14&navoid=1816", "IUP Course Search (Expected)"),
        ("https://www.iup.edu/registrar/catalog/index.html", "IUP Catalog Page"),
        ("https://classes.berkeley.edu/", "UC Berkeley Course Search"),
        ("https://catalog.berkeley.edu/", "UC Berkeley Catalog"),
        ("https://www.google.com", "Google (Negative Control)")
    ]
    
    for url, description in test_urls:
        print(f"\n Testing: {description}")
        print(f"URL: {url}")
        
        try:
            is_valid, confidence, details = finder.validate_course_search_with_classifier(url)
            print(f"Result: {' Course Search' if is_valid else ' Not Course Search'} (confidence: {confidence:.3f})")
            print(f"Details: {details}")
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main test function."""
    print(" ENHANCED COURSE SEARCH DISCOVERY TEST")
    print("=" * 60)
    print("Testing the enhanced system's ability to discover course search pages")
    print("from catalog pages, specifically the IUP example.")
    print()
    
    # Test the IUP example
    test_iup_example()
    
    # Test the enhanced classifier
    test_enhanced_classifier()
    
    print("\n Test completed!")

if __name__ == "__main__":
    main() 
