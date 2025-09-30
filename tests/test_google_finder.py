#!/usr/bin/env python3
"""
Test script for Google Course Search Finder with AI Navigation

This script demonstrates the complete system:
1. Google search for course search pages
2. AI model selects best search result
3. AI model navigates to course search page
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

import pytest

# Add src to the path for legacy scripts
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

google_module = pytest.importorskip(
    "google_course_search_finder",
    reason="Legacy google_course_search_finder module no longer ships; skipping legacy integration test",
)

GoogleCourseSearchFinder = google_module.GoogleCourseSearchFinder
SearchResult = google_module.SearchResult
NavigationResult = google_module.NavigationResult

async def test_search_result_classifier():
    """Test the search result classifier with sample data"""
    print(" Testing Search Result Classifier")
    print("=" * 50)
    
    classifier = google_module.SearchResultClassifier()
    
    # Create sample search results
    sample_results = [
        SearchResult(
            title="UC Berkeley Class Schedule",
            url="https://classes.berkeley.edu/",
            snippet="Search for classes at UC Berkeley. Find course schedules and registration information.",
            domain="classes.berkeley.edu"
        ),
        SearchResult(
            title="UC Berkeley About Page",
            url="https://www.berkeley.edu/about/",
            snippet="Learn about UC Berkeley's history, mission, and general information.",
            domain="berkeley.edu"
        ),
        SearchResult(
            title="UC Berkeley Course Catalog",
            url="https://catalog.berkeley.edu/",
            snippet="Browse the complete course catalog for UC Berkeley.",
            domain="catalog.berkeley.edu"
        )
    ]
    
    # Test classification
    best_result, confidence = classifier.predict_best_result(sample_results)
    
    print(f" Classification Results:")
    for i, result in enumerate(sample_results):
        print(f"  {i+1}. {result.title}")
        print(f"     URL: {result.url}")
        print(f"     Domain: {result.domain}")
    
    print(f"\n Best Result: {best_result.title}")
    print(f"   Confidence: {confidence:.2f}")
    print(f"   URL: {best_result.url}")

async def test_page_navigation_classifier():
    """Test the page navigation classifier with sample links"""
    print("\n Testing Page Navigation Classifier")
    print("=" * 50)
    
    classifier = google_module.PageNavigationClassifier()
    
    # Create sample links
    sample_links = [
        {
            'text': 'Course Search',
            'href': '/course-search'
        },
        {
            'text': 'About Us',
            'href': '/about'
        },
        {
            'text': 'Class Schedule',
            'href': '/class-schedule'
        },
        {
            'text': 'Contact',
            'href': '/contact'
        },
        {
            'text': 'Course Catalog',
            'href': '/catalog'
        }
    ]
    
    # Test classification
    best_link, confidence = classifier.predict_best_link(sample_links)
    
    print(f" Link Classification Results:")
    for i, link in enumerate(sample_links):
        print(f"  {i+1}. {link['text']} -> {link['href']}")
    
    print(f"\n Best Link: {best_link['text']}")
    print(f"   Confidence: {confidence:.2f}")
    print(f"   URL: {best_link['href']}")

async def test_google_search():
    """Test Google search functionality"""
    print("\n Testing Google Search")
    print("=" * 50)
    
    search_engine = google_module.GoogleSearchEngine()
    
    # Test search
    query = '"University of California Berkeley" "course search"'
    results = await search_engine.search_google(query, max_results=5)
    
    print(f" Search Query: {query}")
    print(f" Found {len(results)} results:")
    
    for i, result in enumerate(results[:3]):  # Show first 3
        print(f"  {i+1}. {result.title}")
        print(f"     URL: {result.url}")
        print(f"     Domain: {result.domain}")
        print(f"     Snippet: {result.snippet[:100]}...")
        print()

async def test_complete_system():
    """Test the complete system end-to-end"""
    print("\n Testing Complete System")
    print("=" * 50)
    
    finder = GoogleCourseSearchFinder()
    
    # Test universities
    test_cases = [
        {
            'name': 'University of California, Berkeley',
            'expected_domain': 'berkeley.edu'
        },
        {
            'name': 'University of New Hampshire',
            'expected_domain': 'unh.edu'
        },
        {
            'name': 'Ohio State University',
            'expected_domain': 'osu.edu'
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n Testing: {test_case['name']}")
        print("-" * 40)
        
        try:
            result = await finder.find_course_search_page(test_case['name'])
            
            if result.success:
                print(f" Success!")
                print(f"   Final URL: {result.final_url}")
                print(f"   Confidence: {result.confidence:.2f}")
                print(f"   Navigation Path: {' -> '.join(result.navigation_path)}")
                
                # Check if we found the right domain
                if test_case['expected_domain'] in result.final_url:
                    print(f"    Correct domain found!")
                else:
                    print(f"     Different domain than expected")
                
            else:
                print(f" Failed")
                print(f"   Error: {result.error_message}")
                print(f"   Final URL: {result.final_url}")
            
            results.append({
                'university': test_case['name'],
                'success': result.success,
                'url': result.final_url,
                'confidence': result.confidence,
                'error': result.error_message
            })
            
        except Exception as e:
            print(f" Error: {e}")
            results.append({
                'university': test_case['name'],
                'success': False,
                'url': '',
                'confidence': 0.0,
                'error': str(e)
            })
    
    # Print summary
    print(f"\n Summary")
    print("=" * 50)
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    for result in results:
        status = "" if result['success'] else ""
        print(f"{status} {result['university']}: {result['url']}")

async def demonstrate_ai_learning():
    """Demonstrate how the AI models learn and improve"""
    print("\n AI Learning Demonstration")
    print("=" * 50)
    
    print(" The system uses two AI models:")
    print("   1. Search Result Classifier - Ranks Google search results")
    print("   2. Page Navigation Classifier - Selects best links to click")
    
    print("\n Training Data:")
    print("   - Positive examples: Known course search pages")
    print("   - Negative examples: General university pages")
    print("   - Features: Keywords, URLs, domains, text patterns")
    
    print("\n How it works:")
    print("   1. Google search for 'university name' + 'course search'")
    print("   2. AI ranks search results by relevance")
    print("   3. Click best result and extract all links")
    print("   4. AI selects best link to navigate to course search")
    print("   5. Repeat until course search page is found")
    
    print("\n Benefits:")
    print("   - Learns from examples (supervised learning)")
    print("   - Adapts to different university websites")
    print("   - Handles various navigation patterns")
    print("   - Provides confidence scores for decisions")

async def main():
    """Main test function"""
    print(" Google Course Search Finder with AI Navigation")
    print("Testing the complete system with AI-powered navigation")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test individual components
    await test_search_result_classifier()
    await test_page_navigation_classifier()
    await test_google_search()
    
    # Demonstrate AI learning
    await demonstrate_ai_learning()
    
    # Test complete system
    await test_complete_system()
    
    print(f"\n Testing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n System demonstration completed!")
    print("\nNext steps:")
    print("1. Run the main script: python google_course_search_finder.py")
    print("2. Test with different universities")
    print("3. Improve training data for better accuracy")
    print("4. Add more sophisticated navigation strategies")

if __name__ == "__main__":
    asyncio.run(main()) 
