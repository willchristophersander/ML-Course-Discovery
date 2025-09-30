#!/usr/bin/env python3
"""
Test script for the Integrated Course Discovery System

This script demonstrates the complete pipeline:
1. Load university data from JSON
2. Use Google Course Search Finder to discover course search pages
3. Use CollegeTransfer.net to validate discovered pages
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

legacy_module = pytest.importorskip(
    "integrated_course_discovery_system",
    reason="Legacy integrated_course_discovery_system module removed in refactor",
)

IntegratedCourseDiscoverySystem = legacy_module.IntegratedCourseDiscoverySystem

async def test_single_university():
    """Test the system with a single university"""
    print(" Testing Single University")
    print("=" * 50)
    
    system = IntegratedCourseDiscoverySystem()
    
    # Test with a known university
    test_university_name = "University of California, Berkeley"
    
    print(f" Testing: {test_university_name}")
    
    # Create a mock university data object
    UniversityData = legacy_module.UniversityData

    university = UniversityData(
        name=test_university_name,
        domains=["berkeley.edu"],
        web_pages=["https://www.berkeley.edu/"],
        country="United States",
        alpha_two_code="US",
        state_province="CA",
        course_catalog=None
    )
    
    # Run discovery and validation
    result = await system.discover_and_validate_course_search(university)
    
    print(f"\n Results:")
    print(f"   Discovery Success: {result.discovery_success}")
    print(f"   Discovered URL: {result.discovered_url}")
    print(f"   Discovery Confidence: {result.discovery_confidence:.2f}")
    print(f"   Validation Success: {result.validation_success}")
    print(f"   Validation Confidence: {result.validation_confidence:.2f}")
    print(f"   Test Courses Found: {result.test_courses_found}/{result.total_test_courses}")
    print(f"   Discovery Time: {result.discovery_time:.2f}s")
    print(f"   Validation Time: {result.validation_time:.2f}s")
    
    if result.error_message:
        print(f"   Error: {result.error_message}")

async def test_data_loading():
    """Test loading university data"""
    print("\n Testing Data Loading")
    print("=" * 50)
    
    system = IntegratedCourseDiscoverySystem()
    
    print(f"Total universities loaded: {len(system.universities)}")
    
    # Show some US universities
    us_universities = system.filter_us_universities(system.universities)
    print(f"US universities: {len(us_universities)}")
    
    # Show some known universities
    known_universities = system.filter_known_universities(us_universities)
    print(f"Known universities for testing: {len(known_universities)}")
    
    print("\nSample universities:")
    for i, university in enumerate(known_universities[:5]):
        print(f"  {i+1}. {university.name}")
        print(f"     Domains: {university.domains}")
        print(f"     Country: {university.country}")
        print()

async def test_small_validation():
    """Test with a small set of universities"""
    print("\n Testing Small Validation Set")
    print("=" * 50)
    
    system = IntegratedCourseDiscoverySystem()
    
    # Run with just 2 universities for quick testing
    results = await system.run_training_validation(max_universities=2)
    
    # Analyze results
    analysis = system.analyze_results(results)
    
    print(f"\n Quick Analysis:")
    print(f"   Universities Tested: {analysis['total_universities']}")
    print(f"   Discovery Success Rate: {analysis['discovery_success_rate']:.1%}")
    print(f"   Validation Success Rate: {analysis['validation_success_rate']:.1%}")
    
    if analysis['successful_discoveries']:
        print(f"\n Successful Discoveries:")
        for discovery in analysis['successful_discoveries']:
            print(f"   • {discovery['university']}")
            print(f"     URL: {discovery['url']}")
            print(f"     Discovery Confidence: {discovery['discovery_confidence']:.2f}")
            print(f"     Courses Found: {discovery['courses_found']}")
            print()

async def demonstrate_workflow():
    """Demonstrate the complete workflow"""
    print("\n Complete Workflow Demonstration")
    print("=" * 50)
    
    print("1.  Load university data from JSON file")
    print("2.  Filter to US universities with domains")
    print("3.  Use Google Course Search Finder to discover course search pages")
    print("4.  Get test courses from CollegeTransfer.net")
    print("5.  Validate that discovered pages can find expected courses")
    print("6.  Analyze results for training metrics")
    
    print("\n Benefits:")
    print("   • Automated course search page discovery")
    print("   • Real-world validation using CollegeTransfer.net")
    print("   • Training data generation for ML models")
    print("   • Performance metrics for model improvement")
    print("   • Scalable to thousands of universities")

async def main():
    """Main test function"""
    print(" Integrated Course Discovery System Test")
    print("Testing the complete pipeline with university data and validation")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test data loading
    await test_data_loading()
    
    # Test single university
    await test_single_university()
    
    # Test small validation set
    await test_small_validation()
    
    # Demonstrate workflow
    await demonstrate_workflow()
    
    print(f"\n Testing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n Integration test completed successfully!")
    print("\nNext steps:")
    print("1. Run the full system: python integrated_course_discovery_system.py")
    print("2. Scale to more universities")
    print("3. Improve validation accuracy")
    print("4. Use results to train better ML models")

if __name__ == "__main__":
    asyncio.run(main()) 
