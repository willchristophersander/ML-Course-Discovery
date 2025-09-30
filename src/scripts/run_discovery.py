#!/usr/bin/env python3
"""
Main execution script for the Course Discovery System

This script provides a clean interface to run the integrated course discovery system
with the new organized project structure.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.integrated_system import IntegratedCourseDiscoverySystem

async def main():
    """Main function to run the integrated system"""
    
    print(" Course Discovery System")
    print("Training and validating course search discovery models")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize the system
    system = IntegratedCourseDiscoverySystem()
    
    # Run training validation
    results = await system.run_training_validation(max_universities=5)
    
    # Analyze results
    analysis = system.analyze_results(results)
    
    # Print analysis
    system.print_analysis(analysis)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'results/validation_results/training_validation_results_{timestamp}.json'
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        import json
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis,
            'detailed_results': [
                {
                    'university_name': r.university_name,
                    'discovered_url': r.discovered_url,
                    'discovery_success': r.discovery_success,
                    'discovery_confidence': r.discovery_confidence,
                    'validation_success': r.validation_success,
                    'validation_confidence': r.validation_confidence,
                    'test_courses_found': r.test_courses_found,
                    'total_test_courses': r.total_test_courses,
                    'error_message': r.error_message,
                    'discovery_time': r.discovery_time,
                    'validation_time': r.validation_time
                }
                for r in results
            ]
        }, f, indent=2)
    
    print(f"\n Results saved to: {results_file}")
    print(f"\n Training validation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main()) 