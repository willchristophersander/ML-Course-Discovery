#!/usr/bin/env python3
"""
Training Data Consolidation Script

This script consolidates all scattered training data files into one comprehensive
training dataset. It merges all enhanced_training_data_*.json files and removes
duplicates while preserving the largest/most complete examples.
"""

import os
import json
import glob
from datetime import datetime
from typing import Dict, List, Set
import sys

# Add the src directory to the path
script_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(script_dir, '..'))

def load_training_file(file_path: str) -> Dict:
    """Load a training data file and return its contents"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def consolidate_training_data():
    """Consolidate all training data files into one comprehensive dataset"""
    
    # Find all training data files
    data_dir = os.path.join(script_dir, '..', 'data')
    pattern = os.path.join(data_dir, 'enhanced_training_data_*.json')
    training_files = glob.glob(pattern)
    
    print(f"Found {len(training_files)} training data files to consolidate")
    
    # Load and merge all training data
    consolidated = {
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
    
    total_examples = 0
    duplicates_removed = 0
    
    for file_path in sorted(training_files):
        print(f"Processing: {os.path.basename(file_path)}")
        data = load_training_file(file_path)
        
        if not data:
            continue
            
        # Process each category
        for category in consolidated.keys():
            if category in data:
                for example in data[category]:
                    if 'url' in example and 'html' in example:
                        # Check if we've seen this URL before
                        if example['url'] not in seen_urls[category]:
                            consolidated[category].append(example)
                            seen_urls[category].add(example['url'])
                            total_examples += 1
                        else:
                            duplicates_removed += 1
                            # If this example has more content, replace the existing one
                            existing_idx = None
                            for i, existing in enumerate(consolidated[category]):
                                if existing['url'] == example['url']:
                                    existing_idx = i
                                    break
                            
                            if existing_idx is not None:
                                # Keep the one with more HTML content
                                if len(example['html']) > len(consolidated[category][existing_idx]['html']):
                                    consolidated[category][existing_idx] = example
    
    # Create the consolidated file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    consolidated_file = os.path.join(data_dir, f'consolidated_training_data_{timestamp}.json')
    
    # Add metadata
    consolidated['metadata'] = {
        'consolidated_at': timestamp,
        'total_files_processed': len(training_files),
        'total_examples': total_examples,
        'duplicates_removed': duplicates_removed,
        'course_search_interfaces_count': len(consolidated['course_search_interfaces']),
        'course_catalog_pages_count': len(consolidated['course_catalog_pages']),
        'general_pages_count': len(consolidated['general_pages'])
    }
    
    # Save consolidated data
    with open(consolidated_file, 'w', encoding='utf-8') as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)
    
    # Also create a current training data file (without timestamp)
    current_file = os.path.join(data_dir, 'current_training_data.json')
    with open(current_file, 'w', encoding='utf-8') as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)
    
    print(f"\n Consolidation complete!")
    print(f" Consolidated file: {os.path.basename(consolidated_file)}")
    print(f" Current file: current_training_data.json")
    print(f" Total examples: {total_examples}")
    print(f" Duplicates removed: {duplicates_removed}")
    print(f" Course search interfaces: {len(consolidated['course_search_interfaces'])}")
    print(f" Course catalog pages: {len(consolidated['course_catalog_pages'])}")
    print(f" General pages: {len(consolidated['general_pages'])}")
    
    return consolidated_file

def cleanup_old_files():
    """Optionally clean up old training data files after consolidation"""
    data_dir = os.path.join(script_dir, '..', 'data')
    pattern = os.path.join(data_dir, 'enhanced_training_data_*.json')
    training_files = glob.glob(pattern)
    
    print(f"\n Found {len(training_files)} old training files")
    
    # Keep the 5 most recent files as backup
    training_files.sort(key=os.path.getmtime, reverse=True)
    files_to_keep = training_files[:5]
    files_to_delete = training_files[5:]
    
    if files_to_delete:
        print(f"  Deleting {len(files_to_delete)} old files...")
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"   Deleted: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"   Error deleting {file_path}: {e}")
        
        print(f" Kept {len(files_to_keep)} most recent files as backup")
    else:
        print(" No old files to delete")

if __name__ == "__main__":
    print(" Training Data Consolidation")
    print("=" * 50)
    
    # Consolidate all training data
    consolidated_file = consolidate_training_data()
    
    # Ask user if they want to clean up old files
    if len(sys.argv) > 1 and sys.argv[1] == '--cleanup':
        cleanup_old_files()
    else:
        print("\n To clean up old files, run: python consolidate_training_data.py --cleanup") 