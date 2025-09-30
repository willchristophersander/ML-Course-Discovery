#!/usr/bin/env python3
"""
Update Timeout Settings

This script updates timeout settings across all files to prevent early timeouts.
"""

import re
import os

def update_file_timeouts(file_path, http_timeout=30, progress_timeout=60):
    """
    Update timeout settings in a file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Update HTTP request timeouts
        content = re.sub(r'timeout=10', f'timeout={http_timeout}', content)
        content = re.sub(r'timeout=5', f'timeout={http_timeout}', content)
        
        # Update progress timeout in run_comprehensive_course_search.py
        if 'run_comprehensive_course_search.py' in file_path:
            content = re.sub(r'timeout_seconds = 10', f'timeout_seconds = {progress_timeout}', content)
            content = re.sub(r'10 seconds without progress', f'{progress_timeout} seconds without progress', content)
            content = re.sub(r'Auto-restart after 10 seconds', f'Auto-restart after {progress_timeout} seconds', content)
        
        # Update signal alarm timeout
        content = re.sub(r'signal\.alarm\(10\)', f'signal.alarm({progress_timeout})', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f" Updated timeouts in {file_path}")
            return True
        else:
            print(f"ℹ  No timeout changes needed in {file_path}")
            return False
            
    except Exception as e:
        print(f" Error updating {file_path}: {e}")
        return False

def main():
    """Update timeout settings across all relevant files."""
    
    print("⏰ UPDATING TIMEOUT SETTINGS")
    print("=" * 50)
    
    # Files to update with new timeout settings
    files_to_update = [
        {
            'file': 'run_comprehensive_course_search.py',
            'http_timeout': 30,
            'progress_timeout': 60
        },
        {
            'file': 'ml_course_catalog_finder.py',
            'http_timeout': 30,
            'progress_timeout': 60
        },
        {
            'file': 'ml_course_catalog_finder_enhanced.py',
            'http_timeout': 30,
            'progress_timeout': 60
        },
        {
            'file': 'enhanced_course_search_classifier_v2.py',
            'http_timeout': 30,
            'progress_timeout': 60
        },
        {
            'file': 'course_search_page_classifier.py',
            'http_timeout': 30,
            'progress_timeout': 60
        },
        {
            'file': 'course_page_classifier.py',
            'http_timeout': 30,
            'progress_timeout': 60
        },
        {
            'file': 'yale_catalog_finder.py',
            'http_timeout': 30,
            'progress_timeout': 60
        },
        {
            'file': 'course_catalog_finder.py',
            'http_timeout': 30,
            'progress_timeout': 60
        },
        {
            'file': 'debug_yale_detailed.py',
            'http_timeout': 30,
            'progress_timeout': 60
        },
        {
            'file': 'debug_yale_links.py',
            'http_timeout': 30,
            'progress_timeout': 60
        },
        {
            'file': 'analyze_mccc_page.py',
            'http_timeout': 30,
            'progress_timeout': 60
        },
        {
            'file': 'manage_training_data.py',
            'http_timeout': 30,
            'progress_timeout': 60
        }
    ]
    
    updated_files = 0
    total_files = len(files_to_update)
    
    for file_info in files_to_update:
        file_path = file_info['file']
        http_timeout = file_info['http_timeout']
        progress_timeout = file_info['progress_timeout']
        
        if os.path.exists(file_path):
            if update_file_timeouts(file_path, http_timeout, progress_timeout):
                updated_files += 1
        else:
            print(f"  File not found: {file_path}")
    
    print("\n" + "=" * 50)
    print("TIMEOUT UPDATE SUMMARY")
    print("=" * 50)
    print(f" Updated {updated_files}/{total_files} files")
    print()
    print(" NEW TIMEOUT SETTINGS:")
    print(f"   • HTTP Request Timeout: 30 seconds (was 5-10 seconds)")
    print(f"   • Progress Timeout: 60 seconds (was 10 seconds)")
    print(f"   • Signal Alarm Timeout: 60 seconds (was 10 seconds)")
    print()
    print(" BENEFITS:")
    print("   • More time for slow-loading university websites")
    print("   • Reduced false timeouts during course catalog discovery")
    print("   • Better handling of network delays")
    print("   • Improved reliability for comprehensive searches")
    print()
    print("  NOTE: These changes will make the system more patient")
    print("   but may take longer to complete searches.")

if __name__ == "__main__":
    main() 