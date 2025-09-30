#!/usr/bin/env python3
"""
Cleanup Unused Files

This script identifies and deletes unused files while preserving core functionality.
"""

import os
import shutil

def cleanup_unused_files():
    """
    Clean up unused files while preserving core functionality.
    """
    print(" CLEANING UP UNUSED FILES")
    print("=" * 50)
    
    # Files to KEEP (core functionality)
    keep_files = {
        # Core Pipeline Files
        'ml_course_catalog_finder.py',
        'run_comprehensive_course_search.py', 
        'select_us_universities_final.py',
        'course_page_classifier.py',
        'course_search_classifier.py',
        'enhanced_course_search_classifier_v2.py',
        'ml_course_catalog_finder_enhanced.py',
        'update_timeout_settings.py',
        'cleanup_unused_files.py',  # This script itself
        
        # Data Files
        'positive_course_catalogs.json',
        'negative_examples.json', 
        'trained_model.pkl',
        'course_page_classifier.pkl',
        'course_search_classifier.pkl',
        'world_universities_and_domains.json',
        'world_universities_and_domains_and_courses.json',
        'search_progress.json',
        
        # ScratchPad (preserved as requested)
        'ScratchPad.txt',
        
        # Virtual Environment
        'venv/',
        '__pycache__/',
        
        # Current Status Files
        'CURRENT_STATUS_SUMMARY.md',
        'comprehensive_course_search_results.json',
        'comprehensive_course_search_training_data.json',
        'enhanced_course_search_training_data.json'
    }
    
    # Files to DELETE (unused/obsolete)
    delete_files = [
        # Debug/Test Files (no longer needed)
        'debug_yale_links.py',
        'debug_yale_detailed.py', 
        'test_harvard_ml.py',
        'test_harvard_complex.py',
        'test_harvard_pll.py',
        'test_multiple_universities.py',
        'test_uvm_validation.py',
        'test_course_search_classifier.py',
        'test_mccc_validation.py',
        'test_enhanced_pipeline.py',
        
        # Obsolete Classifiers
        'course_search_page_classifier.py',  # Replaced by enhanced version
        'course_catalog_finder.py',  # Replaced by ML version
        
        # Training/Update Scripts (one-time use)
        'update_training_with_mccc.py',
        'update_training_with_comprehensive_data.py',
        'update_training_from_results.py',
        'update_world_universities_with_catalogs.py',
        'train_and_search.py',
        'manual_training_example.py',
        'manage_training_data.py',
        
        # Integration/Demo Scripts (one-time use)
        'integrate_enhanced_classifier.py',
        'integrate_into_existing_pipeline.py',
        'integrate_course_search_validator.py',
        'simple_integration_demo.py',
        'example_usage.py',
        'demo_course_search_classifier.py',
        
        # Analysis Scripts (one-time use)
        'analyze_mccc_page.py',
        'validation_summary.py',
        
        # Documentation Files (redundant)
        'COURSE_SEARCH_CLASSIFIER_README.md',
        'COURSE_SEARCH_VALIDATION_SOLUTION.md',
        'MODEL_PERSISTENCE_SUMMARY.md',
        'ML_COURSE_VALIDATION_SUMMARY.md',
        'UVM_VALIDATION_RESULTS.md',
        'IMPLEMENTATION_COMPLETE_SUMMARY.md',
        
        # Result Files (temporary)
        'enhanced_validation_results.json',
        'enhanced_pipeline_results.json',
        'validation_integration_results.json',
        'crawler_results.json',
        'course_search_validation_results.json',
        'uvm_validation_results.json',
        'validation_summary_results.json',
        'validation_test_results.json',
        'yale_ml_result.json',
        'uvm_ml_result.json',
        'yale_debug_results.json',
        'uvm_catalog_results.json',
        'multiple_universities_results.json',
        
        # Obsolete Data Files
        'world_universities_and_domains_backup.json',
        'world_universities_and_domains_and_courses.json',
        
        # Obsolete Models
        'course_search_page_classifier.pkl',  # Replaced by enhanced version
        
        # Obsolete Scripts
        'run_course_catalog_search.py'  # Replaced by comprehensive version
    ]
    
    print(" FILES TO DELETE:")
    print("-" * 30)
    
    deleted_count = 0
    not_found_count = 0
    
    for file_path in delete_files:
        if os.path.exists(file_path):
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
                print(f"  Deleted: {file_path}")
                deleted_count += 1
            except Exception as e:
                print(f" Error deleting {file_path}: {e}")
        else:
            print(f"  Not found: {file_path}")
            not_found_count += 1
    
    print("\n" + "=" * 50)
    print("CLEANUP SUMMARY")
    print("=" * 50)
    print(f" Deleted {deleted_count} files")
    print(f"  {not_found_count} files not found")
    print()
    
    # Show remaining files
    remaining_files = []
    for item in os.listdir('.'):
        if item not in keep_files and not item.startswith('.'):
            remaining_files.append(item)
    
    if remaining_files:
        print(" REMAINING FILES:")
        print("-" * 20)
        for file in sorted(remaining_files):
            print(f"   • {file}")
    else:
        print(" All unused files cleaned up!")
    
    print()
    print(" CORE FUNCTIONALITY PRESERVED:")
    print("   • Main pipeline (ml_course_catalog_finder.py)")
    print("   • Enhanced classifier (enhanced_course_search_classifier_v2.py)")
    print("   • Comprehensive search (run_comprehensive_course_search.py)")
    print("   • Training data and models")
    print("   • ScratchPad.txt (as requested)")
    print()
    print(" The codebase is now clean and focused!")

def show_cleanup_plan():
    """
    Show what will be cleaned up before doing it.
    """
    print(" CLEANUP PLAN")
    print("=" * 50)
    print("This will remove unused files while preserving core functionality.")
    print()
    print("  FILES TO DELETE:")
    print("   • Debug/test scripts (no longer needed)")
    print("   • Obsolete classifiers (replaced by enhanced versions)")
    print("   • One-time training/update scripts")
    print("   • Integration/demo scripts (already used)")
    print("   • Analysis scripts (one-time use)")
    print("   • Redundant documentation")
    print("   • Temporary result files")
    print("   • Obsolete data files")
    print()
    print(" FILES TO KEEP:")
    print("   • Core pipeline files")
    print("   • Enhanced classifier")
    print("   • Training data and models")
    print("   • ScratchPad.txt (as requested)")
    print("   • Current status files")
    print()
    print(" BENEFITS:")
    print("   • Cleaner codebase")
    print("   • Easier navigation")
    print("   • Reduced confusion")
    print("   • Focused on core functionality")

def main():
    """Run the cleanup process."""
    
    print(" CODEBASE CLEANUP")
    print("=" * 50)
    print("This script will remove unused files while preserving core functionality.")
    print()
    
    # Show the plan first
    show_cleanup_plan()
    
    print("\n" + "=" * 50)
    response = input("Proceed with cleanup? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        print("\n Starting cleanup...")
        cleanup_unused_files()
    else:
        print(" Cleanup cancelled.")

if __name__ == "__main__":
    main()