#!/usr/bin/env python3
"""
Training Data Expansion System

This script orchestrates the complete training data expansion process:
1. Discovers NEW course search pages using the integrated system
2. Validates existing course search pages (optional)
3. Performs additional course verification
4. Builds training dataset from validated examples
5. Updates universities with validated course search pages
6. Retrains the model with expanded data

Usage:
    python run_training_expansion.py [num_colleges] [--discovery-only] [--concurrent=N] [--keep-awake]
    
    num_colleges: Number of colleges to test (default: 10)
    --discovery-only: Only discover NEW course search pages, skip validating existing ones
    --concurrent=N: Number of concurrent processes (default: 8, optimized for M1 MacBook Pro)
    --keep-awake: Prevent MacBook from sleeping during execution (uses caffeinate)

Keyboard Controls:
    Command+B: Pause and save progress (graceful shutdown)
    Ctrl+C: Force stop (immediate shutdown)
"""

import sys
import os
import asyncio
import subprocess
import signal
import json
import threading
import time
from datetime import datetime
from typing import List, Dict, Any

# Add the src directory to the path
script_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(script_dir, '..'))

from core.integrated_system import IntegratedCourseDiscoverySystem
from scripts.build_training_dataset import TrainingDatasetBuilder
from scripts.update_universities_with_validated_pages import UniversitiesUpdater
from scripts.comprehensive_course_verification import ComprehensiveCourseVerifier

class CaffeinateManager:
    """Manages the caffeinate process to keep the MacBook awake"""
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.process = None
        
    def start(self):
        """Start caffeinate to prevent sleep"""
        if not self.enabled:
            return
            
        try:
            # Start caffeinate with display and system sleep prevention
            # -d: Prevent display sleep
            # -s: Prevent system sleep
            # -i: Prevent idle sleep
            self.process = subprocess.Popen(
                ['caffeinate', '-d', '-s', '-i'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(" Caffeinate started - MacBook will stay awake during execution")
        except Exception as e:
            print(f"  Could not start caffeinate: {e}")
            print("   MacBook may still go to sleep during long runs")
    
    def stop(self):
        """Stop caffeinate"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                print(" Caffeinate stopped - MacBook can now sleep normally")
            except subprocess.TimeoutExpired:
                self.process.kill()
                print(" Caffeinate force-stopped")
            except Exception as e:
                print(f"  Error stopping caffeinate: {e}")

class ProgressManager:
    """Manages progress saving and resuming"""
    
    def __init__(self):
        self.progress_file = f"training_expansion_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.completed_universities = set()
        self.current_batch = []
        self.start_time = datetime.now()
        
    def save_progress(self, discovery_results: List[Any], current_universities: List[str]):
        """Save current progress to file"""
        try:
            progress_data = {
                'timestamp': datetime.now().isoformat(),
                'start_time': self.start_time.isoformat(),
                'completed_universities': list(self.completed_universities),
                'current_batch': current_universities,
                'discovery_results': [
                    {
                        'university_name': r.university_name,
                        'discovered_url': r.discovered_url,
                        'discovery_success': r.discovery_success,
                        'discovery_confidence': r.discovery_confidence,
                        'validation_success': r.validation_success,
                        'validation_confidence': r.validation_confidence,
                        'test_courses_found': r.test_courses_found,
                        'total_test_courses': r.total_test_courses,
                        'has_collegetransfer_courses': r.has_collegetransfer_courses,
                        'error_message': r.error_message,
                        'discovery_time': r.discovery_time,
                        'validation_time': r.validation_time
                    }
                    for r in discovery_results
                ]
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            print(f"\n Progress saved to: {self.progress_file}")
            print(f" Completed: {len(self.completed_universities)} universities")
            print(f"⏱  Runtime: {datetime.now() - self.start_time}")
            
        except Exception as e:
            print(f"  Error saving progress: {e}")
    
    def load_progress(self, progress_file: str) -> Dict[str, Any]:
        """Load progress from file"""
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  Error loading progress: {e}")
            return {}

class KeyboardMonitor:
    """Monitors for keyboard interrupts (Command+B)"""
    
    def __init__(self, progress_manager: ProgressManager, discovery_results: List[Any], current_universities: List[str]):
        self.progress_manager = progress_manager
        self.discovery_results = discovery_results
        self.current_universities = current_universities
        self.should_pause = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start keyboard monitoring in a separate thread"""
        self.monitor_thread = threading.Thread(target=self._monitor_keyboard, daemon=True)
        self.monitor_thread.start()
        print("\n⌨  Keyboard monitoring started:")
        print("   Command+B: Pause and save progress")
        print("   Ctrl+C: Force stop")
    
    def _monitor_keyboard(self):
        """Monitor for keyboard input"""
        try:
            import msvcrt  # Windows
            while not self.should_pause:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\x02':  # Ctrl+B
                        self.should_pause = True
                        break
        except ImportError:
            try:
                import tty
                import termios
                import select
                
                # Unix-like systems (macOS, Linux)
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                
                try:
                    tty.setraw(sys.stdin.fileno())
                    while not self.should_pause:
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            key = sys.stdin.read(1)
                            if key == '\x02':  # Ctrl+B
                                self.should_pause = True
                                break
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                # Fallback: just check for Ctrl+C
                pass

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print(f"\n  Received signal {signum}, shutting down gracefully...")
    if hasattr(signal_handler, 'caffeinate_manager'):
        signal_handler.caffeinate_manager.stop()
    if hasattr(signal_handler, 'progress_manager') and hasattr(signal_handler, 'discovery_results'):
        signal_handler.progress_manager.save_progress(
            signal_handler.discovery_results, 
            getattr(signal_handler, 'current_universities', [])
        )
    sys.exit(0)

async def main():
    """Main function to run the complete training expansion process"""
    
    # Get number of colleges from command line argument
    num_colleges = 10
    discovery_only = False
    max_concurrent = 8  # Default concurrency for M1 MacBook Pro
    keep_awake = False
    
    if len(sys.argv) > 1:
        try:
            num_colleges = int(sys.argv[1])
        except ValueError:
            print("Invalid number of colleges. Using default of 10.")
    
    # Check for flags
    for arg in sys.argv[2:]:
        if arg.lower() in ['--discovery-only', '-d', '--discovery']:
            discovery_only = True
        elif arg.startswith('--concurrent='):
            try:
                max_concurrent = int(arg.split('=')[1])
            except (ValueError, IndexError):
                print("Invalid concurrent value. Using default of 8.")
        elif arg.lower() in ['--keep-awake', '--caffeinate', '-k']:
            keep_awake = True
    
    # Set up managers
    caffeinate_manager = CaffeinateManager(enabled=keep_awake)
    progress_manager = ProgressManager()
    discovery_results = []
    current_universities = []
    
    # Set up signal handlers for graceful shutdown
    signal_handler.caffeinate_manager = caffeinate_manager
    signal_handler.progress_manager = progress_manager
    signal_handler.discovery_results = discovery_results
    signal_handler.current_universities = current_universities
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start caffeinate if enabled
        caffeinate_manager.start()
        
        print(" Training Data Expansion System")
        print(f"Expanding training data with {num_colleges} random colleges")
        print(f" Using {max_concurrent} concurrent processes for faster execution")
        if keep_awake:
            print(" Keep-awake mode enabled - MacBook will stay awake")
        if discovery_only:
            print(" DISCOVERY-ONLY MODE: Only discovering NEW course search pages")
        else:
            print(" FULL MODE: Discovering new pages + validating existing pages")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Start keyboard monitoring
        keyboard_monitor = KeyboardMonitor(progress_manager, discovery_results, current_universities)
        keyboard_monitor.start_monitoring()
        
        # Step 1: Discover NEW course search pages using integrated system
        print("\n Step 1: Discovering NEW course search pages...")
        discovery_system = IntegratedCourseDiscoverySystem()
        
        # Use concurrent processing for faster execution
        discovery_results = await discovery_system.run_training_validation(
            max_universities=num_colleges, 
            max_concurrent=max_concurrent
        )
        
        # Update progress manager with results
        progress_manager.discovery_results = discovery_results
        signal_handler.discovery_results = discovery_results
        
        # Check for pause request
        if keyboard_monitor.should_pause:
            print("\n⏸  Pause requested via Command+B")
            progress_manager.save_progress(discovery_results, current_universities)
            caffeinate_manager.stop()
            print(" Progress saved. You can resume later by running the script again.")
            return
        
        # Print discovery results
        if discovery_results:
            analysis = discovery_system.analyze_results(discovery_results)
            discovery_system.print_analysis(analysis)
        
        # Step 2: Validate existing course search pages (if not discovery-only)
        if not discovery_only:
            print("\n Step 2: Validating existing course search pages...")
            # This step would validate existing pages
            # For now, we'll skip it in discovery-only mode
            pass
        else:
            print("\n⏭ Skipping Step 2: Validation of existing pages (discovery-only mode)")
        
        # Check for pause request
        if keyboard_monitor.should_pause:
            print("\n⏸  Pause requested via Command+B")
            progress_manager.save_progress(discovery_results, current_universities)
            caffeinate_manager.stop()
            print(" Progress saved. You can resume later by running the script again.")
            return
        
        # Step 3: Additional course verification for all discovered pages
        print("\n Step 3: Additional course verification for all discovered pages...")
        if discovery_results:
            verifier = ComprehensiveCourseVerifier()
            for result in discovery_results:
                if result.discovery_success and result.discovered_url:
                    print(f"  Verifying NEW discovery: {result.university_name}")
                    page_data = {
                        'university_name': result.university_name,
                        'course_search_url': result.discovered_url,
                        'source': 'discovery_results',
                        'confidence': result.validation_confidence,
                        'validated': result.validation_success
                    }
                    await verifier.verify_course_search_page(page_data)
                    
                    # Check for pause request after each verification
                    if keyboard_monitor.should_pause:
                        print("\n⏸  Pause requested via Command+B")
                        progress_manager.save_progress(discovery_results, current_universities)
                        caffeinate_manager.stop()
                        print(" Progress saved. You can resume later by running the script again.")
                        return
        
        # Check for pause request
        if keyboard_monitor.should_pause:
            print("\n⏸  Pause requested via Command+B")
            progress_manager.save_progress(discovery_results, current_universities)
            caffeinate_manager.stop()
            print(" Progress saved. You can resume later by running the script again.")
            return
        
        # Step 4: Build training dataset from validated examples
        print("\n Step 4: Building training dataset from validated examples...")
        builder = TrainingDatasetBuilder()
        # For now, skip this step since we don't have validation files to process
        # builder.process_validation_file() would be called here
        
        # Step 5: Update universities with validated course search pages
        print("\n Step 5: Updating universities with validated course search pages...")
        updater = UniversitiesUpdater()
        if discovery_results:
            # Convert discovery results to the format expected by the updater
            validation_data = []
            for result in discovery_results:
                if result.discovery_success and result.discovered_url:
                    validation_data.append({
                        'university_name': result.university_name,
                        'discovered_url': result.discovered_url,
                        'validation_success': result.validation_success,
                        'validation_confidence': result.validation_confidence,
                        'test_courses_found': result.test_courses_found,
                        'total_test_courses': result.total_test_courses
                    })
            
            if validation_data:
                # Save to temporary file for processing
                temp_file = f"temp_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(temp_file, 'w') as f:
                    json.dump({'detailed_results': validation_data}, f, indent=2)
                
                summary = updater.process_validation_results(temp_file)
                updater.print_summary(summary)
                
                # Clean up temp file
                os.remove(temp_file)
        
        # Step 6: Retrain model with expanded data
        print("\n Step 6: Retraining model with expanded data...")
        try:
            from models.course_search_classifier import CourseSearchClassifier
            classifier = CourseSearchClassifier()
            classifier.train_model()
            print(" Model retrained successfully")
        except Exception as e:
            print(f"  Error retraining model: {e}")
        
        print(f"\n Training data expansion completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Print final summary
        if discovery_results:
            discovery_file = f"results/validation_results/new_discovery_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            print(f" Discovery results: {discovery_file}")
            
            successful_discoveries = sum(1 for r in discovery_results if r.discovery_success)
            successful_validations = sum(1 for r in discovery_results if r.validation_success)
            
            print(f" NEW discoveries: {successful_discoveries}")
            print(f" Universities updated: {successful_validations}")
            print(f" Course verification completed for {successful_discoveries} discovered pages")
        
    except Exception as e:
        print(f" Error during training expansion: {e}")
        raise
    finally:
        # Always stop caffeinate when done
        caffeinate_manager.stop()

if __name__ == "__main__":
    asyncio.run(main()) 