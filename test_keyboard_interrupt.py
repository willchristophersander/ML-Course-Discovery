#!/usr/bin/env python3
"""
Test script to demonstrate Command+B keyboard interrupt functionality
"""

import sys
import threading
import time
import signal
import json
from datetime import datetime

class KeyboardMonitor:
    """Monitors for keyboard interrupts (Command+B)"""
    
    def __init__(self):
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
        except Exception as e:
            print(f"  Keyboard monitoring not available: {e}")

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print(f"\n  Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

def main():
    """Main function to test keyboard interrupt"""
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(" Keyboard Interrupt Test")
    print("This script will run for 30 seconds or until you press Command+B")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Start keyboard monitoring
    keyboard_monitor = KeyboardMonitor()
    keyboard_monitor.start_monitoring()
    
    # Simulate work
    start_time = datetime.now()
    counter = 0
    
    try:
        while counter < 30:  # Run for 30 seconds
            if keyboard_monitor.should_pause:
                print(f"\n⏸  Pause requested via Command+B at {datetime.now().strftime('%H:%M:%S')}")
                
                # Save progress
                progress_data = {
                    'timestamp': datetime.now().isoformat(),
                    'start_time': start_time.isoformat(),
                    'counter': counter,
                    'runtime_seconds': (datetime.now() - start_time).total_seconds()
                }
                
                progress_file = f"test_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)
                
                print(f" Progress saved to: {progress_file}")
                print(f" Counter: {counter}/30")
                print(f"⏱  Runtime: {(datetime.now() - start_time).total_seconds():.1f} seconds")
                print(" Progress saved. You can resume later by running the script again.")
                return
            
            print(f"Working... {counter}/30", end='\r')
            counter += 1
            time.sleep(1)
        
        print(f"\n Test completed successfully at {datetime.now().strftime('%H:%M:%S')}")
        print(f"⏱  Total runtime: {(datetime.now() - start_time).total_seconds():.1f} seconds")
        
    except KeyboardInterrupt:
        print(f"\n  Interrupted by user at {datetime.now().strftime('%H:%M:%S')}")
        print(f" Final counter: {counter}/30")

if __name__ == "__main__":
    main() 