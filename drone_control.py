#!/usr/bin/env python3
"""
Air65 Drone Control - Main Launcher
Launch the MSP drone control GUI
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the GUI
from drone_control_gui import main

if __name__ == "__main__":
    print("=" * 50)
    print("Air65 Drone Control - MSP Interface")
    print("=" * 50)
    print()
    print("Starting drone control GUI...")
    print()
    main()
