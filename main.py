#!/usr/bin/env python3
"""
BETAFPV Configurator - UltraThink Edition
Main launcher script
"""
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the GUI
from betafpv_gui import main

if __name__ == "__main__":
    main()
