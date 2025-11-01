#!/usr/bin/env python3
"""
Sim-to-Real Transfer Script
Fine-tune sim-trained policy on real hardware
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, required=True, help="Sim-trained checkpoint")
    parser.add_argument("--port", type=str, default="/dev/cu.usbmodem*", help="Serial port")
    args = parser.parse_args()

    print("=" * 60)
    print("SIM-TO-REAL TRANSFER")
    print("=" * 60)
    print("\nSAFETY WARNINGS:")
    print("  1. Remove propellers for initial testing!")
    print("  2. Have emergency stop ready")
    print("  3. Start with very conservative learning rate")
    print("  4. Monitor battery voltage")
    print("\nPress Ctrl+C to stop at any time.")
    print("=" * 60)

    input("\nPress Enter to continue (or Ctrl+C to cancel)...")

    # TODO: Implement fine-tuning on real hardware
    # This is ADVANCED - requires motion capture or similar for position feedback
    print("\nTODO: This requires external position tracking (Vicon, OptiTrack, etc.)")
    print("For safety, start with simulation training first.")


if __name__ == "__main__":
    main()
