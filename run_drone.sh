#!/bin/bash

# Air65 Drone Control Launcher Script
# Activates conda environment and launches the drone control GUI

echo "================================================"
echo "Air65 Drone Control - MSP Interface"
echo "================================================"
echo ""
echo "Checking for Anaconda environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✓ Conda found. Activating betafpv-ultrathink environment..."

    # Source conda.sh to enable conda command in script
    source "$(conda info --base)/etc/profile.d/conda.sh"

    # Activate environment
    conda activate betafpv-ultrathink

    echo ""
    echo "Launching Drone Control GUI..."
    echo ""

    # Run the application
    python3 drone_control.py
else
    echo "✗ Conda not found. Running with system Python..."
    echo ""

    # Run with system Python
    python3 drone_control.py
fi
