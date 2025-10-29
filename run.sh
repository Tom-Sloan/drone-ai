#!/bin/bash
# Quick start script for BETAFPV Configurator - UltraThink Edition

echo "================================================"
echo "BETAFPV Configurator - UltraThink Edition"
echo "================================================"
echo ""

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Checking for Anaconda environment..."
    if conda env list | grep -q "betafpv-ultrathink"; then
        echo "✓ Environment found. Activating..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate betafpv-ultrathink
    else
        echo "Creating Anaconda environment..."
        conda env create -f environment.yml
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate betafpv-ultrathink
    fi
else
    echo "Anaconda not found. Using system Python..."
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "Launching BETAFPV Configurator..."
echo ""
python main.py
