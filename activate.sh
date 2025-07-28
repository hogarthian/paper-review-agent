#!/bin/bash
# Simple script to activate the virtual environment
# Usage: source activate.sh

if [ -d "paper-review-env" ]; then
    source paper-review-env/bin/activate
    echo "✅ Virtual environment activated!"
    echo "📋 To deactivate, run: deactivate"
    echo "🔧 To install dependencies: pip install -r requirements.txt"
else
    echo "❌ Virtual environment not found!"
    echo "🔧 Create it with: python3 -m venv paper-review-env"
fi 