#!/bin/bash

PYTHON_SCRIPT="aux_scripts/clear_cache.py"

while true
do
    echo "Cleaning cache at $(date)"
    python3 "$PYTHON_SCRIPT"

    # Wait 10 minutes (600 seconds)
    sleep 600
done