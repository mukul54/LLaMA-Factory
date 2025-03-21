#!/bin/bash
# Launcher script for the Healthcare Activity Dataset Creator

echo "Launching Healthcare Activity Dataset Creator..."
cd "$(dirname "$0")"

# Ensure the data directory exists
mkdir -p data/healthcare_activity

# Launch the Gradio app
python scripts/healthcare_dataset_creator.py
