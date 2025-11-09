#!/bin/bash

# Script to run training scripts in all model directories
# Created for MaterialVision project

echo "Starting training for all models..."
echo "=================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Array of directories that contain training.py
TRAINING_DIRS=("Apple_MobileCLIP" "CLIPP_allenai" "CLIPP_bert")

# Function to run training in a directory
run_training() {
    local dir_name=$1
    local script_name=$2
    
    echo ""
    echo "Running training in $dir_name..."
    echo "-----------------------------------"
    
    cd "$SCRIPT_DIR/$dir_name"
    
    if [ -f "$script_name" ]; then
        echo "Executing: python $script_name"
        python "$script_name"
        
        if [ $? -eq 0 ]; then
            echo "✅ Training completed successfully in $dir_name"
        else
            echo "❌ Training failed in $dir_name"
        fi
    else
        echo "⚠️  Warning: $script_name not found in $dir_name"
    fi
    
    echo ""
}

# Run training.py in each directory
for dir in "${TRAINING_DIRS[@]}"; do
    if [ -d "$SCRIPT_DIR/$dir" ]; then
        run_training "$dir" "training.py"
    else
        echo "⚠️  Warning: Directory $dir not found"
    fi
done



echo ""
echo "=================================="
echo "All training scripts have been executed!"
echo "Check the output above for any errors."