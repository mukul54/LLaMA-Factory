#!/bin/bash
# Script to run fine-tuning with the enhanced dataset for Qwen 2.5 VL 3B model

# Set the GPU device (change as needed)
export CUDA_VISIBLE_DEVICES=0

# Path to LLaMA-Factory
LLAMA_FACTORY_PATH="/home/mukul.ranjan/projects/video-vlm/LLaMA-Factory"

# Set the path to the configuration file
CONFIG_PATH="${LLAMA_FACTORY_PATH}/examples/healthcare_activity/qwen25vl_activity_enhanced_dataset.yaml"

# Set paths to scripts
SCRIPTS_PATH="${LLAMA_FACTORY_PATH}/scripts/healthcare_activity"

# Run the training script
cd ${LLAMA_FACTORY_PATH}

echo "Starting training with enhanced dataset for Qwen 2.5 VL 3B model..."
echo "Using configuration: ${CONFIG_PATH}"

python src/train_bash.py \
  --do_train \
  --config_file ${CONFIG_PATH}

echo "Training completed!"
