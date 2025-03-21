#!/bin/bash
# Script to run evaluation on a fine-tuned model

# Set the GPU device (change as needed)
export CUDA_VISIBLE_DEVICES=0

# Path to the test data
TEST_DATA_PATH="/home/mukul.ranjan/projects/video-vlm/LLaMA-Factory/data/healthcare_activity/activity_dataset_enhanced_test.json"

# Check if model path was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_model_directory> [--no-lora]"
  echo "Example: $0 /home/mukul.ranjan/projects/video-vlm/LLaMA-Factory/outputs/qwen25vl_healthcare_enhanced"
  exit 1
fi

MODEL_PATH="$1"
LORA_FLAG="--lora"

# Check if --no-lora flag was provided
if [ "$2" == "--no-lora" ]; then
  LORA_FLAG=""
fi

echo "Evaluating model at: $MODEL_PATH"
echo "Using test data: $TEST_DATA_PATH"
echo "LoRA model: ${LORA_FLAG:+Yes}"

# Run the evaluation script
python /home/mukul.ranjan/projects/video-vlm/LLaMA-Factory/scripts/healthcare_activity/evaluate_finetuned_model.py \
  --model_path "$MODEL_PATH" \
  --test_data "$TEST_DATA_PATH" \
  $LORA_FLAG
