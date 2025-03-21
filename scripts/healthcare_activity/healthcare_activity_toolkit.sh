#!/bin/bash
# Healthcare Activity Recognition Toolkit
# A master script to access all healthcare activity recognition tools

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LLAMA_FACTORY_PATH="/home/mukul.ranjan/projects/video-vlm/LLaMA-Factory"
CONFIG_DIR="${LLAMA_FACTORY_PATH}/examples/healthcare_activity"
DATA_DIR="${LLAMA_FACTORY_PATH}/data/healthcare_activity"

# Make sure the data directory exists
mkdir -p "${DATA_DIR}"

# Display menu
show_menu() {
    clear
    echo "==========================================================="
    echo "  Healthcare Activity Recognition Toolkit for Qwen 2.5 VL"
    echo "==========================================================="
    echo ""
    echo "1. Create Dataset (Gradio UI)"
    echo "2. Create Dataset (Command Line)"
    echo "3. Run LoRA Fine-tuning (Enhanced Dataset)"
    echo "4. Run Full Fine-tuning (Enhanced Dataset)"
    echo "5. Evaluate Fine-tuned Model"
    echo "6. Exit"
    echo ""
    echo "Enter your choice [1-6]: "
}

# Function to launch Gradio app
launch_gradio() {
    echo "Launching Healthcare Activity Dataset Creator UI..."
    cd "${LLAMA_FACTORY_PATH}"
    python "${SCRIPT_DIR}/../healthcare_dataset_creator.py"
}

# Function to create dataset via command line
create_dataset_cli() {
    echo "Creating healthcare activity dataset via command line..."
    
    read -p "Enter video data directory [/l/users/mukul.ranjan/video_data/activity_clips/activity_clips/]: " DATA_INPUT
    DATA_INPUT=${DATA_INPUT:-/l/users/mukul.ranjan/video_data/activity_clips/activity_clips/}
    
    read -p "Enter output file name [activity_dataset_enhanced.json]: " OUTPUT_NAME
    OUTPUT_NAME=${OUTPUT_NAME:-activity_dataset_enhanced.json}
    OUTPUT_PATH="${DATA_DIR}/${OUTPUT_NAME}"
    
    read -p "Enter maximum samples per class [400]: " MAX_PER_CLASS
    MAX_PER_CLASS=${MAX_PER_CLASS:-400}
    
    read -p "Use detailed responses? (y/n) [y]: " USE_DETAILED
    USE_DETAILED=${USE_DETAILED:-y}
    
    if [[ $USE_DETAILED == "y" ]]; then
        DETAILED_FLAG=""
    else
        DETAILED_FLAG="--simple_responses"
    fi
    
    echo "Running dataset creation..."
    cd "${LLAMA_FACTORY_PATH}"
    python "${SCRIPT_DIR}/../prepare_dataset_enhanced.py" \
        --data_dir "${DATA_INPUT}" \
        --output_file "${OUTPUT_PATH}" \
        --max_per_class "${MAX_PER_CLASS}" \
        ${DETAILED_FLAG}
    
    echo "Dataset creation complete!"
    echo "Train data: ${OUTPUT_PATH%.*}_train.json"
    echo "Test data: ${OUTPUT_PATH%.*}_test.json"
}

# Function to run LoRA fine-tuning
run_lora_finetuning() {
    echo "Preparing to run LoRA fine-tuning..."
    
    read -p "Enter GPU ID to use [0]: " GPU_ID
    GPU_ID=${GPU_ID:-0}
    
    read -p "Enter output directory name [qwen25vl_healthcare_enhanced]: " OUTPUT_DIR
    OUTPUT_DIR=${OUTPUT_DIR:-qwen25vl_healthcare_enhanced}
    
    echo "Running LoRA fine-tuning on GPU ${GPU_ID}..."
    
    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    cd "${LLAMA_FACTORY_PATH}"
    
    python src/train_bash.py \
        --do_train \
        --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
        --dataset healthcare_activity \
        --dataset_name activity_dataset_enhanced \
        --dataset_dir ./data \
        --finetuning_type lora \
        --lora_rank 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --freeze_vision_tower \
        --output_dir "./outputs/${OUTPUT_DIR}" \
        --overwrite_output_dir \
        --max_samples 3000 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-5 \
        --num_train_epochs 3 \
        --fp16
    
    echo "LoRA fine-tuning complete!"
    echo "Model saved to: ${LLAMA_FACTORY_PATH}/outputs/${OUTPUT_DIR}"
}

# Function to run full fine-tuning
run_full_finetuning() {
    echo "Preparing to run full fine-tuning (requires significant GPU memory)..."
    
    read -p "Enter GPU ID to use [0]: " GPU_ID
    GPU_ID=${GPU_ID:-0}
    
    read -p "Enter output directory name [qwen25vl_healthcare_full]: " OUTPUT_DIR
    OUTPUT_DIR=${OUTPUT_DIR:-qwen25vl_healthcare_full}
    
    echo "Running full fine-tuning on GPU ${GPU_ID}..."
    
    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    cd "${LLAMA_FACTORY_PATH}"
    
    python src/train_bash.py \
        --do_train \
        --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
        --dataset healthcare_activity \
        --dataset_name activity_dataset_enhanced \
        --dataset_dir ./data \
        --finetuning_type full \
        --freeze_vision_tower \
        --device_map auto \
        --auto_find_batch_size \
        --output_dir "./outputs/${OUTPUT_DIR}" \
        --overwrite_output_dir \
        --max_samples 3000 \
        --gradient_accumulation_steps 8 \
        --max_grad_norm 0.3 \
        --per_device_train_batch_size 1 \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --fp16 \
        --low_cpu_mem_usage
    
    echo "Full fine-tuning complete!"
    echo "Model saved to: ${LLAMA_FACTORY_PATH}/outputs/${OUTPUT_DIR}"
}

# Function to evaluate model
evaluate_model() {
    echo "Preparing to evaluate fine-tuned model..."
    
    read -p "Enter path to fine-tuned model: " MODEL_PATH
    if [ ! -d "${MODEL_PATH}" ]; then
        echo "Error: Model directory does not exist!"
        return 1
    fi
    
    read -p "Is this a LoRA model? (y/n) [y]: " IS_LORA
    IS_LORA=${IS_LORA:-y}
    
    if [[ $IS_LORA == "y" ]]; then
        LORA_FLAG="--lora"
    else
        LORA_FLAG="--no-lora"
    fi
    
    read -p "Enter GPU ID to use [0]: " GPU_ID
    GPU_ID=${GPU_ID:-0}
    
    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    
    echo "Running evaluation..."
    bash "${SCRIPT_DIR}/run_evaluation.sh" "${MODEL_PATH}" ${LORA_FLAG}
}

# Main loop
while true; do
    show_menu
    read choice
    
    case $choice in
        1) launch_gradio ;;
        2) create_dataset_cli ;;
        3) run_lora_finetuning ;;
        4) run_full_finetuning ;;
        5) evaluate_model ;;
        6) echo "Exiting..."; exit 0 ;;
        *) echo "Invalid option. Press Enter to continue..."; read ;;
    esac
    
    echo ""
    echo "Press Enter to return to the menu..."
    read
done
