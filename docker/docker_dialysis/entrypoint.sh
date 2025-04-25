#!/bin/bash

DATA_DIR="/app/data"
MODEL_DIR="/app/models"
OUTPUT_DIR="/app/outputs"

usage() {
    echo "Usage: $0 <command> [options]"
    echo "Commands:"
    echo "  prepare [--data_dir path] [--output_file path] [--train_ratio ratio]"
    echo "  dataweb [--share]"
    echo "  webui [--share]"
    echo "  train [--model path] [--dataset path] [--output path] [additional args...]"
    echo "  evaluate [--model path] [--test_data path]"
    exit 1
}

CMD=$1
shift

case "$CMD" in
    "prepare")
        DATA_PATH="$DATA_DIR"
        OUTPUT_FILE="$DATA_DIR/activity_dataset.json"
        TRAIN_RATIO=0.8
        
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --data_dir) DATA_PATH="$2"; shift 2 ;;
                --output_file) OUTPUT_FILE="$2"; shift 2 ;;
                --train_ratio) TRAIN_RATIO="$2"; shift 2 ;;
                *) echo "Unknown parameter: $1"; usage ;;
            esac
        done
        
        python3 scripts/healthcare_activity/prepare_dataset_enhanced.py \
            --data_dir "$DATA_PATH" \
            --output_file "$OUTPUT_FILE" \
            --train_ratio "$TRAIN_RATIO"
        ;;
    
    "dataweb")
        # Optional parameters
        SHARE_FLAG=""
        
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --share) SHARE_FLAG="--share"; shift ;;
                *) echo "Unknown parameter: $1"; usage ;;
            esac
        done
        
        cd /app/LLaMA-Factory/scripts
        python healthcare_activity/healthcare_dataset_creator.py $SHARE_FLAG
        ;;
        
    "webui")
        if [ "$1" = "--share" ] || [ "$GRADIO_SHARE" = "1" ]; then
            export GRADIO_SHARE=1
            llamafactory-cli webui
        else
            llamafactory-cli webui
        fi
        ;;
        
    "train")
        MODEL_PATH=""
        DATASET_PATH=""
        OUTPUT_PATH="$OUTPUT_DIR"
        EXTRA_ARGS=""
        
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --model) MODEL_PATH="$2"; shift 2 ;;
                --dataset) DATASET_PATH="$2"; shift 2 ;;
                --output) OUTPUT_PATH="$2"; shift 2 ;;
                *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
            esac
        done
        
        python3 scripts/healthcare_activity/train.py \
            --model_name_or_path "$MODEL_PATH" \
            --dataset_dir "$DATASET_PATH" \
            --output_dir "$OUTPUT_PATH" \
            $EXTRA_ARGS
        ;;
        
    "evaluate")
        MODEL_PATH="$OUTPUT_DIR"
        TEST_DATA="$DATA_DIR/activity_dataset_test.json"
        
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --model) MODEL_PATH="$2"; shift 2 ;;
                --test_data) TEST_DATA="$2"; shift 2 ;;
                *) echo "Unknown parameter: $1"; usage ;;
            esac
        done
        
        python3 scripts/healthcare_activity/evaluate_finetuned_model.py \
            --model_path "$MODEL_PATH" \
            --test_data "$TEST_DATA"
        ;;
        
    *)
        usage
        ;;
esac