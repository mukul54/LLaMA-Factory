# Healthcare Activity Recognition with Qwen 2.5 VL

This repository contains tools and configurations for fine-tuning the Qwen 2.5 VL model on healthcare activity recognition tasks. The fine-tuning process utilizes LLaMA-Factory as the underlying framework.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
  - [Native Installation](#native-installation)
  - [Docker Setup](#docker-setup)
- [Dataset Preparation](#dataset-preparation)
  - [Using the Gradio App](#using-the-gradio-app)
  - [Using the Command Line](#using-the-command-line)
- [Fine-tuning](#fine-tuning)
  - [Using WebUI](#using-webui)
  - [Using Command Line](#using-command-line)
- [Evaluating Fine-tuned Models](#evaluating-fine-tuned-models)
- [Troubleshooting](#troubleshooting)

## Overview

This project focuses on fine-tuning Qwen 2.5 VL vision-language models to recognize healthcare activities from video data. The activities include:

1. Arranging/checking drip bottle
2. Bed cleaning/arrangement
3. Blood pressure checking
4. Patient admission
5. Patient discharge

The fine-tuning approach includes both LoRA (Low-Rank Adaptation) for efficient training and full fine-tuning for optimal performance.

## Setup

### Native Installation

```bash
git clone https://github.com/mukul54/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

This will install LLaMA-Factory and all its dependencies. No additional dependency installation is required.

### Docker Setup

We provide Docker support for easy deployment and isolation. The Docker container includes all necessary dependencies and tools.

#### Building and Running the Docker Container

1. Navigate to the Docker directory:
   ```bash
   cd LLaMA-Factory/docker/docker_dialysis
   ```

2. Run the provided utility script:
   ```bash
   ./run.sh
   ```

3. Select the operation you want to perform:
   - `webui`: Launch the LLaMA Factory web interface
   - `dataweb`: Launch the dataset preparation web interface
   - `prepare`: Prepare dataset from command line
   - `train`: Train a model
   - `evaluate`: Evaluate a fine-tuned model
   - `build`: Rebuild the Docker image

#### Docker Usage Notes

- The container uses NVIDIA GPU acceleration and requires NVIDIA Container Toolkit
- Data persistence is handled through volume mounts to `../../data`, `../../models`, and `../../outputs` relative to the Docker directory
- The web interfaces are exposed on port 7860 (or automatically selected alternative if 7860 is in use)

#### Manual Docker Commands

If you prefer to run Docker commands directly:

```bash
# Build the image
podman build --format docker --cgroup-manager=cgroupfs -t llama-factory-health .

# Run the WebUI
podman run --rm --cgroup-manager=cgroupfs --gpus all -p 7860:7860 \
  -v "$(pwd)/../../data:/app/data" \
  -v "$(pwd)/../../models:/app/models" \
  -v "$(pwd)/../../outputs:/app/outputs" \
  -e GRADIO_SHARE=1 \
  --entrypoint llamafactory-cli llama-factory-health webui

# Run the dataset preparation web interface
podman run --rm --cgroup-manager=cgroupfs --gpus all -p 7860:7860 \
  -v "$(pwd)/../../data:/app/data" \
  -v "$(pwd)/../../models:/app/models" \
  -v "$(pwd)/../../outputs:/app/outputs" \
  -e GRADIO_SHARE=1 \
  llama-factory-health dataweb
```

## Dataset Preparation

### Using the Gradio App

We provide a Gradio-based web application for easy dataset preparation:

1. Launch the Gradio app:
   ```bash
   # If using native installation
   cd scripts
   python healthcare_activity/healthcare_dataset_creator.py
   
   # If using Docker
   ./run.sh
   # Select option 2 (dataweb)
   ```

2. In the web interface:
   - Enter the path to your video directory
   - Configure the output path and settings
   - Choose between simple numeric responses or detailed explanatory responses
   - Click "Create Dataset" to generate the training and test datasets

### Using the Command Line

Alternatively, you can use the command line script:

```bash
# If using native installation
cd scripts
python healthcare_activity/prepare_dataset_enhanced.py \
  --data_dir "/path/to/video/clips/" \
  --output_file "/path/to/output/activity_dataset.json" \
  --train_ratio 0.8 \
  --max_per_class 400

# If using Docker
./run.sh
# Select option 3 (prepare)
# Enter the requested parameters when prompted
```

Options:
- `--data_dir`: Directory containing activity video folders
- `--output_file`: Output JSON file path
- `--train_ratio`: Ratio of train:test split (default: 0.8)
- `--max_per_class`: Maximum samples per class (default: None, use all)
- `--simple_responses`: Use simple numeric responses instead of detailed responses

## Fine-tuning

### Using WebUI

1. Launch the LLaMA-Factory WebUI:
   ```bash
   # If using native installation
   llamafactory-cli webui

   # With a public link
   GRADIO_SHARE=1 llamafactory-cli webui
   
   # If using Docker
   ./run.sh
   # Select option 1 (webui)
   ```

2. In the WebUI:
   - Go to the "Dataset" tab and select "healthcare_activity" dataset
   - Go to the "Model" tab and select "Qwen/Qwen2.5-VL-3B-Instruct" model
   - Select "LoRA" or "Full" fine-tuning method
   - Configure training parameters:
     - Freeze vision tower: True
     - Batch size: 1
     - Gradient accumulation steps: 4 or higher
     - Learning rate: 5e-5
     - Training epochs: 3
   - Click "Start" to begin training

#### WebUI Usage Tips

1. **Using LoRA Fine-tuning**:
   - When using LoRA fine-tuning, make sure to leave the **Checkpoint Path** field empty when training a new adapter.
   - The error "adapter_config.json not found" usually occurs when the WebUI tries to load an adapter from the base model path.

2. **Setting up for Multimodal Models (like Qwen-VL)**:
   - Select the appropriate model template from the dropdown menu (e.g., `qwen2_vl` for Qwen2.5-VL)
   - Make sure your dataset includes absolute paths to images or videos (prepare_dataset.py handles this conversion)

3. **Common WebUI Issues**:
   - If you encounter "ValueError: Can't load adapter config.json" with LoRA, ensure your checkpoint path is empty
   - For video or image models, ensure that your dataset JSON files contain **full absolute paths** to media files (WebUI does not have a media path field)
   - Use the "Preview" button to verify your configuration before starting training

#### Healthcare Activity Dataset

This fork includes a custom dataset for healthcare activity recognition using video data:

1. **Preparing the Dataset**:
   ```bash
   # If using native installation
   cd LLaMA-Factory/scripts
   python healthcare_activity/prepare_dataset_enhanced.py
   
   # To prepare dataset with specific parameters
   python healthcare_activity/prepare_dataset_enhanced.py \
     --data_dir "/path/to/video/clips" \
     --output_file "/path/to/output/activity_dataset.json" \
     --train_ratio 0.8 \
     --max_per_class 100
     
   # If using Docker
   ./run.sh
   # Select option 3 (prepare)
   ```
   
   **Command-line Arguments**:
   - `--data_dir`: Directory containing activity video folders (default: "/l/users/mukul.ranjan/video_data/activity_clips/activity_clips/")
   - `--output_file`: Output JSON file path (default: "/home/mukul.ranjan/projects/video-vlm/LLaMA-Factory/data/healthcare_activity/activity_dataset.json")
   - `--train_ratio`: Ratio of train:test split (default: 0.8)
   - `--max_per_class`: Maximum samples per class, useful for balancing the dataset (default: unlimited)
   
   This will create dataset files with the following features:
   - Converts video paths to absolute paths for WebUI compatibility
   - Creates train/test splits of the dataset (saved as "*_train.json" and "*_test.json")
   - Formats data for multimodal training with Qwen2.5-VL

2. **Evaluating Model Performance**:
   After running predictions using the WebUI, you can analyze the model's performance using the evaluation script:
   ```bash
   # If using native installation
   python scripts/healthcare_activity/evaluate_finetuned_model.py \
     --model_path "/path/to/finetuned/model" \
     --test_data "/path/to/test_dataset.json"
   
   # If using Docker
   ./run.sh
   # Select option 5 (evaluate)
   ```
   
   Example usage after WebUI evaluation:
   ```bash
   # If using native installation
   python scripts/healthcare_activity/evaluate_finetuned_model.py \
     --model_path "./outputs/qwen25vl_healthcare_enhanced" \
     --test_data "./data/healthcare_activity/activity_dataset_enhanced_test.json"
   ```
   
   The script will:
   - Calculate overall accuracy
   - Generate detailed results for each test sample
   - Create a JSON report of all predictions

### Using Command Line

1. Prepare a configuration YAML file (example in `examples/healthcare_activity/qwen25vl_activity_finetuning.yaml`)

2. Run the training script:
   ```bash
   # If using native installation
   cd LLaMA-Factory
   
   # For LoRA fine-tuning
   CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
     --do_train \
     --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
     --dataset healthcare_activity \
     --dataset_dir ./data \
     --finetuning_type lora \
     --lora_rank 16 \
     --lora_alpha 32 \
     --lora_dropout 0.05 \
     --freeze_vision_tower \
     --output_dir ./output_healthcare \
     --overwrite_output_dir \
     --max_samples 3000 \
     --per_device_train_batch_size 1 \
     --gradient_accumulation_steps 4 \
     --learning_rate 5e-5 \
     --num_train_epochs 3
     
   # If using Docker
   ./run.sh
   # Select option 4 (train)
   ```

Alternatively, you can use a configuration file:

```bash
# If using native installation
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
  --config_file examples/healthcare_activity/qwen25vl_activity_finetuning.yaml
```

## Evaluating Fine-tuned Models

To evaluate your fine-tuned model on the test set:

```bash
# If using native installation
cd LLaMA-Factory
python src/evaluate_bash.py \
  --model_name_or_path ./output_healthcare \
  --finetuning_type lora \
  --dataset healthcare_activity \
  --dataset_dir ./data \
  --split test
  
# If using Docker
./run.sh
# Select option 5 (evaluate)
```

## Troubleshooting

### Memory Issues

- **Problem**: CUDA out of memory errors
  - **Solution**: Increase gradient accumulation steps, freeze vision tower, or use a smaller model (e.g., 3B instead of 7B)

- **Problem**: "No executable batch size found"
  - **Solution**: Enable `auto_find_batch_size: true` in your configuration or command line

### CUDA Compatibility Issues

- **Problem**: CUDA version compatibility warnings
  - **Solution**: Avoid using `device_map: auto` and manually specify CUDA device with `CUDA_VISIBLE_DEVICES`

### Dataset Format Issues

- **Problem**: Poor training results with simple numeric responses
  - **Solution**: Use the enhanced dataset creator with detailed explanatory responses to improve model learning

### Docker Issues

- **Problem**: Port conflicts when starting web interfaces
  - **Solution**: The run.sh script automatically detects port conflicts and selects an alternative port

- **Problem**: Permission errors with port numbers below 1024
  - **Solution**: Always use port numbers >= 1024 in rootless container environments

- **Problem**: Unable to access mounted volumes
  - **Solution**: Ensure the data, models, and outputs directories exist at the correct relative paths

- **Problem**: Gradio interface not accessible outside container
  - **Solution**: Use the GRADIO_SHARE=1 environment variable to enable public URL access

---

For the original LLaMA-Factory documentation, please refer to [README_old.md](README_old.md).
