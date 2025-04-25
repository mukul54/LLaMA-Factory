#!/bin/bash

# Define colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default port - must be >= 1024
DEFAULT_PORT=7860
ALTERNATE_PORT=7861

# Display banner
echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}   LLaMA Factory Healthcare Tool   ${NC}"
echo -e "${GREEN}====================================${NC}"

# Check if image exists, if not build it
if ! podman image exists llama-factory-health; then
    echo -e "\n${YELLOW}Image not found. Building image...${NC}"
    podman build --format docker --cgroup-manager=cgroupfs -t llama-factory-health .
fi

# Function to check if port is available (returns 0 if available, 1 if in use)
is_port_available() {
    local PORT=$1
    
    # Try multiple methods to check if port is in use
    if command -v netstat >/dev/null 2>&1; then
        if netstat -tuln 2>/dev/null | grep -q ":$PORT "; then
            return 1  # Port is in use
        fi
    elif command -v ss >/dev/null 2>&1; then
        if ss -tuln 2>/dev/null | grep -q ":$PORT "; then
            return 1  # Port is in use
        fi
    elif command -v lsof >/dev/null 2>&1; then
        if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
            return 1  # Port is in use
        fi
    fi
    
    return 0  # Port is available
}

# Function to find an available port
find_available_port() {
    local PORT=$1
    
    # Validate port is not privileged
    if [ "$PORT" -lt 1024 ]; then
        echo -e "${RED}Error: Cannot use privileged port $PORT. Must be >= 1024.${NC}" >&2
        PORT=$DEFAULT_PORT
    fi
    
    # Check if default port is available
    if is_port_available $PORT; then
        echo $PORT
        return
    fi
    
    echo -e "${YELLOW}Port $PORT is in use. Trying alternate port...${NC}" >&2
    
    # Try alternate port
    if is_port_available $ALTERNATE_PORT; then
        echo $ALTERNATE_PORT
        return
    fi
    
    # Find a random available port above 10000
    local RANDOM_PORT=$(( (RANDOM % 10000) + 10000 ))
    echo -e "${YELLOW}Using random port $RANDOM_PORT...${NC}" >&2
    echo $RANDOM_PORT
}

# Function to ensure directories exist
ensure_dirs() {
    for DIR in "../../data" "../../models" "../../outputs"; do
        if [ ! -d "$(pwd)/$DIR" ]; then
            echo -e "${YELLOW}Creating directory: $(pwd)/$DIR${NC}"
            mkdir -p "$(pwd)/$DIR"
        fi
    done
}

# Function to display usage
usage() {
    echo -e "\nUsage: $0 <command> [options]"
    echo -e "\nCommands:"
    echo "  1. webui         - Launch the LLaMA Factory web interface"
    echo "  2. dataweb       - Launch the dataset preparation web interface"
    echo "  3. prepare       - Prepare dataset from command line"
    echo "  4. train         - Train a model"
    echo "  5. evaluate      - Evaluate a fine-tuned model"
    echo "  6. build         - Rebuild the Docker image"
    echo "  7. help          - Show this help message"
    echo "  8. exit          - Exit the script"
}

# Main menu for command selection
select_command() {
    echo -e "\n${BLUE}Select a command to run:${NC}"
    select cmd in "webui" "dataweb" "prepare" "train" "evaluate" "build" "help" "exit"; do
        case $cmd in
            webui)
                ensure_dirs
                
                # Get available port (without color codes mixed in)
                PORT=$(find_available_port $DEFAULT_PORT)
                
                echo -e "\n${GREEN}Launching WebUI on port $PORT...${NC}"
                podman run --rm --cgroup-manager=cgroupfs --gpus all -p $PORT:7860 -e GRADIO_SHARE=1 \
                    -v "$(pwd)/../../data:/app/data" \
                    -v "$(pwd)/../../models:/app/models" \
                    -v "$(pwd)/../../outputs:/app/outputs" \
                    --entrypoint llamafactory-cli llama-factory-health webui
                break
                ;;
            dataweb)
                ensure_dirs
                
                # Get available port (without color codes mixed in)
                PORT=$(find_available_port $DEFAULT_PORT)
                
                echo -e "\n${GREEN}Launching Dataset Creation WebUI on port $PORT...${NC}"
                podman run --rm --cgroup-manager=cgroupfs --gpus all -p $PORT:7860 -e GRADIO_SHARE=1 \
                    -v "$(pwd)/../../data:/app/data" \
                    -v "$(pwd)/../../models:/app/models" \
                    -v "$(pwd)/../../outputs:/app/outputs" \
                    llama-factory-health dataweb
                break
                ;;
            prepare)
                ensure_dirs
                
                echo -e "\n${BLUE}Dataset Preparation${NC}"
                read -p "Data directory path (default: /app/data): " DATA_DIR
                DATA_DIR=${DATA_DIR:-"/app/data"}
                read -p "Output file path (default: /app/data/activity_dataset.json): " OUTPUT_FILE
                OUTPUT_FILE=${OUTPUT_FILE:-"/app/data/activity_dataset.json"}
                read -p "Train ratio (default: 0.8): " TRAIN_RATIO
                TRAIN_RATIO=${TRAIN_RATIO:-0.8}
                
                echo -e "\n${GREEN}Preparing dataset...${NC}"
                podman run --rm --cgroup-manager=cgroupfs --gpus all \
                    -v "$(pwd)/../../data:/app/data" \
                    -v "$(pwd)/../../models:/app/models" \
                    -v "$(pwd)/../../outputs:/app/outputs" \
                    llama-factory-health prepare \
                    --data_dir "$DATA_DIR" \
                    --output_file "$OUTPUT_FILE" \
                    --train_ratio "$TRAIN_RATIO"
                break
                ;;
            train)
                ensure_dirs
                
                echo -e "\n${BLUE}Model Training${NC}"
                read -p "Model path: " MODEL_PATH
                read -p "Dataset path: " DATASET_PATH
                read -p "Output path (default: /app/outputs): " OUTPUT_PATH
                OUTPUT_PATH=${OUTPUT_PATH:-"/app/outputs"}
                read -p "Additional arguments: " EXTRA_ARGS
                
                echo -e "\n${GREEN}Starting training...${NC}"
                podman run --rm --cgroup-manager=cgroupfs --gpus all \
                    -v "$(pwd)/../../data:/app/data" \
                    -v "$(pwd)/../../models:/app/models" \
                    -v "$(pwd)/../../outputs:/app/outputs" \
                    llama-factory-health train \
                    --model "$MODEL_PATH" \
                    --dataset "$DATASET_PATH" \
                    --output "$OUTPUT_PATH" \
                    $EXTRA_ARGS
                break
                ;;
            evaluate)
                ensure_dirs
                
                echo -e "\n${BLUE}Model Evaluation${NC}"
                read -p "Model path: " MODEL_PATH
                read -p "Test data path (default: /app/data/activity_dataset_test.json): " TEST_DATA
                TEST_DATA=${TEST_DATA:-"/app/data/activity_dataset_test.json"}
                
                echo -e "\n${GREEN}Evaluating model...${NC}"
                podman run --rm --cgroup-manager=cgroupfs --gpus all \
                    -v "$(pwd)/../../data:/app/data" \
                    -v "$(pwd)/../../models:/app/models" \
                    -v "$(pwd)/../../outputs:/app/outputs" \
                    llama-factory-health evaluate \
                    --model "$MODEL_PATH" \
                    --test_data "$TEST_DATA"
                break
                ;;
            build)
                echo -e "\n${GREEN}Building Docker image...${NC}"
                podman build --format docker --cgroup-manager=cgroupfs -t llama-factory-health .
                select_command
                break
                ;;
            help)
                usage
                select_command
                break
                ;;
            exit)
                echo -e "\n${GREEN}Exiting...${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option${NC}"
                ;;
        esac
    done
}

# Main execution
usage
select_command

echo -e "\n${GREEN}Done!${NC}"