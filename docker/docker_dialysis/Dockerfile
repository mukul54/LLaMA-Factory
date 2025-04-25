# Use NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.09-py3

# Define environments
ENV MAX_JOBS=4
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create directories
RUN mkdir -p /app/data /app/models /app/outputs

# Copy and make executable the entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Clone and install LLaMA Factory
RUN git clone https://github.com/mukul54/LLaMA-Factory && \
    cd LLaMA-Factory && \
    pip install -e .

WORKDIR /app/LLaMA-Factory

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["webui"]