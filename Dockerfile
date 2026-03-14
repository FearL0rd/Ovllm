# Ovllm Docker image
# Provides Ollama-like experience with vLLM backend

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Ovllm source
COPY . .

# Install Ovllm
RUN pip install -e .

# Create models directory
RUN mkdir -p /root/.ovllm/models

# Expose API port
EXPOSE 11434

# Default command
ENTRYPOINT ["ovllm"]
CMD ["serve", "--host", "0.0.0.0", "--port", "11434"]
