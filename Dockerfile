# Stage 1: Builder
FROM python:3.12-slim AS builder

# Install uv for extremely fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation for faster cold starts
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Create a virtual environment
RUN uv venv /app/.venv
ENV VIRTUAL_ENV=/app/.venv

# Copy dependency definition
COPY pyproject.toml ./

# 1. Install CPU-only PyTorch first to prevent downloading huge GPU drivers
RUN uv pip install "torch>=2.12.0" "torchvision>=0.27.0" --index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining dependencies
RUN uv pip install -r pyproject.toml

# 3. Copy application code
COPY . /app

# 4. Pre-download HuggingFace model weights so the app starts instantly
# Note: Pass your HF token during build: docker build --build-arg HF_TOKEN=hf_...
ARG HF_TOKEN
ENV HF_HOME=/model_cache
RUN /app/.venv/bin/python -c "import os; from transformers import AutoProcessor, AutoModel; \
    token = os.environ.get('HF_TOKEN'); \
    AutoProcessor.from_pretrained('google/siglip-base-patch16-224', token=token); \
    AutoModel.from_pretrained('google/siglip-base-patch16-224', token=token)"


# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Copy the highly optimized virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy the pre-downloaded HuggingFace model weights
COPY --from=builder /model_cache /model_cache
ENV HF_HOME=/model_cache

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Copy the actual application files
COPY . /app

# Ensure the start script is executable
RUN chmod +x /app/start.sh

# Expose FastAPI and Chainlit ports
EXPOSE 8000 8501

# Launch both services
CMD ["/app/start.sh"]
