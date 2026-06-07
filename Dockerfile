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

# 4. Pre-download CLIP model weights so the app starts instantly
ENV TORCH_HOME=/model_cache
ENV XDG_CACHE_HOME=/model_cache
RUN /app/.venv/bin/python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')"


# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Copy the highly optimized virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy the pre-downloaded CLIP model weights
COPY --from=builder /model_cache /model_cache
ENV TORCH_HOME=/model_cache
ENV XDG_CACHE_HOME=/model_cache

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Copy the actual application files
COPY . /app

# Ensure the start script is executable
RUN chmod +x /app/start.sh

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Launch both services
CMD ["/app/start.sh"]
