# Use official Python 3.11 slim image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies for PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy pyproject.toml and poetry.lock first to leverage Docker cache
COPY pyproject.toml poetry.lock* /app/

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch==2.9.1 \
    torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install dependencies into container
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --only main

# Copy the rest of the project files
COPY . /app

# Expose port for FastAPI
EXPOSE 8000

# Run FastAPI app via uvicorn
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]