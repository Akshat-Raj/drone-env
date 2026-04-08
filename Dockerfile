# Use a multi-stage build to keep the image size small
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project dependencies (pyproject.toml and the core package files)
COPY pyproject.toml .

# Copy everything else
COPY drone_env/ ./drone_env/

# Install project dependencies
RUN pip install --no-cache-dir .

# Expose the server port
EXPOSE 7860

# Set environment variables for competition
# These will be overriden by the grading script
ENV API_BASE_URL="http://localhost:7860"
ENV MODEL_NAME="llama3"

# Command to start the server using uvicorn
CMD ["python", "-m", "uvicorn", "drone_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
