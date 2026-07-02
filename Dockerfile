FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Use Render's PORT env var (defaults to 10000)
ENV PORT=10000
EXPOSE ${PORT}

# Health check using the dynamic port
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",10000)}/health')" || exit 1

# Start the server immediately — index is already built
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT}"
