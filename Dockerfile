FROM python:3.10-slim

# Set environment variables
ENV HOME=/app \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLEXSRFPROTECTION=false \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    PYTHONUNBUFFERED=1

# Skip system dependencies - using headless OpenCV

# Create app directory and set permissions
RUN mkdir -p /app/.streamlit /tmp/matplotlib /tmp/numba_cache && \
    chmod -R 777 /app /tmp/matplotlib /tmp/numba_cache

WORKDIR /app

# Copy and install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories and config
RUN mkdir -p /app/thumbnails /app/exported_reports && \
    chmod -R 777 /app && \
    echo "[server]\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\n" > /app/.streamlit/config.toml

EXPOSE 7860

# Run as non-root user
USER 1000

CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0", "--server.fileWatcherType", "none"]
