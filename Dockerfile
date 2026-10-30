# Multi-stage Dockerfile for Data Placement Predictor
# Optimized for Kubernetes deployment

FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheel distributions
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip==25.0.1 setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt


FROM python:3.11-slim

LABEL maintainer="Data Placement Predictor"
LABEL description="Kubernetes-ready data placement predictor for Onedata"

WORKDIR /app

# Install runtime dependencies only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 predictor

# Copy wheels from builder
COPY --from=builder /build/wheels /tmp/wheels

# Install Python packages from wheels
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip==25.0.1 && \
    pip install --no-cache-dir --no-index --find-links /tmp/wheels -r requirements.txt && \
    rm -rf /tmp/wheels

# Copy application code
COPY . .

# Set ownership to non-root user
RUN chown -R predictor:predictor /app

# Switch to non-root user
USER predictor

# Health check: can we import the main module?
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.main import main; print('OK')" || exit 1

# Default entrypoint: placement-job mode
ENTRYPOINT ["python", "-m", "src"]
CMD ["--log-level", "INFO"]
