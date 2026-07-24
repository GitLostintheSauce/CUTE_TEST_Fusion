# CUTE dashboard: a self-contained image for the diagnostic dashboard and the
# ML surrogate. The heavy Open Fusion Toolkit (TokaMaker) solver is NOT needed
# to view shots or run the surrogate, so it is intentionally left out to keep
# the image small and the build fast.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8050

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e ".[deploy]"

# Copy the rest of the project (scripts, models, config, data placeholders).
COPY . .

# Generate the synthetic demo shot at build time so the dashboard has content
# without committing a large binary to git. (Uses only sensor geometry, no OFT.)
RUN python scripts/generate_synthetic_shot.py

EXPOSE 8050

# Bind to the platform-provided port when present (Render, HF Spaces, etc.).
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8050} --workers 2 --timeout 120 wsgi:server"]
