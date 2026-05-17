# Hugging Face Spaces Docker deployment for iQueue Flask backend
# Port 7860 is required by Hugging Face Spaces

FROM python:3.11-slim

# Create a non-root user (HF Spaces best practice)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy and install dependencies first (layer caching)
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all backend source files
COPY --chown=user . /app

# Expose HF Spaces required port
EXPOSE 7860

# Run with gunicorn - 1 worker to stay within free-tier memory
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "--preload", "app:app"]
