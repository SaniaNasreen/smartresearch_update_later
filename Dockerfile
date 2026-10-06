# Use lightweight official Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies (spaCy model is pinned in requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK resources
RUN python -c "import nltk; nltk.download('stopwords')"

# Copy app files into container
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Basic health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the app when the container starts
CMD ["streamlit", "run", "smartresearch_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
