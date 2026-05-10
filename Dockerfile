# Use a lightweight Python image
FROM python:3.11-slim

# Install system dependencies required by LightGBM and OpenMP
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency file first (for better caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the fixed port (Railway's target port will be 8000)
EXPOSE 8000

# Run the FastAPI app on a fixed port
CMD uvicorn api.api:app --host 0.0.0.0 --port 8000
