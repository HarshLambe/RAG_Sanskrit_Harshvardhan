# Use an official Python slim image to keep it lightweight
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for compilation
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Install PyTorch for CPU explicitly to save massive amounts of space (no CUDA)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code into the container
COPY . /app/

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit application
CMD ["python", "-m", "streamlit", "run", "code/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
