FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p content
RUN mkdir -p faiss_index

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["python", "run_server.py"]