# Use official Python runtime
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy everything into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Railway provides PORT env automatically
ENV PORT=8080

# Expose the port for Railway
EXPOSE 8080

# Start your Flask app
CMD ["python", "main.py"]
