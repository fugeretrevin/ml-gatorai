# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Cloud Run uses
ENV PORT=8080

# Start the web server
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 main:app
