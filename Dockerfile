# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy required files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire src directory into the container
COPY src/ src/

# Set the working directory to src
WORKDIR /app/src

# Run prediction script
CMD ["python", "predict.py"]

