# Use slim base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy only necessary files first to improve build caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# NLTK data download for punkt tokenizer
RUN python -m nltk.downloader punkt

# Expose the default port (Render automatically sets the actual port)
EXPOSE 8000

# Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

