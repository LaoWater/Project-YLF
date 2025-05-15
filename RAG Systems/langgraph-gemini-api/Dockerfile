# Use Python 3.12 base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Expose the correct port
EXPOSE 8080

# Run the FastAPI app
CMD ["uvicorn", "api_server:api", "--host", "0.0.0.0", "--port", "8080"]
