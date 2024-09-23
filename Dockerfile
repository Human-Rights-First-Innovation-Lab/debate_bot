# Use the official Python base image
FROM python:3.12-slim-bookworm

# Set environment variables to ensure real-time logs (output is not buffered)
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the Docker container
WORKDIR /app

# Install system dependencies required by some Python libraries (e.g., MySQL client)
RUN apt-get update && apt-get install -y --no-install-recommends libmariadb-dev

# Copy requirements.txt to the working directory in the container
COPY requirements.txt .

# Install Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the app directory into the container at /app
COPY app/ ./app/

# Copy README.md for documentation purposes
COPY README.md .

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "*"]



# run the following after building
# docker run --env-file .env my-python-app
