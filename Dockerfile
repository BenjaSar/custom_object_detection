#Author: FS
#Date: February 2025

# Use the official Python image
FROM python:3.11.4

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY app/requirements.txt .

# Install dependencies
RUN apt-get update && apt-get install -y \
libgl1-mesa-glx \
libglib2.0-0 \
&& rm -rf /var/lib/apt/lists/* \
&& pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the application
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["gunicorn", \
      "-w", "2", \
      "-k" , "uvicorn.workers.UvicornWorker", \  
      "--bind", "0.0.0.0:8000", "main:app"]
