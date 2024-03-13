# Use the official Python image as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Uvicorn
RUN pip install uvicorn

# Expose the port Uvicorn will run on
EXPOSE 8000

# Set the working directory to WTranscriptor where server.py is located
WORKDIR /app/WTranscriptor

# Use the following command to run your Uvicorn server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
