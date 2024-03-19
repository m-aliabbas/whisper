# Start from an NVIDIA CUDA base image
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Ensure non-interactive mode for apt-get commands
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA for newer Python versions
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

# Install Python 3.9
RUN apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --set python3 /usr/bin/python3.9

# Install pip for Python 3.9
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Update pip and install requirements using Python 3.9
RUN python3.9 -m pip install --upgrade pip && \
    python3.9 -m pip install --no-cache-dir --default-timeout=120 -r requirements.txt

# Install Uvicorn
RUN python3.9 -m pip install uvicorn

# Expose the port Uvicorn will run on
EXPOSE 8000

# Set the working directory to WTranscriptor where server.py is located
WORKDIR /app/WTranscriptor

# Use the following command to run your Uvicorn server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
