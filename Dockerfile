# Use base image from AllenAI OLMo (has python 3.10)
FROM ghcr.io/allenai/pytorch:2.0.0-cuda11.8-python3.10
ENV CUDA_HOME=/opt/conda

# Set the working directory in the Docker image
WORKDIR /nlp-from-scratch

# Copy the requirements.txt file from your local system to the Docker image
COPY requirements.txt ./

# Install sox, libsndfile1, and ffmpeg
RUN apt-get update && apt-get install -y sox libsndfile1 ffmpeg

# Upgrade pip in the Docker image
RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Install olmo and nemo_toolkit (keep these separate from requirements.txt to avoid conflicts with the other Dockerfile)
RUN pip install ai2-olmo
RUN pip install nemo_toolkit['all']