# Dockerfile for training
# Start from a base Python image
FROM python:3.10

ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

WORKDIR /app

# Copy data folder to Docker
COPY ../data/ /app/data

# Copy models folder to Docker
COPY ../outputs /app/outputs

# Copy the code itself
COPY ../src/inference/run_inference.py /app/src/inference/run_inference.py

# Copy usefull scripts and configs
COPY ../${CONF_PATH} /app

# Install any necessary packages listed in requirements.txt
COPY ../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Train the model
RUN python3 src/inference/run_inference.py