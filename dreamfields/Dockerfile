# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/tensorflow:21.12-tf2-py3

# Install dependencies
RUN apt-get update && apt-get install -y ffmpeg
WORKDIR /install
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY . /dreamfields

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
WORKDIR /dreamfields
