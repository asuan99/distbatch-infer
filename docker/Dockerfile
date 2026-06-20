# Phase 5 stub: build image with CUDA + gRPC for reproducible runs.
# Requires nvidia-container-toolkit on the host.
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake g++ libgrpc++-dev protobuf-compiler-grpc libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN cmake -B build -S . && cmake --build build -j
