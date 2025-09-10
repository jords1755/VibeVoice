FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy & install only core dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source & handler
COPY . .

CMD ["python3", "handler.py"]
