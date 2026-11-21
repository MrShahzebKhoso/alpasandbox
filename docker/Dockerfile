# App image (no GPU): python:3.12-slim + alpasandbox.
# Model image: build with --build-arg MODEL=edge on a CUDA host (installs NVlabs/alpamayo2-edge).
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
ARG MODEL=""
RUN pip install --no-cache-dir . && \
    if [ "$MODEL" = "edge" ]; then pip install --no-cache-dir "alpamayo2-edge @ git+https://github.com/NVlabs/alpamayo2-edge"; fi
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["alpasandbox"]
