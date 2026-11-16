# MaDoc: the web application (FastAPI: kraken segmentation + recognition, NER models). Ollama runs in its own
# container. The acts and the settings are kept in /data (a volume in docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HOME=/cache/huggingface

# PyTorch for CUDA 12.8 first (the RTX 50 series needs it); kraken 7.0.2 accepts torch 2.4 to 2.10
RUN pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# the code, the dictionaries, the samples, the logos and the models (a new model = a new image)
COPY . .
RUN useradd -m -u 1000 madoc && mkdir -p /cache/huggingface /data && chown -R madoc /cache /data
USER madoc

EXPOSE 7860
ENV MADOC_DATA_DIR=/data
CMD ["python", "src/server.py", "--host", "0.0.0.0", "--device", "auto"]
