FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Defina variáveis de ambiente úteis
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/huggingface_cache
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32


# Copia o arquivo de dependências e instala
COPY requirements.txt .
# Use --no-cache-dir para economizar espaço na imagem
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação
COPY handler.py .

RUN mkdir -p models/flux1-dev loras/flux-ghibli-lora

# Define o comando para iniciar o handler do Runpod
CMD ["python", "-u", "handler.py"]
