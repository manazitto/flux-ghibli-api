# 📸 Flux Ghibli Style - FastAPI Serverless API

Uma API serverless baseada em **FastAPI** e **RunPod**, utilizando **Stable Diffusion + LoRA** para transformar imagens reais em **arte estilo Studio Ghibli**, rodando **offline** de forma rápida e otimizada para GPU.

## ✨ Funcionalidades

- Recebe imagens via upload.
- Aplica transformação para o estilo Studio Ghibli.
- Retorna a imagem estilizada diretamente.
- 100% **offline** (sem necessidade de conexão com HuggingFace ou outros serviços externos).
- Memória otimizada para VRAM em GPUs.

## 🚀 Como funciona

A API recebe um arquivo de imagem (`multipart/form-data`), aplica o modelo `flux1-dev` combinado com o LoRA `flux-chatgpt-ghibli-lora`, e retorna a imagem transformada.

---

## 🛠️ Requisitos

- Python 3.10
- PyTorch com suporte CUDA
- Docker + RunPod (para serverless deployment)

---

## 📄 Estrutura do Projeto

```bash
.
├── app_fastapi.py       # Código principal da API
├── Dockerfile           # Instruções para buildar o container
├── requirements.txt     # Dependências do projeto
└── README.md            # Documentação
```

---

## 🧩 Endpoints

### `POST /ghibli`
- **Input**: Imagem enviada como `multipart/form-data`.
- **Output**: Arquivo `.png` transformado no estilo Studio Ghibli.

**Exemplo de cURL:**
```bash
curl -X POST http://localhost:3000/ghibli \
  -F "file=@caminho/para/sua/imagem.jpg" \
  --output resultado.png
```

---

## 🐳 Docker

Para rodar localmente usando Docker:

```bash
docker build -t flux-ghibli-api .
docker run -p 3000:3000 flux-ghibli-api
```

---

## ⚡ Deploy no RunPod

1. Suba este repositório no GitHub.
2. No RunPod, crie um **Serverless Endpoint** usando a opção **GitHub Repo**.
3. Escolha uma GPU compatível (24GB ou mais).
4. Configure a porta para `3000`.
5. Pronto: sua API Ghibli estará online, escalável e pronta para ser usada!

---

## 🤖 Créditos

- Base model: [flux1-dev](https://huggingface.co/openfree/flux1-dev)
- LoRA: [flux-chatgpt-ghibli-lora](https://huggingface.co/openfree/flux-chatgpt-ghibli-lora)
- Plataforma: [RunPod](https://runpod.io)

---

## 🛡️ Licença

Este projeto é distribuído sob a licença **MIT**.

---

### 🔥 Let's Ghiblify the World!
