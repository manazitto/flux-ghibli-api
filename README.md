# API Serverless Estilo Ghibli com FLUX no Runpod

Este repositório contém o código para uma API serverless no Runpod que aplica um estilo Ghibli (usando um LoRA específico) a uma imagem de entrada usando o modelo FLUX.

## Estrutura

- `handler.py`: Lógica principal da API, carregamento do modelo e processamento de requisições.
- `requirements.txt`: Dependências Python.
- `Dockerfile`: Define a imagem Docker para o ambiente Runpod.

## Configuração

1.  **Modelos e LoRAs:**
    - Certifique-se de que o modelo base (`flux1-dev`) e o LoRA (`flux-ghibli-lora`) estejam acessíveis dentro do container.
    - **Opção A (Recomendado para Serverless):** Monte um volume de rede do Runpod contendo os modelos em `./models` e os LoRAs em `./loras` dentro do container.
    - **Opção B:** Modifique o `Dockerfile` para baixar os modelos/LoRAs durante a construção da imagem (descomente as linhas `RUN huggingface-cli download...`). Isso aumentará o tamanho da imagem.

2.  **Imagem Docker:**
    - Construa a imagem Docker:
      ```bash
      docker build -t seu-usuario/runpod-ghibli-api .
      ```
    - Envie a imagem para um registro (Docker Hub, ECR, etc.).

3.  **Runpod Serverless Endpoint:**
    - Crie um novo Endpoint Serverless no Runpod.
    - Configure-o para usar a imagem Docker que você enviou.
    - Associe o volume de rede (se estiver usando a Opção A para modelos).
    - Ajuste as configurações de GPU, memória e concorrência conforme necessário.

## Uso

Envie uma requisição POST para o URL do seu endpoint Runpod com um JSON no corpo:

```json
{
  "input": {
    "image_base64": "<sua imagem codificada em base64 aqui>",
    "prompt": "(Opcional) Studio Ghibli painting style, high detail...",
    "num_inference_steps": 30, 
    "guidance_scale": 7.5
  }
}
```

**Parâmetros de Entrada:**

- `image_base64` (Obrigatório): String contendo a imagem de entrada codificada em Base64.
- `prompt` (Opcional): Sobrescreve o prompt padrão.
- `num_inference_steps` (Opcional): Número de passos de inferência (padrão: 25).
- `guidance_scale` (Opcional): Escala de orientação (padrão: 7.0).

**Resposta:**

Um JSON contendo a imagem resultante codificada em base64:

```json
{
  "image_base64": "<imagem resultante codificada em base64 aqui>"
}
```

Ou um erro:

```json
{
  "error": "Mensagem de erro detalhada."
}
```

## Otimizações

O código inclui várias otimizações para uso de memória CUDA:

- Configuração `PYTORCH_CUDA_ALLOC_CONF`.
- Limpeza agressiva de memória (`force_gc`) antes e depois da inferência.
- Uso de `torch.cuda.amp.autocast` para precisão mista.
- Carregamento do modelo fora do handler principal.
