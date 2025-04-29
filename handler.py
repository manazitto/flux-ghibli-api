import os
import gc
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import io
import base64
import runpod
import time

# --- Configurações e Funções Auxiliares ---

# Configuração CUDA mais agressiva
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"

# Função agressiva para limpeza de memória
def force_gc():
    print("Forçando coleta de lixo e limpeza de cache CUDA...")
    start_mem = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()
    end_mem = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
    print(f"Memória liberada: {(start_mem - end_mem) / (1024**2):.2f} MB")

# --- Carregamento do Modelo (Executado uma vez por worker) ---

print("Iniciando carregamento do modelo...")
force_gc()

MODEL_PATH = "./models/flux1-dev" # Certifique-se que este caminho existe no container
LORA_PATH = "./loras/flux-ghibli-lora" # Certifique-se que este caminho existe
LORA_WEIGHT = "flux-chatgpt-ghibli-lora.safetensors"

pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

print("Movendo pipeline para CUDA...")
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    print("Pipeline movido para CUDA.")
else:
    print("CUDA não disponível, usando CPU.")

print("Carregando LoRA...")
pipe.load_lora_weights(
    LORA_PATH,
    weight_name=LORA_WEIGHT
)
print("LoRA carregado.")

force_gc()
print("Modelo e LoRA carregados com sucesso.")

# --- Handler do Runpod (Executado para cada requisição) ---

def handler(event):
    print("Recebido evento:", event)
    job_input = event.get('input', {})

    image_base64 = job_input.get('image_base64')
    prompt_override = job_input.get('prompt') # Opcional para sobrescrever prompt
    num_steps = job_input.get('num_inference_steps', 25)
    guidance = job_input.get('guidance_scale', 7.0)

    if not image_base64:
        return {"error": "Parâmetro 'image_base64' não encontrado no input."}

    try:
        # Decodifica a imagem base64
        image_bytes = base64.b64decode(image_base64)
        init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((512, 512))
    except Exception as e:
        return {"error": f"Erro ao decodificar ou abrir a imagem: {str(e)}"}

    # Define o prompt
    default_prompt = (
        "Studio Ghibli painting style, soft light, pastel colors, dreamy background, "
        "original face preserved, high detail, watercolor texture"
    )
    prompt = prompt_override if prompt_override else default_prompt

    print(f"Iniciando inferência com prompt: '{prompt[:50]}...' e {num_steps} steps.")
    start_time = time.time()

    # Limpa memória antes da geração
    force_gc()

    try:
        # Executa a inferência com precisão mista
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            result_image = pipe(
                prompt=prompt,
                image=init_image,
                num_inference_steps=int(num_steps),
                guidance_scale=float(guidance)
            ).images[0]
    except Exception as e:
        force_gc() # Limpa memória em caso de erro
        return {"error": f"Erro durante a inferência: {str(e)}"}

    # Limpa memória após a geração bem-sucedida
    force_gc()

    end_time = time.time()
    print(f"Inferência concluída em {end_time - start_time:.2f} segundos.")

    # Codifica a imagem resultante para base64
    buffered = io.BytesIO()
    result_image.save(buffered, format="PNG") # Salva como PNG para melhor qualidade
    output_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Retorna o resultado
    return {"image_base64": output_image_base64}

# Inicia o handler do Runpod
if __name__ == "__main__":
    print("Iniciando worker Runpod...")
    runpod.serverless.start({"handler": handler})
