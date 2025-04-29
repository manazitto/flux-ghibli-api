import os
import gc
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import io

# Configurações mais agressivas para economizar memória
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"

app = FastAPI()

# Função mais agressiva para limpeza de memória
def force_gc():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()

# Função auxiliar para uso de GPU (0 a 1)
def _gpu_usage():
    total = torch.cuda.get_device_properties(0).total_memory
    return torch.cuda.memory_allocated(0) / total if total else 0.0

# Limpa toda a memória antes de carregar
force_gc()

# Tente carregar o modelo com opções mais agressivas de economia de memória
pipe = AutoPipelineForText2Image.from_pretrained(
    "./models/flux1-dev",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    # device_map="auto"  # Não suportado nessa versão
)

# Mova o pipeline inteiro para CUDA da maneira padrão
if torch.cuda.is_available():
    pipe = pipe.to("cuda")

pipe.load_lora_weights(
    "./loras/flux-ghibli-lora",
    weight_name="flux-chatgpt-ghibli-lora.safetensors"
)

@app.post("/ghibli")
async def ghibli_style(file: UploadFile = File(...)):
    contents = await file.read()
    init_image = Image.open(io.BytesIO(contents)).convert("RGB").resize((512, 512))

    prompt = (
        "Studio Ghibli painting style, soft light, pastel colors, dreamy background, "
        "original face preserved, high detail, watercolor texture"
    )

    # Limpa cache antes de gerar
    force_gc()

    # Tente executar com cuidado para evitar estouros de memória
    with torch.cuda.amp.autocast():  # Usa precisão mista automática
        result = pipe(
            prompt=prompt, 
            image=init_image,
            num_inference_steps=25,  # Reduz steps para economizar memória
            guidance_scale=7.0       # Valor padrão, mas explícito
        ).images[0]

    # Limpa memória imediatamente após uso
    force_gc()

    output_path = "output.png"
    result.save(output_path)
    return FileResponse(output_path, media_type="image/png")
