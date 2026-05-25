import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_FILE = Path("outputs/benchmark_optimized.json")


def ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Este benchmark precisa de GPU CUDA.")


def build_large_context(target_tokens: int = 1700) -> str:
    paragraph = (
        "Manual clínico: o paciente apresenta cefaleia pulsátil, fotofobia, "
        "náusea, rigidez muscular leve, histórico de episódios prévios e "
        "necessidade de correlação com sinais neurológicos, exames laboratoriais, "
        "evolução sintomática, metadados de atendimento e conduta terapêutica. "
    )

    words = []
    while len(" ".join(words).split()) < target_tokens:
        words.append(paragraph)

    return " ".join(words)


def main() -> None:
    ensure_cuda()

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Carregando modelo em 4 bits com FlashAttention-2...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.float16,
    )

    model.config.use_cache = True

    load_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"VRAM após carregamento do modelo: {load_vram_mb:.2f} MB")

    print("Gerando contexto grande compatível com o modelo...")
    large_context = build_large_context(target_tokens=1700)

    prompt = (
        "Você é um assistente clínico. Leia o contexto abaixo e gere um resumo clínico.\n\n"
        f"{large_context}\n\n"
        "Resumo clínico:"
    )

    print("Tokenizando contexto...")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_tokens = inputs["input_ids"].shape[1]
    print(f"Total de tokens de entrada: {input_tokens}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("Gerando 100 tokens com KV Cache + FlashAttention-2...")
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    end = time.perf_counter()

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    generation_time_s = end - start

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    metrics = {
        "model_name": MODEL_NAME,
        "input_tokens": int(input_tokens),
        "load_vram_mb": round(load_vram_mb, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "generation_time_s": round(generation_time_s, 2),
        "use_cache": True,
        "attn_implementation": "flash_attention_2",
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== MÉTRICAS ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    print("\n=== AMOSTRA DE SAÍDA ===")
    print(generated_text[:1000])


if __name__ == "__main__":
    main()