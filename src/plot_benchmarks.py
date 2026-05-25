import json
from pathlib import Path

import matplotlib.pyplot as plt


NO_CACHE_FILE = Path("outputs/benchmark_no_cache.json")
OPTIMIZED_FILE = Path("outputs/benchmark_optimized.json")
PLOTS_DIR = Path("outputs/plots")


def load_json(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_memory_chart(no_cache: dict, optimized: dict) -> None:
    labels = ["Sem cache", "Otimizado"]

    load_vram = [
        no_cache.get("load_vram_mb", 0) or 0,
        optimized.get("load_vram_mb", 0) or 0,
    ]

    peak_vram = [
        no_cache.get("peak_vram_mb", 0) or 0,
        optimized.get("peak_vram_mb", 0) or 0,
    ]

    x = [0, 1]
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(
        [i - width / 2 for i in x],
        load_vram,
        width=width,
        label="VRAM ao carregar (MB)",
    )
    plt.bar(
        [i + width / 2 for i in x],
        peak_vram,
        width=width,
        label="Pico de VRAM (MB)",
    )

    plt.xticks(x, labels)
    plt.ylabel("Memória (MB)")
    plt.title("Comparação de uso de VRAM")
    plt.legend()
    plt.tight_layout()

    output_file = PLOTS_DIR / "comparacao_vram.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gráfico salvo em: {output_file}")

def save_time_chart(no_cache: dict, optimized: dict) -> None:
    labels = ["Sem cache", "Otimizado"]
    times = [
        no_cache.get("generation_time_s", 0),
        optimized.get("generation_time_s", 0),
    ]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, times)
    plt.ylabel("Tempo (s)")
    plt.title("Comparação do tempo de geração")
    plt.tight_layout()

    output_file = PLOTS_DIR / "comparacao_tempo.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gráfico salvo em: {output_file}")


def save_tokens_chart(no_cache: dict, optimized: dict) -> None:
    labels = ["Sem cache", "Otimizado"]
    tokens = [
        no_cache.get("input_tokens", 0),
        optimized.get("input_tokens", 0),
    ]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, tokens)
    plt.ylabel("Quantidade de tokens")
    plt.title("Comparação do tamanho do contexto")
    plt.tight_layout()

    output_file = PLOTS_DIR / "comparacao_tokens.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gráfico salvo em: {output_file}")


def save_status_chart(no_cache: dict, optimized: dict) -> None:
    labels = ["Sem cache", "Otimizado"]

    no_cache_status = 1 if no_cache.get("status") == "success" else 0
    optimized_status = 1 if optimized.get("status") == "success" else 0

    values = [no_cache_status, optimized_status]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values)
    plt.yticks([0, 1], ["Falhou / OOM", "Sucesso"])
    plt.ylim(-0.1, 1.2)
    plt.title("Status da execução dos benchmarks")
    plt.tight_layout()

    output_file = PLOTS_DIR / "status_execucao.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gráfico salvo em: {output_file}")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    no_cache = load_json(NO_CACHE_FILE)
    optimized = load_json(OPTIMIZED_FILE)

    save_memory_chart(no_cache, optimized)
    save_time_chart(no_cache, optimized)
    save_tokens_chart(no_cache, optimized)
    save_status_chart(no_cache, optimized)

    print("\nTodos os gráficos foram gerados com sucesso.")


if __name__ == "__main__":
    main()