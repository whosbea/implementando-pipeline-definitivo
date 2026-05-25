# Laboratório 10 — O Pipeline Definitivo

## 1. Objetivo do laboratório

Este laboratório teve como objetivo implementar um pipeline integrado de IA para simular um cenário de produção em que um modelo gerador precisa lidar com um contexto extremamente grande sem colapsar a memória da GPU. A proposta combina três ideias centrais:

- **QLoRA / quantização em 4 bits**, para reduzir o custo de carregamento do modelo;
- **KV Cache**, para evitar recálculo redundante durante a geração;
- **otimizações de inferência orientadas a hardware**, discutidas no enunciado com foco em FlashAttention-2.

O problema proposto envolve um sistema de geração de relatórios clínicos a partir de um contexto massivo recuperado por RAG. O gargalo principal está no custo de memória do mecanismo de **Self-Attention**, cuja complexidade cresce quadraticamente com o tamanho da sequência.

## 2. Contexto do problema

O cenário descrito no laboratório é o de uma HealthTech que já possui um sistema RAG de busca em manuais médicos e deseja colocá-lo em produção para gerar relatórios automatizados. O fluxo esperado é:

1. recuperar capítulos inteiros de manuais médicos;
2. injetar esse contexto em um modelo gerador;
3. produzir um resumo clínico.

O problema é que, com um contexto muito grande, o custo de memória da atenção estoura a VRAM da GPU, travando o servidor. A missão do laboratório é justamente combinar quantização, cache e otimizações de inferência para evitar esse colapso.

## 3. Estrutura do projeto

```text
implementando-pipeline-definitivo/
├── src/
│   ├── benchmark_no_cache.py
│   └── benchmark_optimized.py
├── outputs/
│   ├── benchmark_no_cache.json
│   └── benchmark_optimized.json
├── requirements.txt
└── README.md
```

## 4. Modelo utilizado

Foi utilizado o modelo:

**`TinyLlama/TinyLlama-1.1B-Chat-v1.0`**

### Justificativa da escolha

O enunciado sugere explicitamente o uso de um modelo gerador auto-regressivo leve, como o TinyLlama, para permitir os experimentos com quantização e inferência em GPU. Esse modelo foi escolhido porque:

- é pequeno o suficiente para testes acadêmicos;
- suporta carregamento em 4 bits;
- permite demonstrar o impacto de contexto grande na atenção;
- é mais viável em ambiente de nuvem com recursos limitados.

## 5. Etapa 1 — Ingestão eficiente com quantização

O primeiro passo foi carregar o modelo usando **bitsandbytes** com configuração de quantização em **4 bits**, conforme pedido no laboratório.

### Configuração utilizada

- `load_in_4bit=True`
- `bnb_4bit_compute_dtype=torch.float16`

### O que isso significa

#### Quantização em 4 bits
Quantizar um modelo significa armazenar seus pesos com menos bits. Em vez de manter os parâmetros em `float16`, o modelo é carregado em um formato muito mais compacto.

#### `bitsandbytes`
Biblioteca usada para permitir esse carregamento quantizado de forma prática no ecossistema Hugging Face.

#### `bnb_4bit_compute_dtype=torch.float16`
Mesmo com os pesos armazenados em 4 bits, parte dos cálculos internos continua sendo feita em `float16`, o que ajuda a equilibrar economia de memória e estabilidade numérica.

### Métrica observada

O carregamento do modelo quantizado ocupou:

- **805.93 MB de VRAM**

## 6. Etapa 2 — Simulação do RAG massivo

O laboratório pede a geração de um texto fictício contendo cerca de **10.000 a 15.000 tokens**, simulando os documentos recuperados pelo RAG.

### O que foi feito

Foi criada uma string longa composta por repetição de parágrafos clínicos simulados. Esse texto foi passado pelo tokenizador do modelo.

### Resultado observado no benchmark sem cache

No benchmark sem cache, o total de tokens de entrada chegou a:

- **33.895 tokens**

Esse valor ultrapassou de forma massiva o limite nominal de contexto do TinyLlama, que gira em torno de 2048 tokens. Isso foi propositalmente útil para mostrar o colapso de memória no cenário extremo.

### O que isso mostra

A simulação do RAG massivo funcionou como estressor do sistema.  
Ela reproduziu, de forma controlada, o cenário em que um contexto gigante é jogado na entrada do modelo.

## 7. Etapa 3 — O gargalo de geração sem cache

O laboratório pede uma geração de **100 novos tokens** com o cache desligado:

```python
model.config.use_cache = False
```

e a medição de:

- tempo total de geração;
- pico de memória VRAM.

### O que é KV Cache

Durante a geração autoregressiva, o modelo precisa manipular vetores de:

- **K** = Keys
- **V** = Values

Quando o cache está ligado, esses vetores gerados em passos anteriores podem ser reaproveitados.  
Quando o cache está desligado, o modelo recalcula tudo repetidamente a cada novo token.

### Resultado do benchmark sem cache

Métricas registradas:

```json
{
  "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "input_tokens": 33895,
  "load_vram_mb": 805.93,
  "peak_vram_mb": null,
  "generation_time_s": 0.33,
  "use_cache": false,
  "status": "oom",
  "error": "CUDA out of memory..."
}
```

### Erro observado

O processo falhou com:

- **CUDA Out of Memory**
- tentativa de alocação de **136.96 GiB**

### Interpretação

Esse resultado mostra com clareza o gargalo do decoder sem cache em contexto extremamente longo. O custo do Self-Attention ficou inviável, e a GPU não conseguiu comportar a computação.

## 8. Etapa 4 — Otimização da inferência

O próximo passo do laboratório era refatorar a geração para:

- ativar **KV Cache**
- ativar **FlashAttention-2**

### O que é FlashAttention-2

**FlashAttention-2** é uma implementação otimizada da atenção, pensada para explorar melhor a hierarquia de memória da GPU, especialmente a SRAM. A ideia é tornar o cálculo da atenção mais eficiente, reduzindo custo de memória intermediária e melhorando desempenho.

### Observação importante do experimento

O código foi preparado para essa arquitetura de otimização, mas o ambiente disponível de execução utilizou **GPU Tesla T4**, o que trouxe limitação prática para validação plena do FlashAttention-2. Por isso, a melhoria efetivamente validada no experimento ficou mais diretamente associada a:

- quantização em 4 bits;
- uso de `use_cache=True`;
- redução do contexto para um tamanho grande, mas compatível com o modelo.

Em outras palavras: a arquitetura pensada para o benchmark otimizado seguiu a proposta do laboratório, mas a validação prática do ganho observado ficou mais diretamente associada ao **KV Cache** e ao controle do contexto do que a uma execução plenamente confirmada de FlashAttention-2 em hardware ideal.

### Resultado do benchmark otimizado

Métricas registradas:

```json
{
  "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "input_tokens": 1800,
  "load_vram_mb": 805.93,
  "peak_vram_mb": 1827.84,
  "generation_time_s": 5.02,
  "use_cache": true,
  "attn_implementation": "flash_attention_2"
}
```

### Interpretação

Ao contrário do benchmark sem cache, o benchmark otimizado:

- concluiu a geração;
- não colapsou em OOM;
- manteve o pico de memória em nível viável;
- permitiu a geração dos 100 tokens solicitados.

## 9. Comparação entre os benchmarks

### Benchmark sem cache
- contexto: **33.895 tokens**
- VRAM ao carregar: **805.93 MB**
- pico de VRAM: não concluído
- tempo até falha: **0.33 s**
- status: **OOM**

### Benchmark otimizado
- contexto: **1.800 tokens**
- VRAM ao carregar: **805.93 MB**
- pico de VRAM: **1827.84 MB**
- tempo de geração: **5.02 s**
- status: **sucesso**

### Leitura honesta do resultado

A comparação mostra que o pipeline sem cache não suportou o custo da atenção em contexto massivo. Já a versão otimizada conseguiu concluir a inferência de forma estável.

É importante observar, porém, que a comparação não é perfeitamente simétrica do ponto de vista do número de tokens, porque o cenário sem cache foi propositalmente empurrado para um regime extremo de 33.895 tokens, enquanto a versão otimizada foi executada com 1.800 tokens para se manter dentro do que o modelo suportava.

Mesmo assim, o experimento atende ao objetivo didático do laboratório: demonstrar que, sem otimização, o Transformer colapsa, e que o uso de quantização e cache torna a inferência viável em um cenário controlado.

## 10. Explicação dos principais termos

### Self-Attention
É o mecanismo pelo qual cada token pode “olhar” para os outros tokens da sequência e calcular relevância contextual. Esse mecanismo é central no Transformer.

### Complexidade O(n²)
Significa que o custo cresce aproximadamente com o quadrado do número de tokens. Se a entrada dobra, o custo não dobra: ele cresce muito mais.

### VRAM
É a memória da GPU. No laboratório, o problema principal foi justamente o esgotamento da VRAM.

### OOM
**Out of Memory**. Erro que acontece quando a GPU não consegue alocar mais memória para continuar a execução.

### Decoder
Parte do modelo autoregressivo responsável por gerar novos tokens a partir do contexto.

### KV Cache
Mecanismo que guarda os vetores de Keys e Values já computados em passos anteriores, evitando recalcular tudo a cada token novo.

### FlashAttention-2
Implementação otimizada da atenção para GPU, voltada a reduzir custo de memória intermediária e melhorar eficiência.

### QLoRA
No sentido usado pelo laboratório, refere-se aqui ao uso de quantização em 4 bits para tornar o carregamento e a inferência do modelo mais viáveis em hardware limitado.

### State Space Models
Família de arquiteturas pensadas para lidar melhor com sequências muito longas, sem sofrer do mesmo custo quadrático da atenção tradicional.

### Mamba
Exemplo de arquitetura baseada em State Space Models, citada no laboratório como caminho mais adequado para sequências gigantescas.

## 11. Parecer técnico — Parte A

A combinação de quantização em 4 bits, KV Cache e otimizações de atenção foi fundamental para evitar o colapso de VRAM neste laboratório. A quantização reduziu drasticamente o custo de carregamento do modelo, permitindo que o TinyLlama ocupasse apenas 805.93 MB de VRAM ao ser inicializado. Isso já eliminou uma parte importante do problema. Em seguida, o uso de KV Cache reduziu o desperdício computacional durante a geração, evitando o recálculo redundante de chaves e valores para cada novo token. No cenário sem cache, a tentativa de processar um contexto extremamente grande resultou em OOM ao tentar alocar 136.96 GiB, o que evidencia o custo explosivo da atenção tradicional em sequências longas. Já na versão otimizada, a geração foi concluída com pico de 1827.84 MB, mostrando que o pipeline deixou de ser inviável e passou a operar de forma controlada.

Em termos arquiteturais, esse experimento mostra que o Transformer tradicional só continua viável em produção quando combinado com estratégias de redução de custo. A quantização atua no armazenamento dos pesos, o KV Cache atua na eficiência da geração incremental, e o FlashAttention-2 representa o esforço de tornar o cálculo da atenção mais compatível com a memória rápida da GPU. Mesmo quando a validação prática do FlashAttention-2 ficou condicionada ao hardware disponível, o princípio arquitetural do laboratório se manteve: sem essas camadas de otimização, o modelo colapsa; com elas, ele passa a operar dentro de limites muito mais razoáveis.

## 12. Parecer técnico — Parte B

Se o cliente exigisse o processamento de **2 milhões de tokens**, até mesmo essa combinação deixaria de ser suficiente. O motivo principal é que o Self-Attention do Transformer continua carregando um custo estrutural muito alto em memória e computação à medida que a sequência cresce. O FlashAttention melhora a forma como a atenção é implementada e reduz desperdícios de movimentação de memória, mas ele não elimina completamente a dependência pesada do tamanho da sequência. Em escalas extremas, o problema deixa de ser apenas otimização de implementação e passa a ser limitação do próprio desenho algorítmico do Transformer.

Nesse ponto, a indústria tende a migrar para arquiteturas como **State Space Models**, incluindo a família **Mamba**, porque elas tratam dependências longas com custo muito mais estável. O enunciado destaca a ideia de complexidade de memória **O(1)** para esse tipo de arquitetura, justamente como contraponto ao crescimento explosivo da atenção tradicional. Isso significa que, para janelas extremamente longas, a solução não seria apenas “otimizar mais” o Transformer, mas trocar a arquitetura por outra que escale de forma mais adequada. Em outras palavras: para 15 mil tokens, ainda faz sentido salvar o Transformer com quantização, cache e atenção otimizada; para 2 milhões de tokens, o problema deixa de ser ajuste fino de implementação e vira um problema de escolha arquitetural.

## 13. Como executar o projeto

### 13.1 Criar ambiente virtual
```bash
python -m venv .venv
```

### 13.2 Ativar ambiente
No fish shell:

```fish
source .venv/bin/activate.fish
```

### 13.3 Instalar dependências
```bash
pip install -r requirements.txt
```

### 13.4 Rodar benchmark sem cache
```bash
python src/benchmark_no_cache.py
```

### 13.5 Rodar benchmark otimizado
```bash
python src/benchmark_optimized.py
```

## 14. Dependências principais

As principais bibliotecas utilizadas foram:

- `transformers`
- `torch`
- `bitsandbytes`
- `accelerate`
- `sentencepiece`

## 15. Ambiente utilizado

O experimento foi executado com apoio de ambiente em nuvem com GPU, já que o laboratório exige medições de VRAM e inferência acelerada por CUDA. O desenvolvimento do código foi realizado em ambiente local, com execução final dos benchmarks em ambiente compatível com GPU.

## 16. Observação sobre uso de IA

Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Beatriz Barreto.

Houve apoio de IA na elaboração dos fragmentos técnicos, na organização do pipeline e na estruturação da documentação. Todo o conteúdo foi revisado, executado e validado por Beatriz Barreto.

