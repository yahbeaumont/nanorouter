# Resultados

Tudo aqui foi medido neste repositório, no mesmo prompt e na mesma suite. O comando que reproduz
cada linha está em [Reprodução](#reprodução). Nenhum número foi copiado de outra máquina.

## Como foi medido

| item | valor |
|---|---|
| data | 2026-11-24 |
| hardware | 6 vCPU, 11 GB de RAM, sem GPU |
| suite | primeiros 150 itens de `data/test.jsonl` (n=150) |
| dataset | seed 1337, 6.000 registros, 4.556 enunciados distintos |
| 0.5B | `Qwen/Qwen2.5-0.5B-Instruct`, transformers em fp32 |
| 7B | `Qwen2.5-7B-Instruct` em GGUF `q4_k_m` via llama.cpp, 6 threads |
| adapter | LoRA r=16, 400 registros de treino, 1 época, CPU |
| decodificação | greedy (`do_sample=False`, `temperature=0`), teto de 192 tokens |

O 7B roda quantizado em `q4_k_m` porque é a forma em que ele seria servido em CPU. A comparação é
com o 0.5B em fp32, então o modelo grande carrega desvantagem de quantização, não vantagem.

## Tabela

| backend | n | exact_match | tool_acc | spec_valid | arg_f1 | p95 |
|---|---|---|---|---|---|---|
| `Qwen2.5-0.5B-Instruct` zero-shot | 150 | 0,113 | 0,293 | 0,287 | 0,659 | 12,5 s |
| `Qwen2.5-0.5B-Instruct` few-shot (4) | 150 | 0,273 | 0,700 | 0,420 | 0,585 | 6,4 s |
| `Qwen2.5-7B-Instruct` q4_k_m zero-shot | 150 | 0,227 | 0,833 | 0,687 | 0,697 | 9,5 s |
| **`Qwen2.5-0.5B-Instruct` + LoRA** | 150 | **0,587** | **0,973** | **0,807** | **0,744** | **6,4 s** |

`spec_valid` é a fração de respostas que passam na validação contra o spec; o complemento é o que
viola o contrato (slot inventado, obrigatório faltando, tipo errado, JSON inválido).

## O que os números dizem

**O modelo base de 0,5B não serve para roteamento sem ajuste.** Ele acerta a decisão completa em
11,3% dos casos e viola o contrato em 71,3% deles. Não é falta de capacidade de extrair slots
(`arg_f1` de 0,659): é falta de aderência ao formato e de escolha consistente da ferramenta.

**Few-shot melhora a seleção e piora o argumento.** Quatro exemplos no prompt levam `tool_acc` de
0,293 para 0,700 e `exact_match` de 0,113 para 0,273, mas `arg_f1` cai de 0,659 para 0,585: o modelo
passa a copiar o formato dos exemplos e, junto com ele, valores que não estavam no enunciado.

**O 7B é melhor em contrato, não em decisão exata.** Com 15x mais parâmetros, ele sobe `spec_valid`
para 0,687 e `tool_acc` para 0,833, mas fica em 0,227 de `exact_match`, abaixo do 0,5B com quatro
exemplos no prompt (0,273). Com n=150, essa diferença de 0,046 está dentro do ruído amostral e não
deve ser lida como vitória.

**O LoRA de 0,5B ganha do 7B em todas as métricas.** `exact_match` 0,587 contra 0,227; `spec_valid`
0,807 contra 0,687; `tool_acc` 0,973 contra 0,833; `arg_f1` 0,744 contra 0,697. E é mais rápido:
6,4 s de p95 contra 9,5 s, porque aprendeu a emitir uma saída curta e no formato, em vez de divagar.

A leitura que importa: **aderência ao contrato pesa mais que capacidade bruta** em tarefa de saída
fechada. O modelo 15x menor, ensinado a respeitar o schema, erra menos o contrato do que o modelo
grande que nunca foi ensinado.

A ressalva também importa: esse adapter viu 400 registros e 1 época, contra os 4.800 registros e 3
épocas de `configs/lora_sft.yaml`. Uma run completa em GPU deve subir os números, não descer. O que
está publicado aqui é o piso medido, não o teto estimado.

## Custo de reprodução

| etapa | tempo medido |
|---|---|
| 150 itens no 0,5B zero-shot | 896 s (~6,0 s por item) |
| 150 itens no 0,5B few-shot | 730 s (~4,9 s por item) |
| 150 itens no 7B q4_k_m | 799 s (~5,3 s por item) |
| treino LoRA, 400 registros, 1 época, CPU | 4.639 s (~11,6 s por registro) |
| 150 itens no adapter | 800 s (~5,3 s por item) |

O treino em CPU custa cerca de 11,6 s por registro: nos 4.800 registros de treino, uma época sairia
em torno de 15 horas. Em GPU, o README estima 20 minutos. Foi por isso que a run publicada usa um
subconjunto declarado de 400 registros.

## Limitações desta medição

- n=150. O intervalo de confiança de 95% em torno de 0,587 tem cerca de 16 pontos percentuais de
  largura; diferenças menores que isso não são distinguíveis nesta amostra. A distância entre 0,587
  e 0,227, no entanto, está muito fora do ruído.
- O adapter viu 400 dos 4.800 registros de treino, em uma época.
- O 7B está quantizado em `q4_k_m`.
- Uma única seed. A variância entre seeds não foi medida.
- Latências vêm de CPU compartilhada, sem isolamento de núcleo: servem para ordem de grandeza, não
  como benchmark de serving.

## Reprodução

```bash
make data                                    # dataset determinístico, seed 1337

# baselines de 0,5B (zero-shot e few-shot), 150 itens
python -m nanorouter.cli eval --limit 150 --out outputs/eval

# 7B quantizado, mesma suite
python -m nanorouter.cli eval --gguf models/qwen2.5-7b-instruct-q4_k_m.gguf \
  --skip-baselines --threads 6 --limit 150 --out outputs/eval_gguf

# adapter na mesma suite
make train                                   # GPU recomendada; em CPU, ver nota abaixo
python -m nanorouter.cli eval --adapter checkpoints/lora/final --skip-baselines \
  --limit 150 --out outputs/eval_adapter
```

A run de CPU publicada usou `scripts/train_cpu.py`, que fixa o subset e a época reduzida e grava
`run_meta.json` ao lado do checkpoint:

```bash
python scripts/train_cpu.py 400 1 2 4 640 checkpoints/lora_cpu
```

Cada comando de avaliação grava `RESULTS.md` e um `.jsonl` por backend com a predição de cada item,
para auditoria item a item. Os arquivos de `outputs/` não são versionados; a tabela acima é o
registro do que foi medido.
