# Resultados

Este arquivo é **gerado** por `nanorouter eval` (`make eval`), que escreve a tabela de comparação e uma matriz de confusão por backend em `outputs/eval/`.

## Estado atual

Nenhuma run de treino ou avaliação foi executada neste repositório até o commit inicial. Por isso não há tabela de resultados aqui: qualquer número publicado sem a run correspondente seria afirmação sem lastro, e o ponto deste projeto é justamente medir.

Para popular esta página na sua máquina:

```bash
make data          # dataset sintético determinístico (seed 1337)
make train         # LoRA no Qwen2.5-0.5B-Instruct
make eval          # baselines + fine-tunado na mesma suite
```

O `make eval` grava `outputs/eval/RESULTS.md` com as colunas `exact_match`, `tool_acc`, `spec_valid`, `arg_f1` e `p95_ms` por backend. Copie o conteúdo para cá quando a run terminar, junto com a data, o hardware e o commit.

## O que já está verificado

| verificação | comando | resultado |
|---|---|---|
| dataset | `make data` | 6.000 registros, 100% válidos contra o spec |
| diversidade | `make data` | 4.556 enunciados distintos em 6.000 registros |
| distribuição | `make data` | 70,1% tool · 18,0% hard negative · 11,9% `no_tool` |
| testes | `make test` | 35 testes, 85% de cobertura |
| lint e tipos | `make lint` | ruff e mypy sem apontamentos |

Esses números são de propriedades do gerador e da suíte: eles dizem que o dado é válido e o harness funciona, não que o modelo é bom. A segunda afirmação exige a run de treino.
