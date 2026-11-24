# Método

## Por que roteamento e não geração livre

Roteamento tem três propriedades que tornam fine-tuning viável e mensurável: espaço de saída pequeno e fechado, critério de acerto objetivo e distribuição de entrada estreita. Geração livre falha nos três, e por isso vira discussão de opinião em vez de medição.

## Construção do dataset

`src/nanorouter/synth.py` gera por três caminhos.

**Templates.** Quatro a seis paráfrases por ferramenta, em pt-BR, en e es. Cobre variação lexical sem custo de anotação.

**Fuzzing de slots.** Valores amostrados por tipo, dentro das faixas válidas do spec (`passengers` entre 1 e 9, datas num horizonte de 120 dias, IATA e moeda em ISO de três letras).

**Hard negatives.** Enunciados em que a superfície lexical aponta para uma ferramenta e a intenção para outra: `"tem voo direto de São Paulo para Lisboa?"` é `search_flights` mesmo citando clima na sequência, e `"o BK1042 já foi emitido?"` é `booking_status`, não `cancel_booking`. São 18% do dataset porque são a maior fatia dos erros em produção.

Os hard negatives são **templates parametrizados**, não frases fixas. A primeira versão tinha dez frases literais repetidas cerca de cem vezes cada: o dataset passava a ter 6.000 registros e apenas 3.852 enunciados distintos, e o split de teste virava um teste de memorização. Parametrizar cidade, data, código de reserva e moeda subiu a diversidade para 4.556 enunciados distintos e é o que faz o número de teste significar algo.

Split estratificado por `kind`, 80/10/10, semente fixa. O split é por registro, não por template: templates se repetem entre splits de propósito, para medir generalização lexical e não memorização de superfície.

## Treino

LoRA r=16, alpha=32, em todas as projeções lineares. `completion_only_loss=True` mascara o prompt de sistema; sem isso, a maior parte dos tokens de loss é texto fixo e o adapter aprende o catálogo de ferramentas de cor, não a tarefa. Três épocas, cosine, warmup 3%, lr 2e-4, bf16 quando disponível e `packing=False` (as sequências são curtas e o packing mistura fronteiras de exemplo).

O diretório do checkpoint grava `spec_fingerprint.json` com o hash do spec, a semente e o número de registros: um adapter sem o spec correspondente não é reproduzível.

## Avaliação

Cinco métricas independentes, todas determinísticas. `spec_validity` é a que mais importa em produção: um modelo com `exact_match` alto e `spec_validity` baixo quebra o parser de quem chama.

Baselines rodam no mesmo harness, com o mesmo prompt e a mesma suite. O few-shot usa quatro exemplos do split de validação, nunca do teste. O intervalo de confiança de 95% vem de bootstrap sobre os acertos do `exact_match`.

## Decisões que ficaram de fora

- **Constrained decoding (Outlines, XGrammar).** Torna o JSON sempre válido e teria levado o `spec_validity` a 1,0 em todos os backends. Ficou fora do número principal para isolar o efeito do fine-tuning; está documentado como caminho de produção em [LIMITATIONS.md](LIMITATIONS.md).
- **Destilação de um modelo maior.** Provavelmente melhor teto, ao custo de manter um professor rodando.
- **DPO.** Só faz sentido depois de existirem pares de preferência reais de produção.
