# Limitações

**Dataset sintético.** Os templates são meus. Um domínio real traz construções que não previ: gíria, abreviação, erro de digitação, code-switching. Antes de produção, coletar 200 a 500 enunciados reais, avaliar e adicionar os erros ao dataset.

**Diversidade finita.** Mesmo com hard negatives parametrizados, 6.000 registros geram 4.556 enunciados distintos: cerca de 24% do dataset repete uma frase já vista com outro rótulo de slots. Isso é aceitável para treino e para a suite de teste, mas limita o que o `exact_match` no split de teste pode afirmar sobre generalização.

**Escala de catálogo.** Oito ferramentas cabem confortavelmente no prompt. A partir de cem, é preciso retrieval de ferramentas antes do roteador; o modelo não escala sozinho.

**`confidence` não é calibrada.** É um número aprendido, útil para ordenar, não para corte de limiar. Calibrar exige temperature scaling sobre um split separado.

**Duas intenções.** `"quero voo e hotel para Lisboa"` retorna uma ferramenta só. O formato é single-tool por design; multi-tool exige mudança de schema.

**Idioma.** Treinado em pt-BR, en e es. Fora disso, degrada.

**Sem constrained decoding no número principal.** Em produção, use gramática (Outlines, XGrammar) ou `response_format` com JSON Schema no servidor. Isso anula a classe inteira de erro de sintaxe — e é por isso que fica fora da comparação: mascararia a diferença que o fine-tuning produz.

**Dados de slot não são validados contra o mundo.** `GRU` é sintaticamente válido mesmo se o aeroporto não existir na data. A validação é de contrato, não de realidade.

**A run publicada é um piso, não o teto.** O adapter medido em `docs/RESULTS.md` viu 400 dos 4.800 registros de treino, em uma época, em CPU, a 11,6 s por registro. A configuração completa (4.800 registros, 3 épocas) roda em GPU em cerca de 20 minutos e deve produzir números melhores; o que está publicado é o que foi de fato executado.

**Amostra de avaliação.** A tabela usa 150 dos 601 itens de teste. O intervalo de confiança de 95% em torno de 0,587 tem cerca de 16 pontos percentuais de largura: diferenças grandes aparecem, diferenças pequenas não. Uma avaliação completa custa cerca de 50 minutos por backend em CPU.
