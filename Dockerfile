FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NANOROUTER_SPEC=/app/configs/tools.yaml \
    NANOROUTER_ADAPTER=/app/checkpoints/lora/final

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

# O torch do PyPI padrão carrega CUDA e multiplica a imagem por cinco. O wheel
# de CPU vem do índice oficial da PyTorch e é instalado antes do resto, para o
# resolvedor não puxar a variante com GPU.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -e ".[serve]" "transformers>=4.44" "peft>=0.11"

COPY configs ./configs

# O adapter não entra na imagem: `make fetch-adapter` baixa em ./checkpoints e o
# docker-compose monta como volume. Sem o volume, o servidor sobe no modelo base
# e o /health mostra o backend efetivo.
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=90s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "nanorouter.server:app", "--host", "0.0.0.0", "--port", "8000"]
