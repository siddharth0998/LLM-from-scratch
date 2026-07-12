FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache

WORKDIR /app

RUN pip install --no-cache-dir "torch>=2.0" \
      --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY configs ./configs
COPY checkpoints ./checkpoints

RUN mkdir -p "$TIKTOKEN_CACHE_DIR" \
 && python -c "import tiktoken; tiktoken.get_encoding('gpt2').encode('warmup')"

EXPOSE 7860

CMD ["python", "-m", "gpt2_chatbot.serving.app_gradio"]
