FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Tesseract 5 + Persian model (tessdata_best for accuracy), fonts for the synthetic tests
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng ca-certificates curl fonts-dejavu-core \
    && TD=$(dirname "$(find /usr/share/tesseract-ocr -name eng.traineddata | head -1)") \
    && curl -fsSL -o "$TD/fas.traineddata" https://github.com/tesseract-ocr/tessdata_best/raw/main/fas.traineddata \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
# optional: HEIC support
RUN pip install pillow-heif || true

COPY receipt_reader ./receipt_reader

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/v1/health').status==200 else 1)"
CMD ["uvicorn", "receipt_reader.api:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8080", "--workers", "2", "--timeout-keep-alive", "30"]
