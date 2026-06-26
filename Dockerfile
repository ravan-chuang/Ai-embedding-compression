# Conda-forge provides a consistent Linux Faiss CPU build.
FROM condaforge/miniforge3:latest

WORKDIR /app

# pytrec-eval-terrier is pulled by BEIR while generating FiQA metadata.
# On Linux ARM it builds from source, so a C compiler is required.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create the native Faiss environment first so pip never installs a second Faiss build.
COPY environment.yml .
RUN mamba env create -f environment.yml && mamba clean --all --yes

ENV PATH="/opt/conda/envs/rag-api/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ARTIFACT_DIR="artifacts/fiqa_ivfpq_m96"

COPY requirements-api.txt .
RUN python -m pip install -r requirements-api.txt

COPY app ./app
COPY scripts ./scripts
COPY artifacts ./artifacts
COPY docker/entrypoint.sh /usr/local/bin/retrieval-api-entrypoint

RUN chmod +x /usr/local/bin/retrieval-api-entrypoint

EXPOSE 8000

# The first start can download FiQA metadata and the embedding model, so allow
# a longer grace period before health checks begin.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

ENTRYPOINT ["/usr/local/bin/retrieval-api-entrypoint"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
