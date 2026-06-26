#!/usr/bin/env bash
set -euo pipefail

metadata_path="${ARTIFACT_DIR}/documents.jsonl"

if [[ ! -s "${metadata_path}" ]]; then
  echo "FiQA metadata is missing; generating ${metadata_path} ..."
  python scripts/prepare_fiqa_documents.py
fi

exec "$@"
