#!/usr/bin/env bash
# Populates ./public/ next to bittokens_notebook.py so `marimo export html-wasm`
# copies Python helpers and tokenizer JSON into the static site (see marimo
# "public folder" behavior for WASM exports).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
rm -rf public
mkdir -p public/utils public/tokenizers/num_text
cp "$ROOT/utils/notebook_utils.py" "$ROOT/public/utils/"
touch "$ROOT/public/utils/__init__.py"
cp -a \
  "$ROOT/tokenizers/num_text/sd_gpt2" \
  "$ROOT/tokenizers/num_text/td_gpt2" \
  "$ROOT/tokenizers/num_text/bittoken_gpt2" \
  "$ROOT/public/tokenizers/num_text/"
