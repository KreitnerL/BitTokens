# Frontier Model Numeracy Benchmark — Reproduction and Analysis

This folder contains the code used to generate, collect, and analyze frontier model results reported in the paper (ICLR submission). It supports launching API requests to multiple model providers (via OpenRouter and Google GenAI/Gemini), saving standardized CSV outputs, and producing all figures and tables used in the paper.

- Raw result generation: `launch_openrouter_requests.py`
- Example mass launch: `launch_frontier_baselines.sh`
- Analysis and figure/table generation: `evaluate_frontier_outputs.py`
- Final “BitToken benchmark” figure: `generate_bittokens_benchmark_plot.py`
- Outputs: `results/` (raw CSVs) and `results/analysis/` (plots, tables)

## Contents

- `launch_openrouter_requests.py`: CLI to query models across all benchmark tasks and store results in CSV form. Handles retries, resume, reasoning controls, and Gemini batch submission.
- `evaluate_frontier_outputs.py`: Parses CSVs, computes metrics, and generates figures (PDF/PNG) and LaTeX/CSV tables in `results/analysis/`.
- `launch_frontier_baselines.sh`: Example script to launch all models and reasoning settings over all tasks.
- `generate_bittokens_benchmark_plot.py`: Generates the final paper plot from curated `analysis/multitask.csv` and `analysis/solotask.csv`.
- `utils.py`: Shared utilities for prompting and parsing.

## Setup

This code uses the same environment as the main project.

- Credentials (supported sources: env vars or `.env` file):
  - `OPENROUTER_API_KEY`: required to call models via OpenRouter (`openrouter.ai`)
  - `GEMINI_API_KEY` or `GOOGLE_API_KEY`: required for Google GenAI (Gemini) models
  - Optional: `PROJECT_PATH` (defaults to `.`); used by `evaluate_frontier_outputs.py` to resolve `results/`

Create a `.env` file if convenient:
```bash
echo 'OPENROUTER_API_KEY=...'
echo 'GEMINI_API_KEY=...'
# or: GOOGLE_API_KEY=...
echo 'PROJECT_PATH=.'
```

## Data layout and file naming

Place benchmark input CSVs in `results/`. Each task’s input file follows:
- `<Task>_decimal_uniform_test_500_shuffle.csv` (e.g., `Addition_decimal_uniform_test_500_shuffle.csv`)

When you run evaluations, a per-(task, model, reasoning) CSV is created in the same folder:
- `<Task>_decimal_uniform_test_500_shuffle_<model>_re_<reasoning>.csv`
  - Example: `Addition_decimal_uniform_test_500_shuffle_google__gemini-2.5-pro_re_minimal.csv`

These file names are used by the analysis scripts to discover and aggregate results.

## Supported tasks

- Regression: `Addition`, `Division`, `Exponentiation`, `Mean`, `Multiplication`, `Std`
- Classification: `Interval`, `MinMax`, `Sorting`

## Supported models (as configured)

- `openai/gpt-5`
- `google/gemini-2.5-pro`
- `qwen/qwen3-235b-a22b-thinking-2507`
- `openai/gpt-oss-120b`
- `deepseek/deepseek-chat-v3.1`
- `qwen/qwen3-235b-a22b-2507`
- `google/gemini-2.5-flash`
- `moonshotai/kimi-k2-0905`
- `meta-llama/llama-4-maverick`

Notes:
- Reasoning controls differ by provider. The script normalizes unsupported settings; see “Reasoning effort” below.

## Step 1 — Generate raw outputs

Single-job example:
```bash
uv run frontier_model_analysis/launch_openrouter_requests.py \
  --input_file results/Addition_decimal_uniform_test_500_shuffle.csv \
  --model_name google/gemini-2.5-pro \
  --reasoning_effort none \
  --use_batch --batch_size 1 --batch_concurrency 10 \
  --save_interval 10 --max_retries 20 --verbose
```

- Required: `--input_file` path to a task CSV
- Important flags:
  - `--model_name`: one of the supported models (see list above)
  - `--reasoning_effort`: choices declared in parser, but internally validated as `high` or `none`
    - The script standardizes unsupported settings:
      - For `openai/gpt-5` and `google/gemini-2.5-pro`, `none` is converted to `minimal`
      - Some models force `none` or `high` depending on capabilities (handled automatically)
  - `--few_shot` and `--fs_examples N`: enable and control few-shot examples
  - `--resume`/`--no-resume`: auto-resume appends to the same CSV by skipping already-populated rows
  - `--use_batch`, `--batch_size`, `--batch_concurrency`: for Gemini batch mode (recommended)
  - Throttling/retry: `--max_retries`, `--backoff_base`, `--backoff_max`, `--jitter`, `--min_request_interval`
  - `--run_id`: optional suffix to disambiguate output files
  - `--limit`: upper bound on additional rows processed in this run (helpful for smoke tests)

Gemini-specific recommendation (as used in the paper):
- Use `--use_batch --batch_size 1 --batch_concurrency 10`

Mass launch (example script):
```bash
bash frontier_model_analysis/launch_frontier_baselines.sh
```
- Adjust the internal command to match your environment (use `launch_openrouter_requests.py` if your repo does not have `openrouter_eval.py`).

Outputs:
- CSVs are written under `results/`, following the standardized naming pattern.
- The script is robust to empty `raw_response` rows; it will retry and upsert new content.

## Step 2 — Evaluate and generate figures/tables

Run from the repo root so that `evaluate_frontier_outputs.py` can import `eval_scripts.utils`:
```bash
uv run frontier_model_analysis/evaluate_frontier_outputs.py \
  --data_dir results \
  --out_dir analysis \
  --save_cleaned \
  --limit_n 500
```

Key flags:
- `--data_dir`: where result CSVs live (default: `$PROJECT_PATH/results/`)
- `--out_dir`: subfolder created inside `data_dir` for outputs (default: `analysis`)
- `--save_cleaned`: write cleaned per-file CSVs with robust parsed answers and metrics
- `--load_saved`: skip recomputation and load `combined_samples_long.csv` if present
- `--individual_difficulty`: also produce per-task difficulty plots
- `--limit_n`: evaluate only the first N rows per file (default: 500)

Outputs (in `results/analysis/`):
- Combined data: `combined_samples_long.csv`
- Plots:
  - `radar_models.pdf`
  - `difficulty_vs_performance_matrix.pdf`
  - `tokens_vs_performance_matrix.pdf` and `.png`
  - `samples_vs_performance_matrix.pdf` and `.png`
- Tables (LaTeX and CSV):
  - `table_performance_by_task.tex` / `.csv`
  - `table_tokens_by_task.tex` / `.csv`
  - `table_regression_exact15_by_task.tex` / `.csv`
- Summaries:
  - `perf_by_model_reasoning_task.csv`
  - `avg_tokens_perf_by_task_model_reasoning.csv`
  - `coverage_by_model_reasoning.csv`
  - `overview_per_model_by_reasoning.csv`
- Parser diagnostics (if any): `invalid_json_rows.csv`
- Reasoning tool mentions (if detected): `reasoning_tool_mentions.csv`

Notes:
- The analysis script filters/remaps specific models (e.g., normalizes reasoning labels) to match the paper setup.
- Ensure the repository root contains `eval_scripts/utils.py` (imported by `evaluate_frontier_outputs.py`). If you are running the script outside the repo root, add the repo root to `PYTHONPATH` or run from the root.

## Step 3 — Final paper plot (BitToken benchmark)

This separate script expects curated inputs in `analysis/`:
- `analysis/multitask.csv`
- `analysis/solotask.csv`

Generate the plot:
```bash
uv run frontier_model_analysis/generate_bittokens_benchmark_plot.py
```

Outputs:
- `analysis/results.pdf`
- `analysis/results.png`

## Reasoning effort and normalization

- Not all models support explicit reasoning controls. The launcher normalizes requests:
  - If a model does not support `--reasoning_effort`, it standardizes to the closest supported behavior (e.g., some models force `none`, others `high`; Gemini/GPT-5 convert `none` to `minimal`).
- Token accounting:
  - For Gemini “none” on `gemini-2.5-flash`, extreme token counts (>200) are excluded from average token summaries in analysis (see code comments).

## Tips and troubleshooting

- Rate limits/transient errors: the launcher retries with exponential backoff; adjust `--max_retries`, `--backoff_*`, `--jitter`.
- Empty responses: the launcher retries and, for batch mode, upserts fixed rows by `text_prompt`.
- Resuming: `--resume` skips rows already having non-empty `raw_response` in the existing output file.
- Determinism: set `--seed` for jitter and sampling determinism where applicable.
- Performance metric:
  - Regression tasks: log-sMAPE accuracy in [0,1]
  - Classification tasks: exact match
  - Additional exact-15 significant digits accuracy is computed for regression tasks in the tables.

## Citation

If you use this code or benchmark results, please cite the associated paper (details to be provided after review).

## License

TBD. For review purposes, please use this code as-is for reproducibility and analysis.