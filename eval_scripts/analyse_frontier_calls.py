#!/usr/bin/env python3
"""
Analyze OpenRouter eval trial run: completion checks and usage plots.

- Validates that each (task, model[, reasoning_effort]) has 10 results
- Aggregates usage metrics across tasks and per task
- Saves combined CSV and plots

Usage:
  python analyze_frontier_outputs.py \
    --base-dir [PROJECT_PATH]/test_data_2025-09-04_shuffled \
    --out-dir  [PROJECT_PATH]/frontier_outputs_analysis

"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()
import os  # noqa: E402

PROJECT_PATH = os.getenv("PROJECT_PATH")


@dataclass(frozen=True)
class FileMeta:
    file_path: Path
    task: str
    model_safe: str
    model_name: str
    reasoning_effort: str


EXPECTED_PER_GROUP = 1000  # per run in this trial


def find_result_files(base_dir: Path) -> List[Path]:
    """
    Return all CSV result files that look like model outputs: *_re_*.csv
    Excludes source input CSVs (which lack the _re_ segment).
    """
    return sorted(base_dir.rglob("*_re_*.csv"))


def parse_meta_from_filename(p: Path) -> FileMeta:
    """
    Files are named as:
      <TaskPrefix>..._<safe_model>_re_<reasoning>.csv
    where safe_model is original model with '/' -> '__' and ':' -> '_'.

    We infer the task from the filename stem (prefix before the first underscore),
    since files may no longer be nested in per-task subdirectories.
    """
    stem = p.stem  # e.g., Std_decimal_uniform_test_10_openai__gpt-5_re_high
    # Infer task as the first token before the first underscore
    if "_" in stem:
        task = stem.split("_", 1)[0]
    else:
        task = stem  # fallback: entire stem if no underscore
    # Split on the trailing `_re_...`
    if "_re_" not in stem:
        raise ValueError(f"Unexpected file stem (no _re_): {stem}")
    left, reasoning_effort = stem.rsplit("_re_", 1)

    # The safe_model is the segment after the dataset prefix; it’s the last underscore-separated token in `left`
    # However dataset prefixes can contain underscores, so split from the right once.
    # Example: left = "Std_decimal_uniform_test_10_openai__gpt-5"
    if "_" not in left:
        raise ValueError(f"Unexpected file stem (no model segment): {stem}")
    dataset_prefix, model_safe = left.rsplit("_", 1)

    # Reconstruct model name: reverse '/' mapping; keep underscores (':' mapping) as-is since originals rarely contain ':'
    model_name = model_safe.replace("__", "/")
    return FileMeta(
        file_path=p,
        task=task,
        model_safe=model_safe,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
    )


def load_results(files: Iterable[Path]) -> pd.DataFrame:
    """
    Load and concatenate result CSVs, attaching parsed metadata columns.
    Coerce usage metrics to numeric (NaN on failure).
    """
    records: List[pd.DataFrame] = []
    for fp in files:
        meta = parse_meta_from_filename(fp)
        try:
            df = pd.read_csv(fp, dtype=str)
        except Exception as e:
            print(f"Warning: failed to read {fp}: {e}")
            continue
        df["task"] = meta.task
        df["model_safe"] = meta.model_safe
        df["model_name"] = meta.model_name
        df["reasoning_effort"] = meta.reasoning_effort
        df["file_path"] = str(fp)

        # Numeric coercions for usage metrics and accuracy
        for col in ("usage.total_tokens", "usage.cost", "logSMAPE_acc"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = np.nan

        records.append(df)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def check_completion(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return two reports:
      - by (task, model_name, reasoning_effort): expected 10 each
      - by (task, model_name): expected 10 (if you intended one run) or 20+ if multiple reasoning runs exist
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    grp_cols_full = ["task", "model_name", "reasoning_effort"]
    s_full = df.groupby(grp_cols_full, dropna=False).size()
    counts_full = s_full.reset_index()
    counts_full = counts_full.rename(columns={0: "n_rows", "size": "n_rows"})
    counts_full = counts_full.sort_values(grp_cols_full)
    counts_full["ok"] = counts_full["n_rows"] == EXPECTED_PER_GROUP

    grp_cols_model = ["task", "model_name"]
    s_model = df.groupby(grp_cols_model, dropna=False).size()
    counts_model = s_model.reset_index()
    counts_model = counts_model.rename(columns={0: "n_rows", "size": "n_rows"})
    counts_model = counts_model.sort_values(grp_cols_model)
    # We don’t enforce 10 here because some tasks were run multiple times by design.
    # Still flag groups below 10 to surface incomplete runs.
    counts_model["ok_at_least_10"] = counts_model["n_rows"] >= EXPECTED_PER_GROUP

    return counts_full, counts_model


def plot_overall_per_model(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Plot mean usage.cost and usage.total_tokens per model over all tasks/runs.
    """
    if df.empty:
        return
    metrics = ["usage.cost", "usage.total_tokens"]
    agg = (
        df.groupby("model_name", dropna=False)[metrics]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("model_name")
    )
    for metric in metrics:
        plt.figure(figsize=(10, 4))
        sns.barplot(data=agg, x="model_name", y=metric, color="#4C78A8")
        plt.title(f"Mean {metric} per model (all tasks)")
        plt.xlabel("Model")
        plt.ylabel(f"Mean {metric}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out_path = out_dir / f"mean_{metric.replace('.', '_')}_per_model.png"
        plt.savefig(out_path, dpi=150)
        plt.close()


def plot_per_task_per_model(df: pd.DataFrame, out_dir: Path) -> None:
    """
    For each task, plot mean usage metrics per model.
    """
    if df.empty:
        return
    metrics = ["usage.cost", "usage.total_tokens"]
    for task, df_task in df.groupby("task"):
        agg = (
            df_task.groupby("model_name", dropna=False)[metrics]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("model_name")
        )
        for metric in metrics:
            plt.figure(figsize=(10, 4))
            sns.barplot(data=agg, x="model_name", y=metric, color="#F58518")
            plt.title(f"{task}: mean {metric} per model")
            plt.xlabel("Model")
            plt.ylabel(f"Mean {metric}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            out_path = out_dir / f"{task}_mean_{metric.replace('.', '_')}_per_model.png"
            plt.savefig(out_path, dpi=150)
            plt.close()


def save_reports(
    df_all: pd.DataFrame,
    counts_full: pd.DataFrame,
    counts_model: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Save combined data and completion reports to disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = out_dir / "combined_results.csv"
    df_all.to_csv(combined_csv, index=False)
    if not counts_full.empty:
        counts_full.to_csv(
            out_dir / "completion_by_task_model_reasoning.csv", index=False
        )
    if not counts_model.empty:
        counts_model.to_csv(out_dir / "completion_by_task_model.csv", index=False)


def print_completion_summary(
    counts_full: pd.DataFrame, counts_model: pd.DataFrame
) -> None:
    """
    Print concise summaries for quick inspection.
    """
    if counts_full.empty and counts_model.empty:
        print("No data loaded.")
        return

    if not counts_full.empty:
        total_groups = len(counts_full)
        ok_groups = int(counts_full["ok"].sum())
        print(
            f"[per task+model+reasoning] groups OK: {ok_groups}/{total_groups} (expected exactly {EXPECTED_PER_GROUP} rows each)"
        )
        incomplete = counts_full[~counts_full["ok"]]
        if not incomplete.empty:
            print("Incomplete groups (task, model, reasoning, n_rows):")
            for row in incomplete.itertuples(index=False):
                print(
                    f"  {row.task}, {row.model_name}, {row.reasoning_effort}: {row.n_rows}"
                )

    if not counts_model.empty:
        total_groups = len(counts_model)
        ok_groups = int(counts_model["ok_at_least_10"].sum())
        print(
            f"[per task+model] groups with at least {EXPECTED_PER_GROUP}: {ok_groups}/{total_groups}"
        )
        too_small = counts_model[~counts_model["ok_at_least_10"]]
        if not too_small.empty:
            print("Groups with < 10 rows (task, model, n_rows):")
            for row in too_small.itertuples(index=False):
                print(f"  {row.task}, {row.model_name}: {row.n_rows}")


def _is_strict_json_object(text: str) -> bool:
    """Return True if text is exactly a JSON object string.

    Strictness criteria:
    - Must be a string that, after stripping whitespace, starts with '{' and ends with '}'.
    - json.loads must succeed and yield a dict.
    This flags any extra prose around JSON, arrays, or invalid JSON as non-strict.
    """
    if not isinstance(text, str):
        return False
    trimmed = text.strip()
    if not (trimmed.startswith("{") and trimmed.endswith("}")):
        return False
    try:
        obj = json.loads(trimmed)
    except Exception:
        return False
    return isinstance(obj, dict)


def invalid_json_reports(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute reports of invalid JSON raw responses.

    Returns two DataFrames filtered to groups with at least one invalid row:
    - by (task, model_name, reasoning_effort) with columns [n_rows, n_invalid, frac_invalid]
    - by (task, model_name) with columns [n_rows, n_invalid, frac_invalid]
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "raw_response" in df.columns:
        invalid_mask = ~df["raw_response"].apply(_is_strict_json_object)
    else:
        # If column missing, consider all invalid
        invalid_mask = pd.Series(True, index=df.index)

    df2 = df.assign(invalid_json=invalid_mask)

    grp_full_cols = ["task", "model_name", "reasoning_effort"]
    by_full = (
        df2.groupby(grp_full_cols, dropna=False)
        .agg(n_rows=("raw_response", "size"), n_invalid=("invalid_json", "sum"))
        .reset_index()
    )
    by_full = by_full[by_full["n_invalid"] > 0].copy()
    if not by_full.empty:
        by_full["frac_invalid"] = by_full["n_invalid"] / by_full["n_rows"].replace(
            0, np.nan
        )

    grp_model_cols = ["task", "model_name"]
    by_model = (
        df2.groupby(grp_model_cols, dropna=False)
        .agg(n_rows=("raw_response", "size"), n_invalid=("invalid_json", "sum"))
        .reset_index()
    )
    by_model = by_model[by_model["n_invalid"] > 0].copy()
    if not by_model.empty:
        by_model["frac_invalid"] = by_model["n_invalid"] / by_model["n_rows"].replace(
            0, np.nan
        )

    return by_full.sort_values(grp_full_cols), by_model.sort_values(grp_model_cols)


def print_invalid_json_summary(by_full: pd.DataFrame, by_model: pd.DataFrame) -> None:
    """Print concise summaries of groups with invalid JSON responses."""
    if by_full.empty and by_model.empty:
        print("All raw responses are strict JSON objects (by inspected groups).")
        return

    if not by_full.empty:
        print("Groups with invalid JSON (per task+model+reasoning):")
        for rec in by_full.to_dict(orient="records"):
            total = int(rec.get("n_rows", 0) or 0)
            bad = int(rec.get("n_invalid", 0) or 0)
            frac = (bad / total) if total else 0.0
            print(
                f"  {rec.get('task')}, {rec.get('model_name')}, {rec.get('reasoning_effort')}: {bad}/{total} ({frac:.1%}) invalid"
            )

    if not by_model.empty:
        print("Groups with invalid JSON (per task+model):")
        for rec in by_model.to_dict(orient="records"):
            total = int(rec.get("n_rows", 0) or 0)
            bad = int(rec.get("n_invalid", 0) or 0)
            frac = (bad / total) if total else 0.0
            print(
                f"  {rec.get('task')}, {rec.get('model_name')}: {bad}/{total} ({frac:.1%}) invalid"
            )


def make_overview_per_model(
    df: pd.DataFrame, out_dir: Path, target_samples: int = 10_000
) -> None:
    """Create an overview table and a single combined plot for cost and tokens per model.

    The table includes:
      - n_samples: number of rows observed for the model
      - mean_usage_cost, sum_usage_cost
      - mean_usage_total_tokens, sum_usage_total_tokens
      - est_cost_for_target_samples, est_tokens_for_target_samples (linear scaling from per-sample means)

    Also saves a single twin-axis bar plot with cost (left axis) and tokens (right axis) per model.
    """
    if df.empty:
        return

    metrics = ["usage.cost", "usage.total_tokens"]
    for col in metrics:
        if col not in df.columns:
            df[col] = np.nan

    # Aggregate per model
    grp = df.groupby("model_name", dropna=False)
    agg = grp.agg(
        n_samples=("raw_response", "size"),
        mean_usage_cost=("usage.cost", "mean"),
        sum_usage_cost=("usage.cost", "sum"),
        mean_usage_total_tokens=("usage.total_tokens", "mean"),
        sum_usage_total_tokens=("usage.total_tokens", "sum"),
        mean_logSMAPE_acc=("logSMAPE_acc", "mean"),
    ).reset_index()

    # Replace NaNs with 0 for display and scaling
    agg["mean_usage_cost"] = agg["mean_usage_cost"].fillna(0.0)
    agg["sum_usage_cost"] = agg["sum_usage_cost"].fillna(0.0)
    agg["mean_usage_total_tokens"] = agg["mean_usage_total_tokens"].fillna(0.0)
    agg["sum_usage_total_tokens"] = agg["sum_usage_total_tokens"].fillna(0.0)

    # Linear scaling to target_samples based on per-sample mean
    agg["est_cost_for_target_samples"] = agg["mean_usage_cost"] * float(target_samples)
    agg["est_tokens_for_target_samples"] = agg["mean_usage_total_tokens"] * float(
        target_samples
    )

    # Save overview table
    out_dir.mkdir(parents=True, exist_ok=True)
    overview_csv = out_dir / "overview_per_model.csv"
    agg.to_csv(overview_csv, index=False)

    # Single combined figure with three subplots: cost, tokens, logSMAPE
    models = agg["model_name"].tolist()
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # Cost subplot
    axes[0].bar(x, agg["mean_usage_cost"].astype(float), color="#4C78A8")
    axes[0].set_title("Mean cost ($)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha="right")
    axes[0].set_ylabel("$ per sample")
    # Tokens subplot
    axes[1].bar(x, agg["mean_usage_total_tokens"].astype(float), color="#F58518")
    axes[1].set_title("Mean tokens")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha="right")
    # logSMAPE subplot (may contain NaNs)
    axes[2].bar(x, agg["mean_logSMAPE_acc"].astype(float).fillna(0.0), color="#54A24B")
    axes[2].set_title("Mean logSMAPE_acc (regression)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45, ha="right")
    axes[2].set_ylim(0.0, 1.0)

    fig.suptitle(
        f"Per-model means: cost, tokens, logSMAPE; {target_samples}x scaling in CSV",
        y=1.02,
    )
    fig.tight_layout()
    out_plot = out_dir / "overview_allmetrics_per_model.png"
    fig.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_overview_by_task_model(
    df: pd.DataFrame, out_dir: Path, target_samples: int = 10_000
) -> None:
    """Create a wide summary by (task, model) including 10k-sample scaling; saved as CSV.

    This is a table-only companion to the per-model overview, useful if you want per-task scaling.
    """
    if df.empty:
        return

    for col in ("usage.cost", "usage.total_tokens"):
        if col not in df.columns:
            df[col] = np.nan

    grp_cols = ["task", "model_name"]
    agg = (
        df.groupby(grp_cols, dropna=False)
        .agg(
            n_samples=("raw_response", "size"),
            mean_usage_cost=("usage.cost", "mean"),
            sum_usage_cost=("usage.cost", "sum"),
            mean_usage_total_tokens=("usage.total_tokens", "mean"),
            sum_usage_total_tokens=("usage.total_tokens", "sum"),
            mean_logSMAPE_acc=("logSMAPE_acc", "mean"),
        )
        .reset_index()
    )

    agg["mean_usage_cost"] = agg["mean_usage_cost"].fillna(0.0)
    agg["sum_usage_cost"] = agg["sum_usage_cost"].fillna(0.0)
    agg["mean_usage_total_tokens"] = agg["mean_usage_total_tokens"].fillna(0.0)
    agg["sum_usage_total_tokens"] = agg["sum_usage_total_tokens"].fillna(0.0)

    agg["est_cost_for_target_samples"] = agg["mean_usage_cost"] * float(target_samples)
    agg["est_tokens_for_target_samples"] = agg["mean_usage_total_tokens"] * float(
        target_samples
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_dir / "overview_by_task_model.csv", index=False)


def write_invalid_json_rows(df: pd.DataFrame, out_dir: Path) -> None:
    """Write a CSV with all rows whose raw_response is not a strict JSON object.

    Ensures `model_name` and `task` columns are present and placed first in the output.
    """
    if df.empty:
        return
    if "raw_response" not in df.columns:
        return

    invalid_mask = ~df["raw_response"].apply(_is_strict_json_object)
    invalid_rows = df[invalid_mask].copy()
    if invalid_rows.empty:
        # Nothing to write
        return

    # Ensure required columns exist
    for col in ("model_name", "task"):
        if col not in invalid_rows.columns:
            invalid_rows[col] = ""

    # Ensure reasoning column exists as well
    if "reasoning_effort" not in invalid_rows.columns:
        invalid_rows["reasoning_effort"] = ""

    # Reorder to show model, task, reasoning first
    ordered_cols = ["model_name", "task", "reasoning_effort"] + [
        c
        for c in invalid_rows.columns
        if c not in ("model_name", "task", "reasoning_effort")
    ]
    invalid_rows = invalid_rows[ordered_cols]

    out_dir.mkdir(parents=True, exist_ok=True)
    invalid_rows.to_csv(out_dir / "invalid_json_rows.csv", index=False)


def make_overview_per_model_by_reasoning(
    df: pd.DataFrame, out_dir: Path, target_samples: int = 10_000
) -> None:
    """Per-model, split by reasoning_effort: table and per-reasoning combined cost/tokens plot.

    Writes:
      - overview_per_model_by_reasoning.csv
      - overview_cost_tokens_per_model_re_<reasoning>.png (one per reasoning value)
    """
    if df.empty:
        return
    for col in ("usage.cost", "usage.total_tokens"):
        if col not in df.columns:
            df[col] = np.nan

    grp_cols = ["model_name", "reasoning_effort"]
    agg = (
        df.groupby(grp_cols, dropna=False)
        .agg(
            n_samples=("raw_response", "size"),
            mean_usage_cost=("usage.cost", "mean"),
            sum_usage_cost=("usage.cost", "sum"),
            mean_usage_total_tokens=("usage.total_tokens", "mean"),
            sum_usage_total_tokens=("usage.total_tokens", "sum"),
            mean_logSMAPE_acc=("logSMAPE_acc", "mean"),
        )
        .reset_index()
    )
    agg["mean_usage_cost"] = agg["mean_usage_cost"].fillna(0.0)
    agg["sum_usage_cost"] = agg["sum_usage_cost"].fillna(0.0)
    agg["mean_usage_total_tokens"] = agg["mean_usage_total_tokens"].fillna(0.0)
    agg["sum_usage_total_tokens"] = agg["sum_usage_total_tokens"].fillna(0.0)

    agg["est_cost_for_target_samples"] = agg["mean_usage_cost"] * float(target_samples)
    agg["est_tokens_for_target_samples"] = agg["mean_usage_total_tokens"] * float(
        target_samples
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_dir / "overview_per_model_by_reasoning.csv", index=False)

    # Plots per reasoning: single figure with 3 subplots (cost, tokens, logSMAPE)
    for reasoning, df_r in agg.groupby("reasoning_effort", dropna=False):
        models = df_r["model_name"].tolist()
        x = np.arange(len(models))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        # Cost
        axes[0].bar(x, df_r["mean_usage_cost"].astype(float), color="#4C78A8")
        axes[0].set_title("Mean cost ($)")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha="right")
        axes[0].set_ylabel("$ per sample")
        # Tokens
        axes[1].bar(x, df_r["mean_usage_total_tokens"].astype(float), color="#F58518")
        axes[1].set_title("Mean tokens")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=45, ha="right")
        # logSMAPE
        axes[2].bar(
            x, df_r["mean_logSMAPE_acc"].astype(float).fillna(0.0), color="#54A24B"
        )
        axes[2].set_title("Mean logSMAPE_acc (regression)")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(models, rotation=45, ha="right")
        axes[2].set_ylim(0.0, 1.0)
        title_re = str(reasoning) if reasoning is not None else "None"
        fig.suptitle(
            f"Per-model means by reasoning={title_re}: cost, tokens, logSMAPE", y=1.02
        )
        fig.tight_layout()
        out_plot = out_dir / f"overview_allmetrics_per_model_re_{title_re}.png"
        fig.savefig(out_plot, dpi=150, bbox_inches="tight")
        plt.close(fig)


def make_overview_by_task_model_reasoning(
    df: pd.DataFrame, out_dir: Path, target_samples: int = 10_000
) -> None:
    """Table summary grouped by (task, model_name, reasoning_effort) with 10k scaling.

    Writes: overview_by_task_model_reasoning.csv
    """
    if df.empty:
        return
    for col in ("usage.cost", "usage.total_tokens"):
        if col not in df.columns:
            df[col] = np.nan

    grp_cols = ["task", "model_name", "reasoning_effort"]
    agg = (
        df.groupby(grp_cols, dropna=False)
        .agg(
            n_samples=("raw_response", "size"),
            mean_usage_cost=("usage.cost", "mean"),
            sum_usage_cost=("usage.cost", "sum"),
            mean_usage_total_tokens=("usage.total_tokens", "mean"),
            sum_usage_total_tokens=("usage.total_tokens", "sum"),
            mean_logSMAPE_acc=("logSMAPE_acc", "mean"),
        )
        .reset_index()
    )
    agg["mean_usage_cost"] = agg["mean_usage_cost"].fillna(0.0)
    agg["sum_usage_cost"] = agg["sum_usage_cost"].fillna(0.0)
    agg["mean_usage_total_tokens"] = agg["mean_usage_total_tokens"].fillna(0.0)
    agg["sum_usage_total_tokens"] = agg["sum_usage_total_tokens"].fillna(0.0)

    agg["est_cost_for_target_samples"] = agg["mean_usage_cost"] * float(target_samples)
    agg["est_tokens_for_target_samples"] = agg["mean_usage_total_tokens"] * float(
        target_samples
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_dir / "overview_by_task_model_reasoning.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=f"{PROJECT_PATH}/test_data_2025-09-04_shuffled_correct/",
        help="Base directory containing per-task subdirectories",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=f"{PROJECT_PATH}/test_data_2025-09-04_shuffled_correct/analysis",
        help="Output directory for combined CSV and plots",
    )
    args = parser.parse_args()

    files = find_result_files(args.base_dir)
    if not files:
        print(f"No result files found under {args.base_dir} matching '*_re_*.csv'")
        return

    df_all = load_results(files)
    counts_full, counts_model = check_completion(df_all)
    print_completion_summary(counts_full, counts_model)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_reports(df_all, counts_full, counts_model, args.out_dir)

    # Check for non-JSON / mixed-prose raw responses and report
    by_full_invalid, by_model_invalid = invalid_json_reports(df_all)
    print_invalid_json_summary(by_full_invalid, by_model_invalid)
    if not by_full_invalid.empty:
        by_full_invalid.to_csv(
            args.out_dir / "invalid_json_by_task_model_reasoning.csv", index=False
        )
    if not by_model_invalid.empty:
        by_model_invalid.to_csv(
            args.out_dir / "invalid_json_by_task_model.csv", index=False
        )

    # Detailed rows file for invalid JSON responses
    write_invalid_json_rows(df_all, args.out_dir)

    # Overview tables and combined plot, including 10k scaling
    make_overview_per_model(df_all, args.out_dir, target_samples=10_000)
    make_overview_by_task_model(df_all, args.out_dir, target_samples=10_000)
    make_overview_per_model_by_reasoning(df_all, args.out_dir, target_samples=10_000)
    make_overview_by_task_model_reasoning(df_all, args.out_dir, target_samples=10_000)

    # plot_overall_per_model(df_all, args.out_dir)
    # plot_per_task_per_model(df_all, args.out_dir)

    print(f"Wrote combined CSV and plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
