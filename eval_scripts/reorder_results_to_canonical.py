#!/usr/bin/env python3
"""
Reorder result CSV files to match the canonical dataset order by prompt.

Use case: Some batched/concurrent inference (e.g., Gemini 2.5 Flash High) may
write rows out of order. This script restores the correct order.

Behavior:
- For each results file matching
  <TASK>_decimal_uniform_test_10k_shuffle_<MODEL_SAFE>_re_<REASONING>.csv
  it loads the canonical file for the task:
  <TASK>_decimal_uniform_test_10k_shuffle.csv
- It reorders rows of the results to match the canonical order by the prompt
  column (prefers 'text_prompt', falls back to 'prompt').
- If duplicates exist in the results (same prompt value), it warns and keeps
  the first occurrence.
- Rows present in canonical but missing from results are reported as warnings
  and skipped.
- Rows present in results but missing from canonical are appended at the end
  in their original order and reported as warnings.
- Before writing, the original results file is backed up with suffix "_raw".

Example:
  python reorder_results_to_canonical.py \
    --data_dir $PROJECT_PATH/test_data_2025-09-04_shuffled_correct/

You can also restrict to specific files via --glob.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import dotenv
import pandas as pd

if True:
    dotenv.load_dotenv()  # Load environment variables from .env file
import os

PROJECT_PATH = os.getenv("PROJECT_PATH")

FILENAME_RE = re.compile(
    r"^(?P<task>[A-Za-z]+)_decimal_uniform_test_10k_shuffle_(?P<model>.+?)_re_(?P<reasoning>[A-Za-z0-9]+)\.csv$"
)


def find_prompt_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("text_prompt", "prompt"):
        if col in df.columns:
            return col
    return None


def backup_file(path: Path) -> Path:
    """Backup file by renaming to <stem>_raw.csv (or _raw.N.csv if exists)."""
    if not path.exists():
        return path
    base = path.with_name(path.stem + "_raw" + path.suffix)
    if not base.exists():
        path.rename(base)
        return base
    # Find next available suffix _raw.N
    n = 1
    while True:
        candidate = path.with_name(f"{path.stem}_raw.{n}{path.suffix}")
        if not candidate.exists():
            path.rename(candidate)
            return candidate
        n += 1


def reorder_one(results_path: Path, canonical_path: Path, warnings: List[str]) -> None:
    try:
        df_res = pd.read_csv(results_path, dtype=str)
    except Exception as e:
        warnings.append(f"Failed to read results file {results_path.name}: {e}")
        return
    if not canonical_path.exists():
        warnings.append(
            f"Canonical file not found for {results_path.name}: {canonical_path.name}"
        )
        return
    try:
        df_can = pd.read_csv(canonical_path, dtype=str)
    except Exception as e:
        warnings.append(f"Failed to read canonical file {canonical_path.name}: {e}")
        return

    key_res = find_prompt_column(df_res)
    key_can = find_prompt_column(df_can)
    if key_res is None or key_can is None:
        warnings.append(
            f"Missing prompt column ('text_prompt' or 'prompt') in {results_path.name} or {canonical_path.name}"
        )
        return

    # Normalize keys to strings and strip
    res_prompts: List[str] = df_res[key_res].astype(str).str.strip().tolist()
    can_prompts: List[str] = df_can[key_can].astype(str).str.strip().tolist()

    # Build first-occurrence mapping for results prompts
    prompt_to_index: Dict[str, int] = {}
    duplicate_counts: Dict[str, int] = {}
    for idx, p in enumerate(res_prompts):
        if p not in prompt_to_index:
            prompt_to_index[p] = idx
        else:
            duplicate_counts[p] = duplicate_counts.get(p, 0) + 1

    if duplicate_counts:
        dup_total = sum(duplicate_counts.values())
        warnings.append(
            f"{results_path.name}: found {len(duplicate_counts)} duplicate prompts ({dup_total} extra rows). Keeping first occurrences."
        )

    # Construct reordered indices following canonical order UNTIL first missing; truncate thereafter
    ordered_indices: List[int] = []
    first_missing_at: Optional[int] = None
    for pos, p in enumerate(can_prompts):
        idx = prompt_to_index.get(p)
        if idx is None:
            first_missing_at = pos
            break
        ordered_indices.append(idx)

    if first_missing_at is not None:
        remaining = len(can_prompts) - first_missing_at
        warnings.append(
            f"{results_path.name}: first missing canonical prompt at position {first_missing_at}; truncating output and skipping last {remaining} canonical entries."
        )

    # Build final DataFrame: matched in canonical order, then unmatched in original order
    df_matched = (
        df_res.iloc[ordered_indices].copy()
        if ordered_indices
        else df_res.iloc[0:0].copy()
    )
    # No appending of unmatched rows; we strictly output the contiguous canonical prefix found
    df_unmatched = df_res.iloc[0:0].copy()

    # Optional replacement for Gemini Flash High: overwrite canonical-leading fields up to 'answer'
    needs_replace = False
    do_replace = "google__gemini-2.5-flash_re_high" in results_path.name
    if do_replace and not df_matched.empty and "answer" in df_can.columns:
        # Align canonical rows to df_matched via prompt values
        matched_prompts = df_matched[key_res].astype(str).str.strip().tolist()
        can_index_by_prompt = {p: i for i, p in enumerate(can_prompts)}
        aligned_indices: List[int] = [
            can_index_by_prompt.get(p, -1) for p in matched_prompts
        ]
        aligned_indices = [i for i in aligned_indices if i >= 0]
        if len(aligned_indices) == len(df_matched):
            canonical_sub = df_can.iloc[aligned_indices].reset_index(drop=True)
            # Determine columns to copy: from start through 'answer' inclusive, intersect with results columns
            ans_pos = list(df_can.columns).index("answer")
            copy_cols = [
                c
                for c in list(df_can.columns)[: ans_pos + 1]
                if c in df_matched.columns
            ]
            if copy_cols:
                before = df_matched[copy_cols].reset_index(drop=True)
                after = canonical_sub[copy_cols].reset_index(drop=True)
                needs_replace = not before.equals(after)
                if needs_replace:
                    # Copy exact strings from canonical
                    df_matched.loc[:, copy_cols] = canonical_sub[copy_cols].values

    # Rebuild final after potential replacement
    df_final = pd.concat([df_matched, df_unmatched], axis=0, ignore_index=True)

    # Determine if reordering/deduplication or replacement changed the file
    original_order = list(range(len(df_res)))
    new_order = ordered_indices
    changed = (
        not (len(df_final) == len(df_res) and new_order == original_order)
    ) or needs_replace

    if not changed:
        print(f"Order correct: {results_path.name}")
        return

    # Backup and write only when changed
    backup_path = backup_file(results_path)
    try:
        df_final.to_csv(results_path, index=False)
    except Exception as e:
        warnings.append(
            f"Failed to write reordered file for {results_path.name} (backup at {backup_path.name}): {e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=f"{PROJECT_PATH}/test_data_2025-09-04_shuffled_correct",
        help="Directory containing canonical and result CSV files.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*_decimal_uniform_test_10k_shuffle_google__gemini-2.5-flash_re_high.csv",
        help="Glob pattern to select result files (relative to data_dir).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        print(f"Not a directory: {data_dir}", file=sys.stderr)
        sys.exit(1)

    warnings: List[str] = []
    # Find candidate result files
    result_files = sorted(data_dir.glob(args.glob))
    if not result_files:
        print("No result files matched the given glob.")
        sys.exit(0)

    for res_path in result_files:
        m = FILENAME_RE.match(res_path.name)
        if not m:
            # Skip non-matching
            continue
        task = m.group("task")
        canonical_path = data_dir / f"{task}_decimal_uniform_test_10k_shuffle.csv"
        reorder_one(res_path, canonical_path, warnings)

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"- {w}")
    else:
        print("\nCompleted with no warnings.")


if __name__ == "__main__":
    main()
