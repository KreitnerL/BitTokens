#!/usr/bin/env python3
"""Analyze Gemini run costs for _H1/_H2 CSVs.

This script scans a directory for Gemini `_H1.csv` and `_H2.csv` files and
computes the total "new" costs per file. A "new" cost is defined as rows where
the `usage.cost` field is missing or empty. Costs are calculated as:

    usage.prompt_tokens * prompt_cost + usage.tokens * completion_cost

The model name is extracted from the filename and used to look up pricing. The
output is saved to `<input dir>/analysis/gemini_cost_analysis.csv` with columns:
    filename, model, task, costs, MAX_TOKENS

Where MAX_TOKENS is the count of rows with usage.tokens == 65535.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()
import os  # noqa: E402

PROJECT_PATH = os.getenv("PROJECT_PATH")

costs_dict: Dict[str, Dict[str, float]] = {
    "gemini-2.5-flash": {
        "completion_cost": 1.25 / 1_000_000,
        "prompt_cost": 0.15 / 1_000_000,
    },
    "gemini-2.5-pro": {
        "completion_cost": 5.0 / 1_000_000,
        "prompt_cost": 0.625 / 1_000_000,
    },
}


LOGGER = logging.getLogger(__name__)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute new costs for Gemini _H1/_H2 CSVs"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(
            f"{PROJECT_PATH}/test_data_2025-09-04_shuffled_correct"
        ),
        help="Directory containing CSV files (default: repository test_data path)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logging (default WARNING)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="analysis/gemini_cost_analysis.csv",
        help="Output path relative to --dir (default: analysis/gemini_cost_analysis.csv)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def is_empty(value: Optional[str]) -> bool:
    return value is None or value.strip() == ""


def safe_int(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    s = text.strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def extract_model_and_task(filename: str) -> Tuple[Optional[str], Optional[str]]:
    name = filename
    lower = name.lower()
    model: Optional[str] = None
    for candidate in costs_dict.keys():
        if candidate in lower:
            model = candidate
            break
    if model is None:
        return None, None

    # Task: substring before the double underscore prior to the model segment
    # E.g., "Addition_..._google__gemini-2.5-flash_re_H1.csv" -> task before "__"
    parts = name.split("__", 1)
    task = parts[0] if parts else None
    return model, task


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def compute_file_new_cost(path: Path, model: str) -> Tuple[float, int]:
    rows = read_csv_rows(path)
    pricing = costs_dict[model]
    prompt_rate = pricing["prompt_cost"]
    completion_rate = pricing["completion_cost"]

    total_cost = 0.0
    max_tokens_count = 0

    for row in rows:
        usage_cost = row.get("usage.cost")
        if not is_empty(usage_cost):
            # Existing cost: skip (not a new cost)
            pass
        else:
            prompt_tokens = safe_int(row.get("usage.prompt_tokens")) or 0
            completion_tokens = safe_int(row.get("usage.tokens")) or 0
            total_cost += (
                prompt_tokens * prompt_rate + completion_tokens * completion_rate
            )

        usage_tokens = safe_int(row.get("usage.tokens"))
        if usage_tokens == 65535:
            max_tokens_count += 1

    return total_cost, max_tokens_count


def find_target_files(directory: Path) -> List[Path]:
    files: List[Path] = []
    for path in directory.glob("*_H1.csv"):
        if "gemini" in path.name.lower():
            files.append(path)
    for path in directory.glob("*_H2.csv"):
        if "gemini" in path.name.lower():
            files.append(path)
    files.sort()
    return files


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    directory: Path = args.dir
    if not directory.exists() or not directory.is_dir():
        LOGGER.error("Provided --dir is not a directory: %s", directory)
        return 2

    files = find_target_files(directory)
    if not files:
        LOGGER.warning("No Gemini _H1/_H2 files found in %s", directory)
        return 0

    # Prepare output path
    out_path = (directory / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["filename", "model", "task", "costs", "MAX_TOKENS"])

        for file_path in files:
            model, task = extract_model_and_task(file_path.name)
            if model is None:
                LOGGER.warning("Skipping %s (model not recognized)", file_path.name)
                continue
            total_cost, max_tokens = compute_file_new_cost(file_path, model)
            writer.writerow(
                [
                    file_path.name,
                    model,
                    task or "",
                    f"{total_cost:.6f}",
                    max_tokens,
                ]
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
