#!/usr/bin/env python3
"""Merge Gemini benchmarking CSV runs (_H1 and _H2) into a single _high file.

Rules:
- Use the row from _H1 unless the 'raw_response' field in _H1 is empty.
- If 'raw_response' in _H1 is empty, take the row from _H2.
- If both _H1 and _H2 have empty 'raw_response' and both 'tokens' are 65535,
  set raw_response to "Token limit reached" in the merged row.
- If both 'tokens' are not 65535 in this case, log a warning.
- Always copy all fields (union of columns across both files) into output.

Usage:
  python -m eval_scripts.merge_gemini --dir /abs/path/to/test_data

Notes:
- Only files with names containing "gemini" are considered.
- Pairs are identified by replacing the _H1 suffix with _H2.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()
import os  # noqa: E402

PROJECT_PATH = os.getenv("PROJECT_PATH")

LOGGER = logging.getLogger(__name__)


def is_empty_response(value: Optional[str]) -> bool:
    """Return True if the raw response field should be considered empty."""
    if value is None:
        return True
    return value.strip() == ""


def read_csv_dicts(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read a CSV into a list of dict rows and return rows with the header order.

    Args:
        path: CSV file path.

    Returns:
        (rows, fieldnames)
    """
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def union_fieldnames(primary: List[str], secondary: List[str]) -> List[str]:
    """Return an ordered union of two fieldname lists.

    The output preserves the order of the primary list first, then appends any
    names present in the secondary list but missing from the primary, in the
    order they appear in the secondary list.
    """
    seen = set(primary)
    merged = list(primary)
    for name in secondary:
        if name not in seen:
            merged.append(name)
            seen.add(name)
    return merged


def parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def write_csv_dicts(
    path: Path, rows: List[Dict[str, str]], fieldnames: List[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Ensure all fields present; fill missing with empty string
            complete = {name: row.get(name, "") for name in fieldnames}
            writer.writerow(complete)


def merge_rows(
    h1_row: Dict[str, str], h2_row: Dict[str, str], row_index: int, source_label: str
) -> Dict[str, str]:
    """Merge two rows according to the specified rules.

    Args:
        h1_row: Row from _H1.csv
        h2_row: Row from _H2.csv
        row_index: Zero-based index of the row (for logging)
        source_label: A label to include in warnings (typically the base filename)

    Returns:
        Merged row dict (may include keys from both rows)
    """
    h1_resp_empty = is_empty_response(h1_row.get("raw_response"))
    h2_resp_empty = is_empty_response(h2_row.get("raw_response"))

    if not h1_resp_empty:
        # Prefer H1 when it has a response
        return dict(h1_row)

    if not h2_resp_empty:
        # Fall back to H2 when H1 is empty
        return dict(h2_row)

    # Both empty: inspect tokens
    h1_tokens = parse_int(h1_row.get("tokens"))
    h2_tokens = parse_int(h2_row.get("tokens"))

    merged = dict(h1_row)  # start with H1; we'll override raw_response as needed

    if h1_tokens == 65535 and h2_tokens == 65535:
        merged["raw_response"] = "Token limit reached"
        return merged

    LOGGER.warning(
        "[%s] Row %d: both raw_response empty but tokens not both 65535 (H1=%s, H2=%s)",
        source_label,
        row_index + 1,
        h1_row.get("tokens"),
        h2_row.get("tokens"),
    )
    # Keep H1 as base; raw_response stays empty per rule, only a warning is raised
    return merged


def merge_pair(h1_path: Path, h2_path: Path, out_path: Path) -> int:
    """Merge a single (_H1, _H2) pair to out_path.

    Returns the number of rows written.
    """
    h1_rows, h1_fields = read_csv_dicts(h1_path)
    h2_rows, h2_fields = read_csv_dicts(h2_path)

    if len(h1_rows) != len(h2_rows):
        raise ValueError(
            f"Row count mismatch between {h1_path.name} ({len(h1_rows)}) and {h2_path.name} ({len(h2_rows)})"
        )

    fieldnames = union_fieldnames(h1_fields, h2_fields)
    merged_rows: List[Dict[str, str]] = []

    base_label = out_path.name

    for idx, (r1, r2) in enumerate(zip(h1_rows, h2_rows)):
        merged = merge_rows(r1, r2, idx, base_label)
        # Ensure we don't drop fields present only in H2
        for key in r2.keys():
            if key not in merged:
                merged[key] = r2[key]
        merged_rows.append(merged)

    write_csv_dicts(out_path, merged_rows, fieldnames)
    return len(merged_rows)


def find_gemini_pairs(directory: Path) -> List[Tuple[Path, Path, Path]]:
    """Find (_H1, _H2, _high) triplets for gemini files in directory.

    Returns list of tuples (h1_path, h2_path, out_path). Only files whose names
    contain "gemini" (case-insensitive) are considered. The output path replaces
    the _H1 suffix with _high.
    """
    pairs: List[Tuple[Path, Path, Path]] = []
    for h1_path in directory.glob("*_H1.csv"):
        name_lower = h1_path.name.lower()
        if "gemini" not in name_lower:
            continue
        h2_path = h1_path.with_name(h1_path.name.replace("_H1.csv", "_H2.csv"))
        if not h2_path.exists():
            LOGGER.warning(
                "Skipping %s because matching _H2 file not found", h1_path.name
            )
            continue
        out_path = h1_path.with_name(h1_path.name.replace("_H1.csv", "_high.csv"))
        pairs.append((h1_path, h2_path, out_path))
    return pairs


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Gemini _H1/_H2 CSVs to _high.csv"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(
            f"{PROJECT_PATH}/test_data_2025-09-04_shuffled_correct"
        ),
        help="Directory containing CSV files to merge (default: repository test_data path)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (INFO) logging; otherwise warnings and errors only",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


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

    pairs = find_gemini_pairs(directory)
    if not pairs:
        LOGGER.warning("No Gemini _H1/_H2 pairs found in %s", directory)
        return 0

    total_rows = 0
    for h1_path, h2_path, out_path in pairs:
        rows_written = merge_pair(h1_path, h2_path, out_path)
        total_rows += rows_written
        LOGGER.info(
            "Merged %s and %s -> %s (%d rows)",
            h1_path.name,
            h2_path.name,
            out_path.name,
            rows_written,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
