#!/usr/bin/env python3
"""
Analyze numeracy benchmarks and generate publication-quality plots.

Pipeline:
1) Load all CSVs from the data directory.
2) Robustly parse raw_response -> parsed_answer (repair malformed JSON, code blocks, "<thought>" prefixes).
3) Recompute metrics:
   - Regression tasks (Addition, Division, Exponentiation, Mean, Multiplication, Std): logSMAPE_acc in [0,1].
   - Classification tasks (MinMax, Interval, Sorting): exact match (1 or 0) with normalization rules per task.
   If parsing fails, score = 0.
4) Save a cleaned copy per file (optional) and a global parsing-errors CSV.
5) Create plots:
   - Radar charts: one subplot per model, axes are tasks; include multiple reasoning efforts as lighter/darker shades.
   - Tokens vs performance scatter: x=tokens, y=performance; color=model (shade=reasoning), marker=task.
   - Difficulty vs performance: one plot per task; line per model+reasoning across difficulty_sd.
6) Print and save a summary table listing model/reasoning combinations missing results by task.

Assumptions:
- File pattern: <TASK>_decimal_uniform_test_10k_shuffle_<MODEL_SAFE>_re_<REASONING>.csv
- Columns present: difficulty_sd, answer, tokens, raw_response; may already include parsed_answer, logSMAPE_acc or correct.
- Data dir default: [PROJECT_PATH]/test_data_2025-09-04_shuffled/
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decimal import Decimal, localcontext, Context
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from dotenv import load_dotenv
from matplotlib.lines import Line2D

# Add repo root to path so we can import eval_scripts.utils
if True:
    THIS_FILE = Path(__file__).resolve()
    REPO_ROOT = THIS_FILE.parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
from eval_scripts.utils import logSMAPE as logsmape_py
from eval_scripts.utils import parse_answer as base_parse_answer  # type: ignore
from eval_scripts.utils import parse_numbers_from_text  # type: ignore

load_dotenv()
import os  # noqa: E402

PROJECT_PATH = os.getenv("PROJECT_PATH")

# Reuse helpers for baseline behavior

# Tasks as used in the benchmark
TASKS: List[str] = [
    "Addition",
    "Division",
    "Exponentiation",
    "Interval",
    "Mean",
    "MinMax",
    "Multiplication",
    "Sorting",
    "Std",
]
# Display order for tasks in plots (subplots and radar axes)
TASKS_DISPLAY_ORDER: List[str] = [
    "Addition",
    "Multiplication",
    "Division",
    "Mean",
    "Std",
    "MinMax",
    "Interval",
    "Sorting",
    "Exponentiation",
]
TASKS_DISPLAY_ORDER_RADAR: List[str] = [
    "Mean",
    "Std",
    "Exponentiation",
    "Division",
    "Multiplication",
    "Addition",
    "MinMax",
    "Interval",
    "Sorting",
]
REGRESSION_TASKS: set[str] = {
    "Addition",
    "Division",
    "Exponentiation",
    "Mean",
    "Multiplication",
    "Std",
}
CLASSIFICATION_TASKS: set[str] = {"MinMax", "Interval", "Sorting"}

# Reasoning order for shade intensity (light -> dark)
REASONING_LEVELS: List[str] = ["maximal", "medium", "low", "minimal", "none"]
# Requested reasoning order for legends and plotting priority
REASONING_ORDER: List[str] = ["maximal", "low", "minimal", "none"]


def reasoning_sort_key(r: str) -> int:
    try:
        return REASONING_ORDER.index(r)
    except ValueError:
        # place unknown at end
        return len(REASONING_ORDER)


# Marker shapes by reasoning level
REASONING_MARKERS: Dict[str, str] = {
    "maximal": "*",  # star
    "low": "s",  # square
    "minimal": "^",  # triangle
    "none": "o",  # circle
}

# File name regex
FILENAME_RE = re.compile(
    r"^(?P<task>[A-Za-z]+)_decimal_uniform_test_10k_shuffle_(?P<model>.+?)_re_(?P<reasoning>[A-Za-z]+)\.csv$"
)

# Seaborn/Matplotlib styling for publication-quality figures
sns.set_theme(context="talk", style="whitegrid", font_scale=1.2)
mpl.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.titlepad": 14,
        "axes.labelpad": 10,
        "legend.title_fontsize": 12,
        "pdf.fonttype": 42,  # Editable text in PDFs
        "ps.fonttype": 42,
    }
)


@dataclass(frozen=True)
class FileKey:
    task: str
    model: str
    reasoning: str
    path: Path


@dataclass
class ParsedRecord:
    ok: bool
    parsed_answer: str
    reason: str = ""


def find_files(data_dir: Path) -> List[FileKey]:
    files: List[FileKey] = []
    for p in sorted(data_dir.glob("*.csv")):
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        task = m.group("task")
        model = m.group("model")
        reasoning = m.group("reasoning")
        files.append(FileKey(task=task, model=model, reasoning=reasoning, path=p))
    return files


def strip_code_fences(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove code fences ```...```, including optional language tags
    text = re.sub(r"```(?:json|JSON|python|txt)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", " ")
    return text.strip()


def remove_known_prefixes(text: str) -> str:
    if not text:
        return text
    # Remove leading labels like "JSON", "Answer:", "<thought>...</thought>"
    text = text.strip()
    text = re.sub(r"^\s*(JSON|Json|json)\s*[:\-]?\s*", "", text)
    # Drop simple XML-like thought wrappers
    text = re.sub(r"<thought>.*?</thought>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(
        r"<reasoning>.*?</reasoning>", " ", text, flags=re.IGNORECASE | re.DOTALL
    )
    return text.strip()


def extract_first_json_object(text: str) -> Optional[str]:
    """
    Extract the first balanced JSON object substring containing the key "answer".
    Conservative bracket matching to avoid catastrophic regex on nested braces.
    """
    if not text:
        return None
    start_positions = [i for i, ch in enumerate(text) if ch == "{"]
    for start in start_positions:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : end + 1]
                    if '"answer"' in candidate or "'answer'" in candidate:
                        return candidate
                    break
    return None


def parse_interval_label(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    # Match A-F possibly quoted, surrounded by non-word or string boundaries
    m = re.search(r"(?i)(?<![A-Z])([A-F])(?![A-Z])", text)
    if m:
        return m.group(1).upper()
    return None


def parse_list_like(text: str) -> Optional[List[float]]:
    if not isinstance(text, str):
        return None
    # Try JSON list first
    cleaned = text.strip()
    # If looks like a list without surrounding brackets, attempt to add
    looks_like_comma_numbers = (
        re.fullmatch(r"\s*[-+]?[\d.]+(?:\s*,\s*[-+]?[\d.]+)*\s*", cleaned) is not None
    )
    try_candidates = []
    if cleaned.startswith("[") and cleaned.endswith("]"):
        try_candidates.append(cleaned)
    elif looks_like_comma_numbers:
        try_candidates.append("[" + cleaned + "]")
    # Extract list-like substring if present
    list_match = re.search(r"\[(.*?)\]", cleaned, flags=re.DOTALL)
    if list_match:
        try_candidates.append("[" + list_match.group(1) + "]")

    for cand in try_candidates:
        try:
            arr = json.loads(cand)
            if isinstance(arr, list) and all(isinstance(x, (int, float)) for x in arr):
                return [float(x) for x in arr]
            # If values are strings but convertible to floats
            if isinstance(arr, list) and all(
                isinstance(x, (int, float, str)) for x in arr
            ):
                return [float(str(x).strip()) for x in arr]
        except Exception:
            continue

    # Fallback: extract numbers loosely
    nums = parse_numbers_from_text(cleaned)
    return nums if nums else None


def robust_parse_answer(response: str, task: str) -> ParsedRecord:
    """
    Enhanced parser:
    - Try base JSON parse (eval_scripts.utils.parse_answer)
    - If empty, strip code fences/prefixes and try again
    - If still empty, extract embedded JSON object
    - Task-specific fallbacks:
        Interval -> letter A-F
        Sorting  -> list of numbers
        Others   -> first numeric token
    """
    if not isinstance(response, str):
        return ParsedRecord(ok=False, parsed_answer="", reason="non_string_response")

    # 0) If only one pair of json braces, remove everything outside of it
    if response.count("{") == 1 and response.count("}") == 1:
        response = response.split("{")[1].split("}")[0]
        response = "{" + response + "}"

    # 1) Try base parser
    try:
        keep_raw = task in {"MinMax", "Sorting"}
        ans = base_parse_answer(response, keep_answer_raw=keep_raw) or ""
    except Exception:
        ans = ""
    if ans:
        return ParsedRecord(ok=True, parsed_answer=str(ans))

    # 2) Strip fences/prefixes and retry base parser
    cleaned = strip_code_fences(remove_known_prefixes(response))
    try:
        keep_raw = task in {"MinMax", "Sorting"}
        ans2 = base_parse_answer(cleaned, keep_answer_raw=keep_raw) or ""
    except Exception:
        ans2 = ""
    if ans2:
        return ParsedRecord(ok=True, parsed_answer=str(ans2))

    # 3) Extract embedded JSON then parse
    embedded = extract_first_json_object(cleaned)
    if embedded:
        try:
            if task in {"MinMax", "Sorting"}:
                obj = json.loads(embedded, parse_float=str, parse_int=str)
            else:
                obj = json.loads(embedded)
            if isinstance(obj, dict) and "answer" in obj:
                return ParsedRecord(ok=True, parsed_answer=str(obj["answer"]))
        except Exception:
            pass

    # 4) Task-specific heuristics
    if task == "Interval":
        label = parse_interval_label(cleaned)
        if label:
            return ParsedRecord(ok=True, parsed_answer=label)

    if task == "Sorting":
        arr = parse_list_like(cleaned)
        if arr is not None and len(arr) > 0:
            return ParsedRecord(
                ok=True, parsed_answer=json.dumps(arr, separators=(",", ":"))
            )

    # Numeric fallback for numeric tasks
    nums = parse_numbers_from_text(cleaned)
    if nums:
        # For MinMax keep the raw numeric token to avoid e-notation changes
        if task == "MinMax":
            m = re.search(
                r"(?:^| |)[+-]?(?:(?:0(?!\.[0-9]))|(?:[0-9]*[.][0-9]+)|(?:[1-9][0-9]*))(?:[eE][+-]?[0-9]+)?",
                cleaned,
            )
            if m:
                return ParsedRecord(ok=True, parsed_answer=m.group().strip())
        return ParsedRecord(ok=True, parsed_answer=str(nums[0]))

    return ParsedRecord(ok=False, parsed_answer="", reason="unparseable")


def first_numeric_token(text: str) -> Optional[str]:
    """Extract the first numeric token string (keeps sign and exponent if present)."""
    if not isinstance(text, str):
        return None
    m = re.search(
        r"(?:^|\s)[+-]?(?:(?:0(?!\.[0-9]))|(?:[0-9]*[.][0-9]+)|(?:[1-9][0-9]*))(?:[eE][+-]?[0-9]+)?",
        text,
    )
    if m:
        return m.group().strip()
    return None


def sig_signature(num_str: str, ndigits: int = 15) -> Optional[Tuple[int, str, int]]:
    """Return (sign, first ndigits of mantissa, exponent) from scientific notation of Decimal.

    - sign: 1 if negative, 0 otherwise
    - mantissa_digits: first ndigits digits of the significand (no decimal point)
    - exponent: base-10 exponent corresponding to scientific form [1,10) * 10^exponent
    Returns None if parsing fails or value is not finite.
    """
    try:
        with localcontext(Context(prec=60)):
            x = Decimal(str(num_str))
    except Exception:
        return None
    if not x.is_finite():
        return None
    if x == 0:
        return (1 if x.is_signed() else 0, "0" * ndigits, 0)
    sign = 1 if x.is_signed() else 0
    x = x.copy_abs()
    # High-precision scientific string, e.g., 1.2345E+03
    with localcontext(Context(prec=60)):
        s = format(x, "E")
    try:
        mant_str, exp_str = s.split("E")
    except ValueError:
        return None
    exp = int(exp_str)
    mant_digits = re.sub(r"[^0-9]", "", mant_str)
    if not mant_digits:
        return None
    if len(mant_digits) < ndigits:
        mant_digits = mant_digits.ljust(ndigits, "0")
    else:
        mant_digits = mant_digits[:ndigits]
    return (sign, mant_digits, exp)


def sig15_matches(a_str: Optional[str], b_str: Optional[str]) -> bool:
    """Return True if numbers match in sign, exponent, and first 15 significant digits."""
    if not a_str or not b_str:
        return False
    sa = sig_signature(a_str, ndigits=15)
    sb = sig_signature(b_str, ndigits=15)
    if sa is None or sb is None:
        return False
    return sa == sb


def normalize_truth(task: str, truth_raw: str) -> Tuple[str, Any]:
    """
    Return (normalized_string, structured_value) for ground truth to compare fairly.
    - Interval: uppercase single letter A-F
    - Sorting: JSON minified numeric list
    - MinMax: preserve exact numeric string (strip spaces and quotes)
    - Regression tasks: float value
    """
    if task == "Interval":
        v = parse_interval_label(truth_raw or "") or ""
        return v, v
    if task == "Sorting":
        arr = parse_list_like(truth_raw or "")
        return (json.dumps(arr, separators=(",", ":")) if arr is not None else ""), arr
    if task == "MinMax":
        s = str(truth_raw or "").strip().strip('"').strip("'")
        return s, s
    # Regression
    try:
        val = float(str(truth_raw))
    except Exception:
        val = float("nan")
    return str(val), val


def compute_sample_score(
    task: str, parsed_ans: str, truth_norm: Tuple[str, Any]
) -> float:
    """
    Return performance in [0,1].
    - Regression: logSMAPE between parsed float and truth float
    - Interval, Sorting: exact match after normalization
    - MinMax: exact string match (as specified)
    Unparseable -> 0
    """
    if task in REGRESSION_TASKS:
        try:
            pred = float(str(parsed_ans))
            true_val: float = truth_norm[1]
            if math.isnan(true_val):
                return 0.0
            return float(logsmape_py(pred, true_val))
        except Exception:
            return 0.0

    if task == "Interval":
        return (
            1.0 if (str(parsed_ans).strip().upper() == (truth_norm[0] or "")) else 0.0
        )

    if task == "Sorting":
        # Compare as numeric arrays with exact order
        try:
            pred_arr = parse_list_like(parsed_ans) or []
            true_arr = truth_norm[1] or []
            if len(pred_arr) != len(true_arr):
                return 0.0
            return (
                1.0
                if all(float(a) == float(b) for a, b in zip(pred_arr, true_arr))
                else 0.0
            )
        except Exception:
            return 0.0

    if task == "MinMax":
        # Exact numeric string match (trim quotes/spaces)
        pred_s = str(parsed_ans).strip().strip('"').strip("'")
        return 1.0 if pred_s == (truth_norm[0] or "") else 0.0

    return 0.0


def lighten_color(
    color: Tuple[float, float, float], factor: float
) -> Tuple[float, float, float]:
    """Lighten color by blending toward white. factor in [0,1]; 0=no change, 1=white."""
    r, g, b = color
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)


COLOR_HEX = {
    "deepseek-chat": "#0072B2",  # Blue
    "gemini 2.5 pro": "#E69F00",  # Orange
    "gemini 2.5 flash": "#9E0D19",  # Red
    "kimi k2": "#40531B",  # Dark Green
    "llama 4": "#009E73",  # Bluish Green
    "gpt 4.1": "#D55E00",  # Vermilion
    "gpt 5": "#000000",  # Black
    "gpt oss": "#C6A15B",  # Brown
    "qwen3": "#CC79A7",  # Reddish Purple
}


def canonical_model_key(model: str) -> str:
    s = model.replace("__", "/").lower()
    if "deepseek" in s:
        return "deepseek-chat"
    if "gemini-2.5-flash" in s:
        return "gemini 2.5 flash"
    if "gemini-2.5-pro" in s:
        return "gemini 2.5 pro"
    if "llama-4" in s:
        return "llama 4"
    if "kimi" in s:
        return "kimi k2"
    if "gpt-4.1" in s:
        return "gpt 4.1"
    if "gpt-5" in s:
        return "gpt 5"
    if "gpt-oss" in s or "gpt-0ss" in s or "oss-120b" in s or "gpt oss" in s:
        return "gpt oss"
    if "qwen3" in s or "qwen-3" in s or "qwen3-" in s or "qwen/" in s:
        return "qwen3"
    return s


def get_color_maps(models: List[str]) -> Dict[str, Tuple[float, float, float]]:
    """Return a color (RGB tuple) per exact model string using the specified palette.

    - Maps each model to one of the 9 requested base colors via canonical_model_key.
    - Falls back to HUSL for any models not covered by the 9 keys.
    """
    unique_models = sorted(set(models))
    fallback_palette = sns.color_palette("husl", n_colors=len(unique_models))
    fallback_iter = iter(fallback_palette)
    mapping: Dict[str, Tuple[float, float, float]] = {}
    for m in unique_models:
        key = canonical_model_key(m)
        if key in COLOR_HEX:
            mapping[m] = mcolors.to_rgb(COLOR_HEX[key])
        else:
            mapping[m] = next(fallback_iter)
    return mapping


def pretty_model_name(model: str) -> str:
    s = model.replace("__", "/").lower()
    if "deepseek" in s:
        return "DeepSeek v3.1"
    if "gemini-2.5-flash" in s:
        return "Gemini 2.5 Flash"
    if "gemini-2.5-pro" in s:
        return "Gemini 2.5 Pro"
    if "llama-4" in s:
        return "Llama 4 Maverick"
    if "kimi" in s:
        return "Kimi K2 0905"
    if "gpt-4.1" in s:
        return "GPT 4.1"
    if "gpt-5" in s:
        return "GPT 5"
    if "gpt-oss" in s or "gpt-0ss" in s or "oss-120b" in s or "gpt oss" in s:
        return "GPT OSS 120B"
    if "qwen3" in s or "qwen-3" in s or "qwen3-" in s or "qwen/" in s:
        return "Qwen3 235B 2507"
    return model.replace("__", "/")


def prety_task_names(task_names: List[str]) -> List[str]:
    mapping = {
        "Addition": "Add",
        "Multiplication": "Mult",
        "Exponentiation": "Exp",
        "Division": "Div",
        "Mean": "Mean",
        "Std": "Std",
        "MinMax": "MinMax",
        "Interval": "Interval",
    }
    return [mapping.get(task, task) for task in task_names]


def pretty_canonical_name(key: str) -> str:
    key = key.lower()
    mapping = {
        "deepseek-chat": "DeepSeek v3.1",
        "gemini 2.5 flash": "Gemini 2.5 Flash",
        "gemini 2.5 pro": "Gemini 2.5 Pro",
        "llama 4": "Llama 4 Maverick",
        "kimi k2": "Kimi K2 0905",
        "gpt 4.1": "GPT 4.1",
        "gpt 5": "GPT 5",
        "gpt oss": "GPT OSS 120B",
        "qwen3": "Qwen3 235B 2507",
    }
    return mapping.get(key, key)


def reasoning_shade(
    base: Tuple[float, float, float], reasoning: str
) -> Tuple[float, float, float]:
    # Map reasoning to lightness; higher effort -> darker (factor closer to 0)
    rank = REASONING_LEVELS.index(reasoning) if reasoning in REASONING_LEVELS else 0
    # none=0.65, minimal=0.5, low=0.35, medium=0.2, high=0.0
    factors = {0: 0.65, 1: 0.5, 2: 0.35, 3: 0.2, 4: 0.0}
    return lighten_color(base, factors.get(rank, 0.5))


TASK_MARKERS = {
    "Addition": "o",
    "Division": "s",
    "Exponentiation": "v",
    "Interval": "D",
    "Mean": "^",
    "MinMax": "P",
    "Multiplication": "X",
    "Sorting": "*",
    "Std": "h",
}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def make_radar_charts(df_all: pd.DataFrame, out_dir: Path) -> None:
    """
    One subplot per model (up to 10 per figure in a 2x5 grid).
    Axes: tasks (0..100). Multiple reasoning efforts per model as shades of the model color.
    Annotates exact percentage scores on each axis.
    """
    # Aggregate performance; drop GPT 4.1; merge Qwen variants by canonical key
    perf_raw = df_all.groupby(["model", "reasoning", "task"], as_index=False)[
        "performance"
    ].mean()
    perf_raw = perf_raw[
        ~perf_raw["model"].str.contains("gpt-4.1", case=False, na=False)
    ]
    perf_raw["display_key"] = perf_raw["model"].map(canonical_model_key)
    perf = perf_raw.groupby(["display_key", "reasoning", "task"], as_index=False)[
        "performance"
    ].mean()
    # Ordered display models (2x4 grid)
    ordered_keys = [
        "gpt oss",
        "qwen3",
        "deepseek-chat",
        "kimi k2",
        "gpt 5",
        "gemini 2.5 pro",
        "gemini 2.5 flash",
        "llama 4",
    ]
    present_keys = [
        k for k in ordered_keys if k in perf["display_key"].unique().tolist()
    ]
    # Color mapping by display key
    disp_to_color = {
        k: mcolors.to_rgb(COLOR_HEX[k]) for k in present_keys if k in COLOR_HEX
    }

    # Prepare axes order and wrap-around angle
    axes_labels = TASKS_DISPLAY_ORDER_RADAR.copy()
    num_axes = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()
    angles += angles[:1]  # close loop

    # Plot in 2x4 grids; multiple pages if more than 8 display models
    chunk_size = 8
    for start in range(0, len(present_keys), chunk_size):
        subset_models = present_keys[start : start + chunk_size]
        nrows, ncols = 2, 4
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True), figsize=(15, 7)
        )
        axs = axs.flatten()
        fig.subplots_adjust(hspace=0.2)

        for ax_idx, (ax, disp_key) in enumerate(zip(axs, subset_models)):
            sub = perf[perf["display_key"] == disp_key]
            base_col = disp_to_color.get(disp_key, (0.3, 0.3, 0.3))
            reasonings = sorted(sub["reasoning"].unique(), key=reasoning_sort_key)

            for r_idx, reasoning in enumerate(reasonings):
                vals_pct: List[float] = []
                sub_r = sub[sub["reasoning"] == reasoning]
                for t in axes_labels:
                    v = sub_r.loc[sub_r["task"] == t, "performance"].mean()
                    vals_pct.append(100.0 * float(v) if pd.notna(v) else 0.0)
                vals_closed = vals_pct + vals_pct[:1]

                col = reasoning_shade(base_col, reasoning)
                ax.plot(
                    angles,
                    vals_closed,
                    color=col,
                    linewidth=2.5,
                    label=reasoning,
                    zorder=3,
                )
                ax.fill(angles, vals_closed, color=col, alpha=0.15, zorder=2)
                # Use reasoning markers on radar points
                marker = REASONING_MARKERS.get(reasoning, "o")
                ax.scatter(
                    angles[:-1],
                    vals_pct,
                    s=100,
                    color=col,
                    edgecolors="k",
                    linewidths=0.3,
                    marker=marker,
                    zorder=10,
                    clip_on=False,
                )

                # Annotate exact scores with slight outward offset per reasoning to reduce overlap
                offset_per_level = 1.2
                base_offset = 1.5
                for ang, val in zip(angles[:-1], vals_pct):
                    radius = min(100.0, val + base_offset + r_idx * offset_per_level)
                    # ax.text(
                    #     ang,
                    #     radius,
                    #     f"{val:.1f}",
                    #     color=col,
                    #     fontsize=12,
                    #     ha="center",
                    #     va="bottom",
                    #     #weight="bold",
                    #     zorder=5,
                    # )

            if ax_idx >= ncols:  # bottom row (since you have 2 rows)
                ax.text(
                    0.5,
                    -0.25,
                    pretty_canonical_name(disp_key),
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=14,
                )
            else:
                ax.set_title(pretty_canonical_name(disp_key), pad=16, fontsize=14)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(prety_task_names(axes_labels), fontsize=12)
            # ax.tick_params(axis='x', pad=8)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels([20, 40, 60, 80, 100], fontsize=11)
            ax.set_ylim(0, 100)
            ax.set_rlabel_position(270)
            ax.grid(True, alpha=0.4, zorder=1)

            task_tick_padding_dict = {
                "Addition": 5,
                "Multiplication": 5,
                "Exponentiation": 0,
                "Division": 0,
                "Mean": 5,
                "Sorting": 5,
            }
            for tick, label in zip(ax.xaxis.get_major_ticks(), axes_labels):
                tick.set_pad(task_tick_padding_dict.get(label, 0))

            # No legends on radar plots per request

        # Hide any unused axes
        for j in range(len(subset_models), nrows * ncols):
            axs[j].axis("off")

        # Add a horizontal shade legend across the top (Reasoning levels)
        base_demo = (0.2, 0.2, 0.8)
        reasoning_labels = ["maximal", "minimal", "none"]
        reasoning_handles = [
            Line2D(
                [0],
                [0],
                marker=REASONING_MARKERS.get(r, "o"),
                color=reasoning_shade(base_demo, r),
                linestyle="None",
                markersize=10,
                label=r.capitalize(),
            )
            for r in reasoning_labels
        ]
        fig.legend(
            handles=reasoning_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=len(reasoning_handles),
            frameon=True,
            title="Reasoning (marker & shade)",
            fontsize=12,
            title_fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.subplots_adjust(hspace=0.4, wspace=0.0)
        out_path = out_dir / f"radar_models.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def rolling_mean_line(
    x: np.ndarray, y: np.ndarray, bins: int = 40, smooth_window: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute binned and smoothed means for difficulty-performance lines."""
    if len(x) == 0:
        return np.array([]), np.array([])
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    # Bin edges
    edges = np.linspace(x_sorted.min(), x_sorted.max(), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    inds = np.digitize(x_sorted, edges) - 1
    means = np.array(
        [
            np.nanmean(y_sorted[inds == i]) if np.any(inds == i) else np.nan
            for i in range(bins)
        ]
    )
    mask = ~np.isnan(means)
    centers, means = centers[mask], means[mask]
    if means.size >= smooth_window and smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        means = np.convolve(means, kernel, mode="same")
    return centers, means


def make_difficulty_matrix(df_all: pd.DataFrame, out_dir: Path) -> None:
    """Create a 3x3 matrix of difficulty vs performance plots (one per task)."""
    # Merge Qwen variants for display legend only; plotting uses per-row model color shaded by reasoning
    # Drop GPT 4.1 everywhere
    df_all = df_all[~df_all["model"].str.contains("gpt-4.1", case=False, na=False)]
    models = sorted(df_all["model"].unique().tolist())
    model_colors = get_color_maps(models)

    fig, axes = plt.subplots(3, 3, figsize=(22, 18), sharex=False, sharey=True)
    axes = axes.flatten()

    handles_accum: List[Tuple[str, Any]] = []
    for idx, task in enumerate(TASKS_DISPLAY_ORDER):
        ax = axes[idx]
        df_task = df_all[df_all["task"] == task]
        for (model, reasoning), sub in df_task.groupby(["model", "reasoning"]):
            x = sub["difficulty_sd"].to_numpy()
            y = sub["performance"].to_numpy()
            xb, yb = rolling_mean_line(x, y, bins=40, smooth_window=5)
            if xb.size == 0:
                continue
            col = reasoning_shade(model_colors[model], reasoning)
            (line,) = ax.plot(
                xb,
                100.0 * yb,
                color=col,
                linewidth=2.2,
                label=f"{model.replace('__','/')}: {reasoning}",
            )
            handles_accum.append((f"{model}: {reasoning}", line))
        # keep subplot title as the task name
        ax.set_title(task)
        ax.set_xlabel("Difficulty (sd)")
        if idx % 3 == 0:
            ax.set_ylabel("Performance (%)")
        ax.grid(True, alpha=0.3)

    # Remove empty axes if any
    for j in range(len(TASKS), 9):
        axes[j].axis("off")

    # Horizontal legends: models (top), reasoning (below)
    # Build canonical display models (merge Qwen variants)
    disp_keys = []
    for m in models:
        dk = canonical_model_key(m)
        if dk not in disp_keys:
            disp_keys.append(dk)
    disp_to_color: Dict[str, Tuple[float, float, float]] = {}
    for dk in disp_keys:
        # pick any original model under this key for base color
        orig = next((m for m in models if canonical_model_key(m) == dk), models[0])
        disp_to_color[dk] = model_colors[orig]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=disp_to_color[dk],
            linestyle="None",
            markersize=10,
            label=pretty_canonical_name(dk),
        )
        for dk in disp_keys
    ]
    # Two rows of up to 4 models each
    top_legend = fig.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        title="Models",
    )
    fig.add_artist(top_legend)
    base_demo = (0.2, 0.2, 0.8)
    reasoning_labels = ["maximal", "minimal", "none"]
    reasoning_handles_sorted = [
        Line2D(
            [0],
            [0],
            marker=REASONING_MARKERS.get(r, "o"),
            color=reasoning_shade(base_demo, r),
            linestyle="None",
            markersize=10,
            label=r.capitalize(),
        )
        for r in reasoning_labels
    ]
    bottom_legend = fig.legend(
        handles=reasoning_handles_sorted,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=4,
        frameon=True,
        title="Reasoning (Marker & Shade)",
    )
    fig.add_artist(bottom_legend)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "difficulty_vs_performance_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


def make_tokens_matrix(df_all: pd.DataFrame, out_dir: Path) -> None:
    """Create a compact 2x5 matrix of avg tokens vs avg performance (one subplot per task)."""
    # Config: sizes and paddings
    SCATTER_MARKER_SIZE = 350
    LEGEND_MARKER_SIZE = 18
    TITLE_PAD = 6
    XLABEL_PAD = 3
    XTICK_PAD = 2
    TICK_LABEL_SIZE = 20
    AXIS_LABEL_SIZE = 20
    SUBPLOT_TITLE_SIZE = 24
    LEGEND_FONT_SIZE = 18

    # Config: x-limits by task
    LONG_LIMIT = 35_000
    SHORT_LIMIT = 1_600
    TASKS_LONG = {
        "Addition",
        "Multiplication",
        "Division",
        "Exponentiation",
        "Mean",
        "Std",
    }
    TASKS_SHORT = {"MinMax", "Interval", "Sorting"}

    # Drop GPT 4.1
    df_filt = df_all[~df_all["model"].str.contains("gpt-4.1", case=False, na=False)]
    # Remove max reasoning error gemini
    df_filt = df_filt[df_filt["tokens"] != 65535]
    # Build base for performance (keep all finite/valid token rows)
    perf_base = df_filt[
        (df_filt["tokens"] >= 0) & np.isfinite(df_filt["tokens"])
    ].copy()
    # Build base for token averaging with special rule applied
    tokens_base = perf_base.copy()
    special_mask_tokens = (
        tokens_base["model"]
        .astype(str)
        .str.contains("gemini-2.5-flash", case=False, na=False)
        & tokens_base["reasoning"].astype(str).str.lower().eq("none")
        & (tokens_base["tokens"] > 200)
    )
    tokens_base.loc[special_mask_tokens, "tokens"] = np.nan

    grp_tokens = tokens_base.groupby(
        ["task", "model", "reasoning"], as_index=False
    ).agg(avg_tokens=("tokens", "mean"))
    grp_perf = perf_base.groupby(["task", "model", "reasoning"], as_index=False).agg(
        avg_perf=("performance", "mean")
    )
    grp = pd.merge(grp_tokens, grp_perf, on=["task", "model", "reasoning"], how="outer")
    models = sorted(grp["model"].unique().tolist())
    model_colors = get_color_maps(models)

    # fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex=False, sharey=False)
    fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex=False, sharey=True)
    axes = axes.flatten()

    # Collect handles for a single legend
    model_handles_done = set()
    reasoning_handles_done = set()
    model_handles: List[Any] = []
    reasoning_handles: List[Any] = []

    for idx, task in enumerate(TASKS_DISPLAY_ORDER):
        ax = axes[idx]
        sub = grp[grp["task"] == task]
        if sub.empty:
            ax.set_title(task, fontsize=13, pad=TITLE_PAD)
            ax.axis("off")
            continue
        for (model, reasoning), df_grp in sub.groupby(
            ["model", "reasoning"], as_index=False
        ):
            df_grp_df = cast(pd.DataFrame, df_grp)
            col = reasoning_shade(model_colors[model], reasoning)
            ax.scatter(
                df_grp_df["avg_tokens"],
                100.0 * df_grp_df["avg_perf"],
                s=SCATTER_MARKER_SIZE,
                c=[col],
                marker=REASONING_MARKERS.get(reasoning, "o"),
                alpha=0.9,
                edgecolors="k",
                linewidths=0.5,
                zorder=10,
                clip_on=False,
                label=f"{model.replace('__','/')}: {reasoning}",
            )
            if model not in model_handles_done:
                mh = Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=model_colors[model],
                    linestyle="None",
                    markersize=LEGEND_MARKER_SIZE,
                    label=model.replace("__", "/"),
                )
                model_handles.append(mh)
                model_handles_done.add(model)
            if reasoning not in reasoning_handles_done:
                # Use a neutral blue to demonstrate light/dark mapping for reasoning
                base_demo = (0.2, 0.2, 0.8)
                rh = Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=reasoning_shade(base_demo, reasoning),
                    linestyle="None",
                    markersize=LEGEND_MARKER_SIZE,
                    label=reasoning,
                )
                reasoning_handles.append(rh)
                reasoning_handles_done.add(reasoning)

        # keep subplot title as the task name
        ax.set_title(task, fontsize=SUBPLOT_TITLE_SIZE, pad=TITLE_PAD)
        ax.set_xlabel("Average tokens", fontsize=AXIS_LABEL_SIZE, labelpad=XLABEL_PAD)
        ax.set_yticks([0, 25, 50, 75, 100])
        if idx == 0 or idx == 8:
            ax.set_ylabel("log-sMAPE (%)", fontsize=AXIS_LABEL_SIZE)
        if idx == 5:
            ax.set_ylabel("Exact Match\nAccuracy (%)", fontsize=AXIS_LABEL_SIZE)

        # Apply requested x-limits by task
        if task in TASKS_LONG:
            ax.set_xscale("log")  # <<< log scale
            ax.set_xlim(0, 4e4)
            ax.set_xticks([1e2, 1e3, 1e4])
            ax.set_xticklabels(["100", "1k", "10k"])
        elif task in TASKS_SHORT:
            ax.set_xlim(-100, SHORT_LIMIT)
            ax.set_xticks([0, 500, 1000, 1500])

        # Style
        ax.grid(True, alpha=0.3, zorder=0)
        ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE, pad=XTICK_PAD)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)

    # Turn off last empty axis and place a combined legend inside it (bottom-right empty cell)
    if len(axes) >= 10:
        empty_ax = axes[9]
        empty_ax.axis("off")
        # Build combined legend entries: (display model, reasoning)
        # grp_disp = grp.copy()
        # grp_disp["display_model_key"] = grp_disp["model"].map(canonical_model_key)
        # present_pairs = (
        #     grp_disp[["display_model_key", "reasoning"]]
        #     .drop_duplicates()
        #     .values.tolist()
        # )
        # # Colors by display model
        # disp_keys = []
        # for m in models:
        #     dk = canonical_model_key(m)
        #     if dk not in disp_keys:
        #         disp_keys.append(dk)
        # disp_to_color: Dict[str, Tuple[float, float, float]] = {}
        # for dk in disp_keys:
        #     orig = next((m for m in models if canonical_model_key(m) == dk), models[0])
        #     disp_to_color[dk] = model_colors[orig]

        # def pair_sort_key(pr: List[str]) -> Tuple[int, int]:
        #     dk, r = pr
        #     try:
        #         dk_idx = disp_keys.index(dk)
        #     except ValueError:
        #         dk_idx = 999
        #     return (dk_idx, reasoning_sort_key(r))

        # present_pairs.sort(key=pair_sort_key)

        # combined_handles: List[Line2D] = []
        # for dk, r in present_pairs:
        #     color = disp_to_color.get(dk, (0.3, 0.3, 0.3))
        #     marker = REASONING_MARKERS.get(r, "o")
        #     label = f"{pretty_canonical_name(dk):<{20}}- {r.capitalize()}"
        #     combined_handles.append(
        #         Line2D(
        #             [0],
        #             [0],
        #             marker=marker,
        #             color=color,
        #             linestyle="None",
        #             markersize=LEGEND_MARKER_SIZE,
        #             label=label,
        #         )
        #     )

        # from matplotlib.font_manager import FontProperties
        # mono = FontProperties(family='DejaVu Sans Mono', size=12)
        # empty_ax.legend(
        #     handles=combined_handles,
        #     loc="upper left",
        #     bbox_to_anchor=(0.0, 1.2),
        #     frameon=True,
        #     title="Model — Reasoning",
        #     title_fontsize=14,
        #     ncol=1,
        #     prop=mono
        # )

        # --- build model-only legend (top) + reasoning schema legend (under it) ----
        # Figure bottom margin for stacked legends
        # fig.subplots_adjust(bottom=0.22)  # ~22% to fit two legend rows

        # Collect unique display keys in desired order and map to colors
        grp_disp = grp.copy()
        grp_disp["display_model_key"] = grp_disp["model"].map(canonical_model_key)

        disp_keys: List[str] = []
        for m in models:
            dk = canonical_model_key(m)
            if dk not in disp_keys:
                disp_keys.append(dk)

        disp_to_color: Dict[str, Tuple[float, float, float]] = {}
        for dk in disp_keys:
            orig = next((m for m in models if canonical_model_key(m) == dk), models[0])
            disp_to_color[dk] = model_colors[orig]

        # Build one handle per display model (no reasoning in label)
        model_only_handles: List[Line2D] = []
        for dk in disp_keys:
            model_only_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=".",
                    linestyle="None",
                    markersize=0,
                    color=disp_to_color[dk],
                    label=pretty_canonical_name(dk),
                )
            )

        # 1) Model legend (top row at the bottom of the figure)
        #    Tweak ncol to your taste; min(#models, 6) is usually readable.
        ncol_models = 1
        leg_models = fig.legend(
            handles=model_only_handles,
            loc="lower left",
            bbox_to_anchor=(0.84, -0.07),
            ncol=ncol_models,
            frameon=True,
            title="Models",
            title_fontsize=LEGEND_FONT_SIZE,
            fontsize=LEGEND_FONT_SIZE,
            handlelength=0,  # no line length
            handletextpad=0,  # no padding between handle and text
        )
        # Set font weight to bold for legend text
        # for text in leg_models.get_texts():
        #     text.set_fontweight('bold')
        # leg_models.get_title().set_fontweight('bold')
        # remove markers (handles) entirely, then color each label text
        for text, color in zip(
            leg_models.get_texts(), [disp_to_color[dk] for dk in disp_keys]
        ):
            text.set_color(color)
        fig.add_artist(leg_models)  # ensure this legend remains when adding another

        # 2) Reasoning schema legend (stacked directly under the model legend)
        # base_demo = (0.2, 0.2, 0.8)
        # reasoning_labels = ["high", "minimal", "none"]
        # reasoning_handles = [
        #     Line2D(
        #         [0], [0],
        #         marker=REASONING_MARKERS.get(r, "o"),
        #         color=reasoning_shade(base_demo, r),
        #         linestyle="None",
        #         markersize=LEGEND_MARKER_SIZE,
        #         label=r.capitalize(),
        #     )
        #     for r in reasoning_labels
        # ]
        # reasoning_legend = fig.legend(
        #     handles=reasoning_handles,
        #     loc="lower left",
        #     bbox_to_anchor=(0.94, 0.225),
        #     ncol=1,
        #     frameon=True,
        #     title="Reasoning\n(Marker & Shade)",
        #     fontsize=LEGEND_FONT_SIZE,
        #     title_fontsize=LEGEND_FONT_SIZE,
        # )
        # reasoning_legend.get_title().set_ha('center')

        base_demo = (0.2, 0.2, 0.8)
        reasoning_labels = ["maximal", "minimal", "none"]
        reasoning_handles = [
            Line2D(
                [0],
                [0],
                marker=REASONING_MARKERS.get(r, "o"),
                color=reasoning_shade(base_demo, r),
                linestyle="None",
                markersize=LEGEND_MARKER_SIZE,
                label=r.capitalize(),
            )
            for r in reasoning_labels
        ]
        fig.legend(
            handles=reasoning_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=len(reasoning_handles),
            frameon=True,
            title="Reasoning (Marker & Shade)",
            fontsize=LEGEND_FONT_SIZE,
            title_fontsize=LEGEND_FONT_SIZE,
        )

    fig.tight_layout()
    fig.savefig(out_dir / "tokens_vs_performance_matrix.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "tokens_vs_performance_matrix.png", bbox_inches="tight")
    plt.close(fig)


def make_samples_matrix(df_all: pd.DataFrame, out_dir: Path) -> None:
    """Create a 3x3 matrix of cumulative samples vs performance (one subplot per task).

    X-axis: number of samples (10-step increments).
    Y-axis: log-sMAPE (%) for regression rows, Exact Match Accuracy (%) for classification row.
    Legends: models (top, merged Qwen, two rows of 4), reasoning shades (bottom).
    """
    # Drop GPT 4.1
    df_filt = df_all[~df_all["model"].str.contains("gpt-4.1", case=False, na=False)]
    models = sorted(df_filt["model"].unique().tolist())
    model_colors = get_color_maps(models)

    # Typography inspired by tokens matrix
    SCATTER_MARKER_SIZE = 70
    LEGEND_MARKER_SIZE = 18
    TICK_LABEL_SIZE = 20
    AXIS_LABEL_SIZE = 20
    SUBPLOT_TITLE_SIZE = 24
    LEGEND_FONT_SIZE = 22

    fig, axes = plt.subplots(3, 3, figsize=(22, 18), sharex=False, sharey=True)
    axes = axes.flatten()

    # Arrange classification tasks (MinMax, Interval, Sorting) on the bottom row
    tasks_order = [
        "Addition",
        "Multiplication",
        "Division",
        "Mean",
        "Std",
        "Exponentiation",
        "MinMax",
        "Interval",
        "Sorting",
    ]

    for idx, task in enumerate(tasks_order):
        ax = axes[idx]
        sub_task = df_filt[df_filt["task"] == task]
        if sub_task.empty:
            ax.set_title(task, fontsize=SUBPLOT_TITLE_SIZE)
            ax.axis("off")
            continue

        # For each (model, reasoning), compute cumulative mean and plot at multiples of 10
        for (model, reasoning), sub in sub_task.groupby(
            ["model", "reasoning"], as_index=False
        ):
            perf = sub["performance"].to_numpy()
            if perf.size == 0:
                continue
            cum_mean = np.cumsum(perf) / np.arange(1, perf.size + 1)
            ks = np.arange(10, perf.size + 1, 10)
            if ks.size == 0:
                continue
            y_vals = (cum_mean[ks - 1]) * 100.0
            x_vals = ks

            col = reasoning_shade(model_colors[model], reasoning)
            marker = REASONING_MARKERS.get(reasoning, "o")
            ax.plot(x_vals, y_vals, color=col, linewidth=2.2, zorder=3)
            ax.scatter(
                x_vals,
                y_vals,
                s=SCATTER_MARKER_SIZE,
                color=col,
                marker=marker,
                edgecolors="k",
                linewidths=0.5,
                zorder=4,
            )

        # Keep subplot title as task name
        ax.set_title(task, fontsize=SUBPLOT_TITLE_SIZE)
        ax.set_xlabel("Number of samples", fontsize=AXIS_LABEL_SIZE)
        # Left-most axes per row: set appropriate y-label
        if idx in (0, 3):
            ax.set_ylabel("log-sMAPE (%)", fontsize=AXIS_LABEL_SIZE)
        if idx == 6:
            ax.set_ylabel("Exact Match Accuracy (%)", fontsize=AXIS_LABEL_SIZE)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)

    # Remove empty axes if any
    for j in range(len(TASKS_DISPLAY_ORDER), 9):
        axes[j].axis("off")

    # Legends: models on top (merged Qwen) and reasoning shades below
    disp_keys = []
    for m in models:
        dk = canonical_model_key(m)
        if dk not in disp_keys:
            disp_keys.append(dk)
    disp_to_color: Dict[str, Tuple[float, float, float]] = {}
    for dk in disp_keys:
        orig = next((m for m in models if canonical_model_key(m) == dk), models[0])
        disp_to_color[dk] = model_colors[orig]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=disp_to_color[dk],
            linestyle="None",
            markersize=LEGEND_MARKER_SIZE,
            label=pretty_canonical_name(dk),
        )
        for dk in disp_keys
    ]
    top_legend = fig.legend(
        handles=model_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=True,
        title="Models",
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
    )
    fig.add_artist(top_legend)

    base_demo = (0.2, 0.2, 0.8)
    reasoning_labels = ["maximal", "minimal", "none"]
    reasoning_handles_sorted = [
        Line2D(
            [0],
            [0],
            marker=REASONING_MARKERS.get(r, "o"),
            color=reasoning_shade(base_demo, r),
            linestyle="None",
            markersize=LEGEND_MARKER_SIZE,
            label=r.capitalize(),
        )
        for r in reasoning_labels
    ]
    bottom_legend = fig.legend(
        handles=reasoning_handles_sorted,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=True,
        title="Reasoning (Marker & Shade)",
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
    )
    fig.add_artist(bottom_legend)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "samples_vs_performance_matrix.pdf", bbox_inches="tight")
    # Also save a PNG for quick viewing
    fig.savefig(out_dir / "samples_vs_performance_matrix.png", bbox_inches="tight")
    plt.close(fig)


def export_latex_tables(df_all: pd.DataFrame, out_dir: Path) -> None:
    """Export two LaTeX tables:
    1) Performance (%) by task (log sMAPE for regression, accuracy for classification)
    2) Average tokens by task

    Both tables have columns: Model, Reasoning, <TASK columns in TASKS_DISPLAY_ORDER>.
    Models are rendered with publication-ready names; reasoning shown verbatim.
    """
    # Drop GPT 4.1 to match plotting exclusions
    df = df_all[~df_all["model"].str.contains("gpt-4.1", case=False, na=False)].copy()

    if df.empty:
        return

    # Aggregate
    perf_by = df.groupby(["model", "reasoning", "task"], as_index=False).agg(
        mean_performance=("performance", "mean")
    )
    # Special rule for token averaging: ignore token counts > 200 for gemini-2.5-flash with reasoning "none"
    df_tok = df.copy()
    special_mask_tok = (
        df_tok["model"]
        .astype(str)
        .str.contains("gemini-2.5-flash", case=False, na=False)
        & df_tok["reasoning"].astype(str).str.lower().eq("none")
        & (df_tok["tokens"] > 200)
    )
    df_tok.loc[special_mask_tok, "tokens"] = np.nan
    tok_by = df_tok.groupby(["model", "reasoning", "task"], as_index=False).agg(
        avg_tokens=("tokens", "mean")
    )
    # Regression-only exact-15 accuracy (mean over 0/1 flags)
    exact_df = df[df["task"].isin(list(REGRESSION_TASKS))]
    exact_by = exact_df.groupby(["model", "reasoning", "task"], as_index=False).agg(
        exact15_acc=("exact15", "mean")
    )

    # Desired task order
    task_cols = TASKS_DISPLAY_ORDER

    # Row ordering helpers
    ordered_display_keys = [
        "gpt oss",
        "qwen3",
        "deepseek-chat",
        "kimi k2",
        "llama 4",
        "gpt 5",
        "gemini 2.5 pro",
        "gemini 2.5 flash",
    ]

    def model_sort_key(m: str) -> int:
        key = canonical_model_key(m)
        try:
            return ordered_display_keys.index(key)
        except ValueError:
            return len(ordered_display_keys) + hash(key) % 1000

    # Pivot performance (to %)
    perf_piv = perf_by.pivot(
        index=["model", "reasoning"], columns="task", values="mean_performance"
    ).reindex(columns=task_cols)
    perf_piv = perf_piv.mul(100.0)
    perf_piv = perf_piv.reset_index()
    # Human-friendly model names and explicit sort keys
    perf_piv["Model"] = [pretty_model_name(m) for m in perf_piv["model"]]
    perf_piv["model_order"] = [model_sort_key(m) for m in perf_piv["model"]]
    perf_piv["reasoning_order"] = [reasoning_sort_key(r) for r in perf_piv["reasoning"]]
    perf_piv.rename(columns={"reasoning": "Reasoning"}, inplace=True)
    perf_display_cols = ["Model", "Reasoning"] + task_cols
    perf_piv = perf_piv.sort_values(
        ["model_order", "reasoning_order", "Model", "Reasoning"]
    ).reset_index(drop=True)
    perf_piv = perf_piv[perf_display_cols]

    # Format numbers: one decimal place for percentages
    perf_fmt = perf_piv.copy()
    for t in task_cols:
        if t in perf_fmt:
            perf_fmt[t] = perf_fmt[t].map(
                lambda x: "-" if pd.isna(x) else f"{float(x):.1f}"
            )

    # Pivot tokens (average, round to nearest integer)
    tok_piv = tok_by.pivot(
        index=["model", "reasoning"], columns="task", values="avg_tokens"
    ).reindex(columns=task_cols)
    tok_piv = tok_piv.reset_index()
    tok_piv["Model"] = [pretty_model_name(m) for m in tok_piv["model"]]
    tok_piv["model_order"] = [model_sort_key(m) for m in tok_piv["model"]]
    tok_piv["reasoning_order"] = [reasoning_sort_key(r) for r in tok_piv["reasoning"]]
    tok_piv.rename(columns={"reasoning": "Reasoning"}, inplace=True)
    tok_display_cols = ["Model", "Reasoning"] + task_cols
    tok_piv = tok_piv.sort_values(
        ["model_order", "reasoning_order", "Model", "Reasoning"]
    ).reset_index(drop=True)
    tok_piv = tok_piv[tok_display_cols]
    tok_fmt = tok_piv.copy()
    for t in task_cols:
        if t in tok_fmt:
            tok_fmt[t] = tok_fmt[t].map(
                lambda x: "-" if pd.isna(x) else f"{int(round(float(x))):d}"
            )

    # Column format: two left columns + right-aligned numeric columns
    col_format = "ll" + ("r" * len(task_cols))

    # Write LaTeX files
    ensure_dir(out_dir)
    perf_tex_path = out_dir / "table_performance_by_task.tex"
    tokens_tex_path = out_dir / "table_tokens_by_task.tex"
    exact15_tex_path = out_dir / "table_regression_exact15_by_task.tex"

    with open(perf_tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by evaluate_frontier_outputs.py\n")
        f.write(
            perf_fmt.to_latex(
                index=False, escape=False, na_rep="-", column_format=col_format
            )
        )

    with open(tokens_tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by evaluate_frontier_outputs.py\n")
        f.write(
            tok_fmt.to_latex(
                index=False, escape=False, na_rep="-", column_format=col_format
            )
        )

    # Also save CSV versions for sanity checks
    perf_fmt.to_csv(out_dir / "table_performance_by_task.csv", index=False)
    tok_fmt.to_csv(out_dir / "table_tokens_by_task.csv", index=False)

    # Build and write regression-only exact15 accuracy table (percentage)
    if not exact_by.empty:
        reg_cols = [t for t in TASKS_DISPLAY_ORDER if t in REGRESSION_TASKS]
        exact_piv = exact_by.pivot(
            index=["model", "reasoning"], columns="task", values="exact15_acc"
        ).reindex(columns=reg_cols)
        exact_piv = exact_piv.mul(100.0)
        exact_piv = exact_piv.reset_index()
        exact_piv["Model"] = [pretty_model_name(m) for m in exact_piv["model"]]
        exact_piv["model_order"] = [model_sort_key(m) for m in exact_piv["model"]]
        exact_piv["reasoning_order"] = [
            reasoning_sort_key(r) for r in exact_piv["reasoning"]
        ]
        exact_piv.rename(columns={"reasoning": "Reasoning"}, inplace=True)
        exact_display_cols = ["Model", "Reasoning"] + reg_cols
        exact_piv = exact_piv.sort_values(
            ["model_order", "reasoning_order", "Model", "Reasoning"]
        ).reset_index(drop=True)
        exact_piv = exact_piv[exact_display_cols]
        exact_fmt = exact_piv.copy()
        for t in reg_cols:
            if t in exact_fmt:
                exact_fmt[t] = exact_fmt[t].map(
                    lambda x: "-" if pd.isna(x) else f"{float(x):.1f}"
                )
        col_format_reg = "ll" + ("r" * len(reg_cols))
        with open(exact15_tex_path, "w", encoding="utf-8") as f:
            f.write("% Auto-generated by evaluate_frontier_outputs.py\n")
            f.write(
                exact_fmt.to_latex(
                    index=False, escape=False, na_rep="-", column_format=col_format_reg
                )
            )
        exact_fmt.to_csv(out_dir / "table_regression_exact15_by_task.csv", index=False)


def load_and_process_file(
    fk: FileKey,
    save_cleaned: bool,
    cleaned_dir: Path,
    parsing_errors: List[Dict[str, Any]],
    tool_mention_records: List[Dict[str, Any]],
    sample_limit: Optional[int] = None,
    canonical_df: Optional[pd.DataFrame] = None,
    warnings_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    df = pd.read_csv(fk.path, dtype=str)

    # Limit to first N samples if requested
    if isinstance(sample_limit, int) and sample_limit > 0:
        if len(df) < sample_limit and warnings_list is not None:
            warnings_list.append(
                f"{fk.path.name}: only {len(df)} samples (< {sample_limit})"
            )
        df = df.iloc[:sample_limit].copy()

    # Canonical row-by-row answer matching against task base file (if provided)
    if canonical_df is not None:
        if "answer" not in df.columns or "answer" not in canonical_df.columns:
            if warnings_list is not None:
                warnings_list.append(
                    f"{fk.path.name}: 'answer' column missing in file or canonical file"
                )
        else:
            n_check = min(len(df), len(canonical_df))
            if n_check > 0:
                a_model = df["answer"].astype(str).str.strip().values[:n_check]
                a_canon = (
                    canonical_df["answer"].astype(str).str.strip().values[:n_check]
                )
                mismatches = np.where(a_model != a_canon)[0].tolist()
                if mismatches and warnings_list is not None:
                    first = mismatches[:5]
                    warnings_list.append(
                        f"{fk.path.name}: {len(mismatches)} answer mismatches vs canonical in first {n_check} rows; first indices {first}"
                    )
            # Canonical too short for requested limit
            if (
                isinstance(sample_limit, int)
                and sample_limit > 0
                and len(canonical_df) < min(len(df), sample_limit)
                and warnings_list is not None
            ):
                warnings_list.append(
                    f"Canonical for task {fk.task} has only {len(canonical_df)} rows (< required {min(len(df), sample_limit)})"
                )
    # Coerce core fields
    # Answer as string; tokens numeric; difficulty float
    df["tokens"] = (
        pd.to_numeric(df.get("tokens", pd.Series([-1] * len(df))), errors="coerce")
        .fillna(-1)
        .astype(int)
    )
    df["difficulty_sd"] = pd.to_numeric(
        df.get("difficulty_sd", pd.Series([np.nan] * len(df))), errors="coerce"
    )

    # Normalize truth
    truth_norms = [
        normalize_truth(fk.task, a)
        for a in df.get("answer", pd.Series([""] * len(df))).tolist()
    ]

    parsed_answers: List[str] = []
    performances: List[float] = []
    exact15_flags: List[int] = []

    # Precompile tool-mention regex
    tool_re = re.compile(
        r"\b(calculator|python|wolfram\s*alpha|tool)\b", flags=re.IGNORECASE
    )

    for pos in range(len(df)):
        row = df.loc[df.index[pos]]
        raw = row.get("raw_response", "")
        parsed_rec = robust_parse_answer(raw, fk.task)
        if not parsed_rec.ok:
            parsing_errors.append(
                {
                    "file": fk.path.name,
                    "row_index": pos,
                    "task": fk.task,
                    "model": fk.model,
                    "reasoning": fk.reasoning,
                    "raw_response": str(raw)[:4000],  # cap for CSV
                    "reason": parsed_rec.reason or "parse_failed",
                }
            )
        pa = parsed_rec.parsed_answer
        parsed_answers.append(pa)
        perf = compute_sample_score(fk.task, pa, truth_norms[pos])
        performances.append(perf)

        # Compute exact-15 significant digits match for regression tasks
        if fk.task in REGRESSION_TASKS:
            truth_num_str = first_numeric_token(str(row.get("answer", "")))
            pred_num_str = first_numeric_token(pa)
            exact_ok = 1 if sig15_matches(pred_num_str, truth_num_str) else 0
            exact15_flags.append(exact_ok)
        else:
            exact15_flags.append(0)

        # Capture tool usage mentions in reasoning if present
        reasoning_text = str(row.get("reasoning", ""))
        if (
            isinstance(reasoning_text, str)
            and reasoning_text
            and tool_re.search(reasoning_text)
        ):
            tool_mention_records.append(
                {
                    "model": fk.model,
                    "task": fk.task,
                    "prompt": str(row.get("text_prompt", "")),
                    "reasoning": reasoning_text,
                    "answer": str(row.get("answer", "")),
                    "parsed_answer": pa,
                    "logSMAPE": performances[-1] if fk.task in REGRESSION_TASKS else "",
                    "correct": (
                        (performances[-1] >= 0.999999)
                        if fk.task not in REGRESSION_TASKS
                        else ""
                    ),
                }
            )

    df["parsed_answer"] = parsed_answers
    # For regression tasks, also expose logSMAPE_acc for compatibility; else 'correct'
    if fk.task in REGRESSION_TASKS:
        df["logSMAPE_acc"] = performances
    else:
        df["correct"] = [bool(p >= 0.999999) for p in performances]

    df["_performance"] = performances  # internal numeric in [0,1]
    df["_exact15"] = exact15_flags  # internal 0/1 for regression tasks (0 otherwise)
    if save_cleaned:
        ensure_dir(cleaned_dir)
        out_path = cleaned_dir / fk.path.name.replace(".csv", "_cleaned.csv")
        df.to_csv(out_path, index=False)
    # Return long form with metadata
    df_out = pd.DataFrame(
        {
            "task": fk.task,
            "model": fk.model,
            "reasoning": fk.reasoning,
            "difficulty_sd": df["difficulty_sd"].astype(float),
            "tokens": df["tokens"].astype(int),
            "performance": df["_performance"].astype(float),
            "exact15": pd.Series(exact15_flags, index=df.index).astype(int),
        }
    )
    return df_out


def summarize_missing(files: List[FileKey]) -> pd.DataFrame:
    """
    Build a table of model/reasoning x task indicating presence (1) or missing (0).
    """
    rows = []
    for fk in files:
        rows.append(
            {
                "model": fk.model,
                "reasoning": fk.reasoning,
                "task": fk.task,
                "present": 1,
            }
        )
    present_df = pd.DataFrame(rows)
    if present_df.empty:
        return pd.DataFrame(columns=["model", "reasoning", "task_missing_count"])
    # All observed model/reasoning combos
    combos = present_df[["model", "reasoning"]].drop_duplicates()
    # Desired tasks set: TASKS list
    recs = []
    for _, comb in combos.iterrows():
        m = comb["model"]
        r = comb["reasoning"]
        present_tasks = set(
            present_df[(present_df["model"] == m) & (present_df["reasoning"] == r)][
                "task"
            ]
        )
        missing = [t for t in TASKS if t not in present_tasks]
        recs.append(
            {
                "model": m,
                "reasoning": r,
                "tasks_present": len(present_tasks),
                "tasks_missing": len(missing),
                "missing_task_list": ";".join(missing),
            }
        )
    return pd.DataFrame(recs).sort_values(["model", "reasoning"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=f"{PROJECT_PATH}/test_data_2025-09-04_shuffled_correct/",
        help="Directory containing result CSV files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="analysis",
        help="Output directory (created inside data_dir).",
    )
    parser.add_argument(
        "--save_cleaned",
        action="store_true",
        help="Save cleaned per-file CSVs with repaired parsed_answer/metrics.",
    )
    parser.add_argument(
        "--load_saved",
        action="store_true",
        help="If set, load combined long dataframe from disk instead of recomputing.",
    )
    parser.add_argument(
        "--individual_difficulty",
        action="store_true",
        help="If set, also generate individual per-task difficulty vs performance plots.",
    )
    parser.add_argument(
        "--limit_n",
        type=int,
        default=500,
        help="Evaluate only the first N samples per file (default: 500).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = (data_dir / args.out_dir).resolve()
    cleaned_dir = out_dir / "cleaned"
    ensure_dir(out_dir)

    combined_path = out_dir / "combined_samples_long.csv"
    files = find_files(data_dir)
    df_all: pd.DataFrame
    tool_mention_records: List[Dict[str, Any]] = []
    runtime_warnings: List[str] = []

    # Pre-load canonical base files per task for matching
    task_set = sorted({fk.task for fk in files})
    canonical_by_task: Dict[str, Optional[pd.DataFrame]] = {}
    for task in task_set:
        canon_path = data_dir / f"{task}_decimal_uniform_test_10k_shuffle.csv"
        if canon_path.exists():
            try:
                canonical_by_task[task] = pd.read_csv(canon_path, dtype=str)
            except Exception as e:
                canonical_by_task[task] = None
                runtime_warnings.append(
                    f"Failed to load canonical file for task {task}: {e}"
                )
        else:
            canonical_by_task[task] = None
            runtime_warnings.append(
                f"Canonical file not found for task {task}: {canon_path}"
            )
    if args.load_saved and combined_path.exists():
        print(f"Loading combined dataframe from {combined_path}")
        df_all = pd.read_csv(combined_path)
    else:
        if not files:
            print(f"No matching CSV files found in {data_dir}")
            return
        parsing_errors: List[Dict[str, Any]] = []
        frames: List[pd.DataFrame] = []
        for fk in files:
            try:
                df_part = load_and_process_file(
                    fk=fk,
                    save_cleaned=args.save_cleaned,
                    cleaned_dir=cleaned_dir,
                    parsing_errors=parsing_errors,
                    tool_mention_records=tool_mention_records,
                    sample_limit=args.limit_n,
                    canonical_df=canonical_by_task.get(fk.task),
                    warnings_list=runtime_warnings,
                )
                frames.append(df_part)
            except Exception as e:
                print(f"Error processing {fk.path.name}: {e}", file=sys.stderr)
        if not frames:
            print("No data after processing.")
            return
        df_all = pd.concat(frames, ignore_index=True)
        # Save combined long dataframe
        df_all.to_csv(combined_path, index=False)
        print(f"Saved combined dataframe: {combined_path}")
        print("Removing GPT-OSS Low")
        df_all = df_all[
            ~(
                (df_all["model"] == "openai__gpt-oss-120b")
                & (df_all["reasoning"] == "low")
            )
        ]
        print("Removing grok-4-fast_free")
        df_all = df_all[~(df_all["model"] == "x-ai__grok-4-fast_free")]
        # Save parsing errors (if any)
        if parsing_errors:
            err_df = pd.DataFrame(parsing_errors)
            err_df.to_csv(out_dir / "invalid_json_rows.csv", index=False)
            print(f"Saved parsing errors: {out_dir / 'invalid_json_rows.csv'}")
        else:
            print("No parsing errors detected by robust parser.")

        # Save tool-mention detections
        if tool_mention_records:
            pd.DataFrame(tool_mention_records).to_csv(
                out_dir / "reasoning_tool_mentions.csv", index=False
            )
            print(
                f"Saved reasoning tool mentions: {out_dir / 'reasoning_tool_mentions.csv'}"
            )

    # Save aggregated summary dataframes for reuse
    # Drop GPT 4.1 from saved summaries as well
    df_all_no_gpt41 = df_all[
        ~df_all["model"].str.contains("gpt-4.1", case=False, na=False)
    ].copy()
    perf_by_mrt = df_all_no_gpt41.groupby(
        ["model", "reasoning", "task"], as_index=False
    ).agg(mean_performance=("performance", "mean"))
    perf_by_mrt.to_csv(out_dir / "perf_by_model_reasoning_task.csv", index=False)
    # Apply special token averaging rule before aggregation
    df_tokens = df_all_no_gpt41.copy()
    special_mask_main = (
        df_tokens["model"]
        .astype(str)
        .str.contains("gemini-2.5-flash", case=False, na=False)
        & df_tokens["reasoning"].astype(str).str.lower().eq("none")
        & (df_tokens["tokens"] > 200)
    )
    df_tokens.loc[special_mask_main, "tokens"] = np.nan
    tokens_perf_by_trm = df_tokens.groupby(
        ["task", "model", "reasoning"], as_index=False
    ).agg(avg_tokens=("tokens", "mean"), avg_perf=("performance", "mean"))
    tokens_perf_by_trm.to_csv(
        out_dir / "avg_tokens_perf_by_task_model_reasoning.csv", index=False
    )

    # Convert high reasoning to maximal
    df_all["reasoning"] = df_all["reasoning"].replace("high", "maximal")

    # Plots
    make_radar_charts(df_all, out_dir)
    # Super-plots
    make_difficulty_matrix(df_all, out_dir)
    make_tokens_matrix(df_all, out_dir)
    make_samples_matrix(df_all, out_dir)

    # LaTeX tables for supplementary materials
    export_latex_tables(df_all, out_dir)

    # Summary table for missing model/reasoning combos by task
    # If we loaded from saved without enumerating files, rebuild file list for summary
    if not files:
        files = find_files(data_dir)
    missing_df = summarize_missing(files)
    sum_path = out_dir / "overview_per_model_by_reasoning.csv"
    missing_df.to_csv(sum_path, index=False)
    print("\nMissing results summary (model/reasoning):")
    if not missing_df.empty:
        with pd.option_context("display.max_colwidth", 100):
            print(missing_df.to_string(index=False))
    else:
        print("None")
    print(f"\nSaved summary: {sum_path}")

    # High-level dataset coverage
    coverage = (
        df_all_no_gpt41.groupby(["model", "reasoning"])
        .agg(
            n_samples=("performance", "size"),
            mean_perf=("performance", "mean"),
        )
        .reset_index()
    )
    cov_path = out_dir / "coverage_by_model_reasoning.csv"
    coverage.to_csv(cov_path, index=False)
    print(f"Saved coverage stats: {cov_path}")

    # Final warnings block
    if runtime_warnings:
        print("\nWarnings:")
        for w in runtime_warnings:
            print(f"- {w}")
    else:
        print("\nNo warnings.")


if __name__ == "__main__":
    main()
