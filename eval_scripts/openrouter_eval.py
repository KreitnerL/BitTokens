from dotenv import load_dotenv

load_dotenv()
from openai import OpenAI
import os
from utils import (
    parse_answer,
    NUMERIC_SYSTEM_MESSAGE,
    INTERVAL_SYSTEM_MESSAGE,
    SORTING_SYSTEM_MESSAGE,
    MIN_MAX_SYSTEM_MESSAGE,
    AI_MESSAGE,
    eval_regression,
)
import argparse
from sys import maxsize
import pandas as pd
from tqdm.auto import tqdm
import datetime
from pathlib import Path
import json
import time
import random
import sys
import re
from google import genai as google_genai
from google.genai import types as gtypes
from google.genai.types import ContentListUnionDict
from openai._types import NOT_GIVEN
from typing import Any, Sequence, cast

MODELS = [
    "openai/gpt-5",  # Reasoning always on. Valid values are minimal, low, medium, high. None is converted to minimal.
    # "x-ai/grok-4", # reasoning_effort not supported. Always on. Standardized to high.
    "x-ai/grok-4-fast:free",
    # "openai/o3-pro", # reasoning_effort not supported. Always on. Standardized to high.
    "google/gemini-2.5-pro",  # Reasoning always on. none converted to 128 tokens.
    "qwen/qwen3-235b-a22b-thinking-2507",  # reasoning_effort not supported. Always on. Standardized to high.
    "openai/gpt-oss-120b",  # Reasoning always on. Valid values are low, medium, high. None is converted to low.
    # "anthropic/claude-opus-4.1", # Reasoning & Non-reasoning
    "deepseek/deepseek-chat-v3.1",  # Reasoning & Non-reasoning
    "qwen/qwen3-235b-a22b-2507",  # No reasoning available. Standardized to none.
    "google/gemini-2.5-flash",  # Reasoning & Non-reasoning
    # "openai/gpt-4.1", # No reasoning available. Standardized to none.
    "moonshotai/kimi-k2-0905",  # No reasoning available. Standardized to none.
    "meta-llama/llama-4-maverick",  # No reasoning available. Standardized to none.
]

TASKS = [
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

REGRESSION_TASKS = [
    "Addition",
    "Division",
    "Exponentiation",
    "Mean",
    "Multiplication",
    "Std",
]


def task_to_system_message(task: str) -> str:
    if task == "Interval":
        return INTERVAL_SYSTEM_MESSAGE
    elif task == "Sorting":
        return SORTING_SYSTEM_MESSAGE
    elif task == "MinMax":
        return MIN_MAX_SYSTEM_MESSAGE
    else:
        return NUMERIC_SYSTEM_MESSAGE


def build_response_format(task: str) -> dict:
    """Construct response_format enforcing JSON with an "answer" field per task.

    - Interval: one of A-F (string)
    - Sorting: array of numbers
    - Default: number
    """
    if task == "Interval":
        answer_schema = {"type": "string", "enum": ["A", "B", "C", "D", "E", "F"]}
    elif task == "Sorting":
        answer_schema = {"type": "array", "items": {"type": "number"}}
    else:
        answer_schema = {"type": "number"}

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": answer_schema},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }


def build_gemini_response_schema(task: str) -> dict:
    """Construct plain JSON Schema for Gemini response_schema matching build_response_format."""
    if task == "Interval":
        answer_schema = {"type": "string", "enum": ["A", "B", "C", "D", "E", "F"]}
    elif task == "Sorting":
        answer_schema = {"type": "array", "items": {"type": "number"}}
    else:
        answer_schema = {"type": "number"}

    return {
        "type": "object",
        "properties": {"answer": answer_schema},
        "required": ["answer"],
    }


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_file", type=str, required=True, help="Path to input CSV file"
)
parser.add_argument(
    "--model_name", type=str, default="meta-llama/llama-4-maverick", help="Model name"
)
parser.add_argument(
    "--save_interval",
    type=int,
    default=10,
    help="Number of samples after which to sync to disk",
)
parser.add_argument("--few_shot", action="store_true", help="Use few-shot prompting")
parser.add_argument(
    "--fs_examples",
    type=int,
    default=3,
    help="Number of few-shot examples to prepend when --few_shot is set",
)
parser.add_argument(
    "--resume",
    dest="resume",
    action="store_true",
    default=True,
    help="Auto-resume from existing output file (default)",
)
parser.add_argument(
    "--no-resume",
    dest="resume",
    action="store_false",
    help="Disable auto-resume and start from the beginning",
)
parser.add_argument(
    "--reasoning_effort",
    type=str,
    choices=["high", "medium", "low", "none"],
    default="none",
    help="Reasoning effort level for models that support it",
)
parser.add_argument(
    "--limit",
    type=int,
    default=0,
    help="If > 0, only process up to this many rows; still runs post-processing if already exceeded",
)
parser.add_argument(
    "--max_retries", type=int, default=20, help="Maximum retries on API failure"
)
parser.add_argument(
    "--backoff_base",
    type=float,
    default=1.8,
    help="Exponential backoff base multiplier",
)
parser.add_argument(
    "--backoff_max", type=float, default=60.0, help="Maximum backoff seconds"
)
parser.add_argument(
    "--jitter",
    type=float,
    default=0.2,
    help="Random jitter added to backoff in seconds",
)
parser.add_argument(
    "--min_request_interval",
    type=float,
    default=0.0,
    help="Minimum seconds between API requests",
)
parser.add_argument(
    "--run_id",
    type=str,
    default="",
    help="(Optional) Included in output filename for disambiguation",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random seed for jitter and sampling determinism",
)
parser.add_argument("--verbose", action="store_true", help="Print verbose output")
parser.add_argument(
    "--use_batch",
    action="store_true",
    help="Use Google Batch API (JSONL upload) for Google models",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=100,
    help="Number of requests per Google batch job when --use_batch is set",
)
parser.add_argument(
    "--batch_concurrency",
    type=int,
    default=4,
    help="Maximum number of Google batch jobs submitted in parallel",
)
parser.add_argument(
    "--batch_poll_interval",
    type=float,
    default=5.0,
    help="Seconds between polling Google batch jobs for completion",
)
args = parser.parse_args()

assert args.model_name in MODELS, f"Model {args.model_name} not in {MODELS}"
assert args.reasoning_effort in [
    "high",
    "none",
], f"Reasoning effort {args.reasoning_effort} not in [high, none]"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Google GenAI client (uses GEMINI_API_KEY or GOOGLE_API_KEY)
google_client = google_genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
)


random.seed(args.seed)

print(
    f"Evaluating {args.model_name} on {args.input_file} (single output file; resume={args.resume})"
)
# Load dataset
NAME = Path(args.input_file).stem
TASK = NAME.split("_")[0]
assert TASK in TASKS, f"Task {TASK} not in {TASKS}"
regression = TASK in REGRESSION_TASKS
if regression:
    SYSTEM_MESSAGE = NUMERIC_SYSTEM_MESSAGE
else:
    SYSTEM_MESSAGE = task_to_system_message(TASK)

# Determine output file path (same directory as input; single consolidated file)
in_path = Path(args.input_file)
safe_model = args.model_name.replace("/", "__").replace(":", "_")

if safe_model == "x-ai__grok-4":
    print(
        "WARNING: --reasoning_effort is not supported for this model. Standardizing to high."
    )
    args.reasoning_effort = "high"
elif safe_model == "openai__o3-pro":
    print(
        "WARNING: --reasoning_effort is not supported for this model. Standardizing to high."
    )
    args.reasoning_effort = "high"
elif safe_model == "qwen__qwen3-235b-a22b-thinking-2507":
    print(
        "WARNING: --reasoning_effort is not supported for this model. Standardizing to high."
    )
    args.reasoning_effort = "high"
elif safe_model == "qwen__qwen3-235b-a22b-2507":
    print(
        "WARNING: --reasoning_effort is not supported for this model. Standardizing to none."
    )
    args.reasoning_effort = "none"
elif safe_model == "openai__gpt-4.1":
    print(
        "WARNING: --reasoning_effort is not supported for this model. Standardizing to none."
    )
    args.reasoning_effort = "none"
elif safe_model == "moonshotai__kimi-k2-0905":
    print(
        "WARNING: --reasoning_effort is not supported for this model. Standardizing to none."
    )
    args.reasoning_effort = "none"
elif safe_model == "meta-llama__llama-4-maverick":
    print(
        "WARNING: --reasoning_effort is not supported for this model. Standardizing to none."
    )
    args.reasoning_effort = "none"

if args.reasoning_effort == "none":
    if safe_model == "openai__gpt-5":
        print(
            "WARNING: --reasoning_effort=none is not supported for this model. Converting to minimal."
        )
        args.reasoning_effort = "minimal"
    elif safe_model == "google__gemini-2.5-pro":
        print(
            "WARNING: --reasoning_effort=none is not supported for this model. Converting to minimal (128 tokens)."
        )
        args.reasoning_effort = "minimal"
    elif safe_model == "openai__gpt-oss-120b":
        print(
            "WARNING: --reasoning_effort=none is not supported for this model. Converting to low."
        )
        args.reasoning_effort = "low"


base_name_parts = [in_path.stem]
if args.run_id:
    base_name_parts.append(args.run_id)
base_name_parts.append(safe_model)
base_name_parts.append(f"re_{args.reasoning_effort}")
out_name = "_".join(base_name_parts) + ".csv"
file_path = str(in_path.parent / out_name)

# Read entire dataset; resume is handled by skipping already written rows later
df_chunk = pd.read_csv(args.input_file, dtype=str)

# Do not drop duplicates; preserve exact input order

# Few-shot examples (if requested)
fs_k = max(0, int(args.fs_examples)) if args.few_shot else 0
fs_examples = []
if fs_k > 0 and len(df_chunk) >= fs_k:
    for i in range(fs_k):
        fs_examples.append(
            (
                df_chunk.iloc[i]["text_prompt"],
                (
                    AI_MESSAGE.format(answer=df_chunk.iloc[i]["answer"])
                    if "answer" in df_chunk.columns
                    else ""
                ),
            )
        )

# Auto-resume by already written row count (absolute position within chunk)
already_done = 0
processed_set: set[str] = set()
retry_set: set[str] = set()
if args.resume and Path(file_path).exists():
    try:
        existing_df_full = pd.read_csv(file_path, dtype=str)
        if "text_prompt" in existing_df_full.columns:
            # Count only rows with non-empty raw_response as done; retry empties
            rr = existing_df_full.get("raw_response")
            non_empty_mask = (
                rr.notna()
                & (existing_df_full["raw_response"].astype(str).str.strip() != "")
                if rr is not None
                else pd.Series([], dtype=bool)
            )
            already_done = int(non_empty_mask.sum())
            processed_set = set(
                existing_df_full.loc[non_empty_mask, "text_prompt"].astype(str).tolist()
            )
            retry_set = set(
                existing_df_full.loc[~non_empty_mask, "text_prompt"]
                .astype(str)
                .tolist()
            )
        else:
            already_done = len(existing_df_full)
    except Exception as e:
        print(
            f"Warning: could not determine resume position from existing results: {e}",
            file=sys.stderr,
        )
        already_done = 0

# Build working dataframe: process all rows, resuming by absolute index
start_idx = 0
# Build retries first (rows present but with empty raw_response), then new/unprocessed rows
df_retries = (
    df_chunk[df_chunk["text_prompt"].astype(str).isin(retry_set)]
    if len(retry_set) > 0 and "text_prompt" in df_chunk.columns
    else df_chunk.iloc[0:0]
)
df_remaining = (
    df_chunk[~df_chunk["text_prompt"].astype(str).isin(processed_set)]
    if len(processed_set) > 0 and "text_prompt" in df_chunk.columns
    else df_chunk.copy()
)
# Exclude retries from remaining to avoid duplicates in this run
if len(df_retries) > 0:
    df_remaining = df_remaining[
        ~df_remaining["text_prompt"].astype(str).isin(retry_set)
    ]

df_work = pd.concat([df_retries, df_remaining], axis=0)

if args.limit and args.limit > 0:
    # Effective remaining rows allowed after counting filled outputs
    remaining = max(0, args.limit - len(processed_set))
    if remaining <= 0:
        print(
            f"Limit {args.limit} reached or exceeded (filled={len(processed_set)}). Skipping API calls and proceeding to post-processing."
        )
        df_work = df_chunk.iloc[0:0].copy()
    else:
        df_work = df_work.iloc[:remaining].copy()

records_iter = df_work.to_dict(orient="records")
total_remaining = len(records_iter)

results_buffer = list()
last_request_time = [0.0]
retry_updates_buffer = list()


def _gemini_thinking_from_effort() -> gtypes.ThinkingConfig:
    include_thoughts: bool
    thinking_budget: int
    if args.reasoning_effort == "none" and safe_model == "google__gemini-2.5-flash":
        include_thoughts = False
        thinking_budget = 0
    elif args.reasoning_effort == "minimal" and safe_model == "google__gemini-2.5-pro":
        include_thoughts = True
        thinking_budget = 128
    elif args.reasoning_effort == "high":
        if safe_model == "google__gemini-2.5-flash":
            include_thoughts = True
            thinking_budget = 24576
        elif safe_model == "google__gemini-2.5-pro":
            include_thoughts = True
            thinking_budget = 32768
        else:
            raise ValueError(
                f"Invalid reasoning effort {args.reasoning_effort} for model {safe_model}"
            )
    else:
        raise ValueError(
            f"Invalid reasoning effort {args.reasoning_effort} for model {safe_model}"
        )
    return gtypes.ThinkingConfig(
        thinking_budget=thinking_budget, include_thoughts=include_thoughts
    )


def safe_append_csv(df: pd.DataFrame, path: str) -> None:
    path_obj = Path(path)
    if not path_obj.exists():
        df.to_csv(path, index=False, mode="w", header=True)
        return
    # If file exists, ensure header is the union of old and new columns
    try:
        existing_header_df = pd.read_csv(path, nrows=0)
        existing_cols = list(existing_header_df.columns)
    except Exception:
        existing_cols = []
    new_cols = [c for c in df.columns if c not in existing_cols]
    if len(new_cols) == 0:
        # No header change; append aligned to existing columns order
        try:
            aligned = df.reindex(columns=existing_cols, fill_value="")
        except Exception:
            aligned = df
        aligned.to_csv(path, index=False, mode="a", header=False)
        return
    # Header must expand. Load entire existing data, add missing columns, rewrite, then append.
    try:
        existing_df = pd.read_csv(path, dtype=str)
    except Exception:
        existing_df = pd.DataFrame(columns=existing_cols)
    # Add missing columns to existing_df and df
    for col in new_cols:
        if col not in existing_df.columns:
            existing_df[col] = ""
    # unified order: existing + new in the order they appear in df
    unified_cols = existing_cols + new_cols
    existing_df = existing_df.reindex(columns=unified_cols, fill_value="")
    existing_df.to_csv(path, index=False, mode="w", header=True)
    # Now append new rows aligned to unified header
    aligned_new = df.reindex(columns=unified_cols, fill_value="")
    aligned_new.to_csv(path, index=False, mode="a", header=False)


def upsert_by_text_prompt(df_updates: pd.DataFrame, path: str) -> None:
    """Replace rows in CSV matching text_prompt where existing raw_response is empty; append if no match.

    Ensures header is union of existing and update columns and preserves row order otherwise.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        df_updates.to_csv(path, index=False, mode="w", header=True)
        return
    try:
        existing_df = pd.read_csv(path, dtype=str)
    except Exception:
        existing_df = pd.DataFrame()
    # Make sure columns union
    existing_cols = list(existing_df.columns)
    update_cols = list(df_updates.columns)
    union_cols = list(dict.fromkeys(existing_cols + update_cols))
    for col in union_cols:
        if col not in existing_df.columns:
            existing_df[col] = ""
        if col not in df_updates.columns:
            df_updates[col] = ""
    existing_df = existing_df.reindex(columns=union_cols, fill_value="")
    df_updates = df_updates.reindex(columns=union_cols, fill_value="")
    # Upsert by first empty raw_response match for each text_prompt
    if "text_prompt" not in existing_df.columns:
        # Fallback: append
        existing_df = pd.concat([existing_df, df_updates], axis=0, ignore_index=True)
    else:
        rr_col = existing_df.get("raw_response")
        empty_mask = (
            rr_col.isna() | (existing_df["raw_response"].astype(str).str.strip() == "")
            if rr_col is not None
            else pd.Series([False] * len(existing_df))
        )
        for _, upd_row in df_updates.iterrows():
            tp = str(upd_row.get("text_prompt", ""))
            if tp == "":
                # Append if we cannot match
                existing_df.loc[len(existing_df)] = upd_row
                continue
            candidates = existing_df.index[
                (existing_df["text_prompt"].astype(str) == tp) & (empty_mask)
            ]
            if len(candidates) > 0:
                idx = int(candidates[0])
                # Replace row fields with update values
                for col in union_cols:
                    existing_df.at[idx, col] = upd_row[col]
                # mark as non-empty now
                empty_mask.at[idx] = False
            else:
                # No empty match; append as new row
                existing_df.loc[len(existing_df)] = upd_row
    existing_df.to_csv(path, index=False, mode="w", header=True)


def _usage_to_dict(completion) -> dict | None:
    try:
        # Try direct attribute
        if hasattr(completion, "usage") and completion.usage is not None:
            usage_obj = completion.usage
            # Pydantic v2
            try:
                return usage_obj.model_dump()
            except Exception:
                pass
            # Pydantic v1 / SDK helper
            try:
                return usage_obj.to_dict()
            except Exception:
                pass
            # Fallback to asdict-like repr
            try:
                return dict(usage_obj)
            except Exception:
                pass
        # Try full dict
        try:
            comp_dict = completion.to_dict()
            return comp_dict.get("usage")
        except Exception:
            return None
    except Exception:
        return None


def extract_token_count(usage: dict | None) -> int:
    """Return best-available token count preference: reasoning -> completion -> total -> output -> -1.

    Supports nested OpenRouter fields under completion_tokens_details and prompt_tokens_details.
    """
    if not isinstance(usage, dict):
        return -1
    # direct keys
    for key in (
        "reasoning_tokens",
        "completion_tokens",
        "total_tokens",
        "output_tokens",
    ):
        if key in usage and usage[key] is not None:
            try:
                return int(usage[key])
            except Exception:
                try:
                    return int(float(usage[key]))
                except Exception:
                    continue
    # nested keys
    try:
        nested_reason = usage.get("completion_tokens_details", {}).get(
            "reasoning_tokens"
        )
        if nested_reason is not None:
            return int(nested_reason)
    except Exception:
        pass
    return -1


def _flatten_usage(usage: dict | None, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested usage dict into dot-joined keys.

    Example: {"completion_tokens_details": {"reasoning_tokens": 1}} ->
    {"completion_tokens_details.reasoning_tokens": 1}
    """
    if not isinstance(usage, dict):
        return {}
    items = {}
    for k, v in usage.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_usage(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def call_with_retries(messages):
    last_exc = None
    for attempt in range(args.max_retries + 1):
        # Throttle if needed
        now = time.time()
        wait_needed = args.min_request_interval - (now - last_request_time[0])
        if wait_needed > 0:
            time.sleep(wait_needed)
        try:
            # Google Gemini models: route via Google GenAI SDK
            if safe_model.startswith("google__"):
                include_thoughts: bool
                thinking_budget: int
                if args.reasoning_effort == "none":
                    include_thoughts = False
                    thinking_budget = 0
                elif (
                    args.reasoning_effort == "minimal"
                    and safe_model == "google__gemini-2.5-pro"
                ):
                    include_thoughts = True
                    thinking_budget = 128
                else:
                    if safe_model == "google__gemini-2.5-flash":
                        include_thoughts = True
                        thinking_budget = 24576
                    elif safe_model == "google__gemini-2.5-pro":
                        include_thoughts = True
                        thinking_budget = 32768
                    else:
                        include_thoughts = True
                        thinking_budget = -1

                thinking_cfg = gtypes.ThinkingConfig(
                    thinking_budget=thinking_budget, include_thoughts=include_thoughts
                )

                # Convert messages to Gemini contents and system instruction
                contents: list[gtypes.Content] = []
                for m in messages:
                    role = m.get("role", "user")
                    text = m.get("content", "")
                    if not isinstance(text, str) or text == "":
                        continue
                    if role == "system":
                        continue
                    mapped_role = "user" if role == "user" else "model"
                    contents.append(
                        gtypes.Content(
                            role=mapped_role, parts=[gtypes.Part.from_text(text=text)]
                        )
                    )

                config = gtypes.GenerateContentConfig(
                    temperature=0,
                    thinking_config=thinking_cfg,
                    system_instruction=gtypes.Part.from_text(text=SYSTEM_MESSAGE),
                    response_mime_type="application/json",
                    response_schema=build_gemini_response_schema(TASK),
                    max_output_tokens=128,
                )

                model_name = (
                    args.model_name.split("/", 1)[1]
                    if "/" in args.model_name
                    else args.model_name
                )

                # The SDK accepts either a single Content or a union; pass single when list has one
                contents_arg: gtypes.ContentListUnionDict
                if len(contents) == 0:
                    contents_arg = gtypes.Content(
                        role="user", parts=[gtypes.Part.from_text(text="")]
                    )
                elif len(contents) == 1:
                    contents_arg = contents[0]
                else:
                    # Merge multiple turns into a single user Content text to satisfy type checker
                    merged_segments: list[str] = []
                    for c in contents:
                        role_label = (c.role or "user").capitalize()
                        part_texts = []
                        for p in c.parts or []:
                            t = getattr(p, "text", None)
                            if isinstance(t, str) and t:
                                part_texts.append(t)
                        if part_texts:
                            merged_segments.append(
                                f"{role_label}: " + " ".join(part_texts)
                            )
                    merged_text = "\n\n".join(merged_segments)
                    contents_arg = gtypes.Content(
                        role="user", parts=[gtypes.Part.from_text(text=merged_text)]
                    )
                resp = google_client.models.generate_content(
                    model=model_name, contents=contents_arg, config=config
                )
                last_request_time[0] = time.time()

                content = getattr(resp, "text", None)

                um = getattr(resp, "usage_metadata", None)
                usage: dict[str, object] | None = None
                if um is not None:
                    usage = {
                        "usage_metadata": {
                            "prompt_token_count": getattr(
                                um, "prompt_token_count", None
                            ),
                            "candidates_token_count": getattr(
                                um, "candidates_token_count", None
                            ),
                            "total_token_count": getattr(um, "total_token_count", None),
                            "thoughts_token_count": getattr(
                                um, "thoughts_token_count", None
                            ),
                        }
                    }
                    try:
                        pt = getattr(um, "prompt_token_count", None)
                        rt = getattr(um, "thoughts_token_count", None)
                        ct = getattr(um, "candidates_token_count", None)
                        tt = getattr(um, "total_token_count", None)
                        if pt is not None:
                            usage["prompt_tokens"] = pt
                        if rt is not None:
                            usage["reasoning_tokens"] = rt
                        if ct is not None:
                            usage["completion_tokens"] = ct
                            usage["output_tokens"] = ct
                        if tt is not None:
                            usage["total_tokens"] = tt
                        if pt is not None and tt is not None:
                            usage["tokens"] = tt - pt
                    except Exception:
                        pass
                else:
                    usage = {}

                reasoning_segments: list[str] = []
                try:
                    for cand in getattr(resp, "candidates", []) or []:
                        content_obj = getattr(cand, "content", None)
                        if content_obj is None:
                            continue
                        for part in getattr(content_obj, "parts", []) or []:
                            try:
                                if getattr(part, "thought", False) and isinstance(
                                    getattr(part, "text", None), str
                                ):
                                    reasoning_segments.append(part.text)
                            except Exception:
                                continue
                except Exception:
                    pass
                reasoning = " ".join(s.strip() for s in reasoning_segments if s).strip()

                return content, usage, reasoning

            # Build extra_body with usage
            extra_body: dict[str, object] = {"usage": {"include": True}}

            # Add reasoning effort if supported
            if (
                not safe_model == "meta-llama__llama-4-maverick"
                and not safe_model == "moonshotai__kimi-k2-0905"
                and not safe_model == "qwen__qwen3-235b-a22b-2507"
            ):
                if args.reasoning_effort == "none":
                    extra_body["reasoning"] = {"max_tokens": 0, "enabled": False}
                else:
                    extra_body["reasoning"] = {"effort": args.reasoning_effort}

            # Enforce JSON response format via extra_body for OpenRouter
            if (
                not (
                    safe_model
                    == "deepseek__deepseek-chat-v3.1"  # For some reason adding the format disables reasoning for DeepSeek
                    and args.reasoning_effort == "high"
                )
                and not safe_model == "qwen__qwen3-235b-a22b-thinking-2507"
                and not safe_model == "qwen__qwen3-235b-a22b-2507"
                and not safe_model == "meta-llama__llama-4-maverick"
            ):
                extra_body["response_format"] = build_response_format(TASK)

            extra_body["provider"] = {"require_parameters": True}

            # Provider ignore handling for empty content:
            provider_obj: dict[str, object] = {}
            existing_provider = extra_body.get("provider")
            if isinstance(existing_provider, dict):
                # Copy existing keys
                provider_obj.update(existing_provider)
            provider_obj["require_parameters"] = True
            ignore_list: list[str] = []
            existing_ignore = provider_obj.get("ignore")
            if isinstance(existing_ignore, list):
                ignore_list = [str(x) for x in existing_ignore]
            provider_obj["ignore"] = ignore_list
            extra_body["provider"] = provider_obj

            completion = client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                temperature=(
                    0 if safe_model != "openai__gpt-5" else NOT_GIVEN
                ),  # GPT 5 does not support temperature
                extra_body=extra_body,
            )
            last_request_time[0] = time.time()
            content = completion.choices[0].message.content
            usage = _usage_to_dict(completion)
            # Extract reasoning if available
            try:
                comp_dict = completion.to_dict()
            except Exception:
                comp_dict = {}
            reasoning = ""
            try:
                choices = comp_dict.get("choices", [])
                if isinstance(choices, list) and len(choices) > 0:
                    msg = (
                        choices[0].get("message", {})
                        if isinstance(choices[0], dict)
                        else {}
                    )
                    r = msg.get("reasoning")
                    if isinstance(r, str):
                        reasoning = r
                    elif isinstance(r, dict):
                        reasoning = (
                            r.get("content")
                            if isinstance(r.get("content"), str)
                            else json.dumps(r, ensure_ascii=False)
                        )
                    elif isinstance(r, list):
                        parts = []
                        for seg in r:
                            if isinstance(seg, dict) and "content" in seg:
                                parts.append(str(seg["content"]))
                            else:
                                parts.append(str(seg))
                        reasoning = " ".join(p.strip() for p in parts if p)
                    if not reasoning:
                        for alt in (
                            "thinking",
                            "chain_of_thought",
                            "cot",
                            "reasoning_content",
                        ):
                            v = msg.get(alt)
                            if isinstance(v, str):
                                reasoning = v
                                break
                            if isinstance(v, dict):
                                if isinstance(v.get("content"), str):
                                    reasoning = v["content"]
                                    break
                                try:
                                    reasoning = json.dumps(v, ensure_ascii=False)
                                    break
                                except Exception:
                                    pass
                    if (
                        not reasoning
                        and isinstance(choices, list)
                        and len(choices) > 0
                        and isinstance(choices[0], dict)
                    ):
                        v = choices[0].get("reasoning")
                        if isinstance(v, str):
                            reasoning = v
                        elif isinstance(v, dict):
                            if isinstance(v.get("content"), str):
                                reasoning = v["content"]
                            else:
                                reasoning = json.dumps(v, ensure_ascii=False)
            except Exception:
                pass
            # If content is empty, add current provider to ignore and retry up to 3 times
            if (content is None or str(content).strip() == "") and attempt < min(
                3, args.max_retries
            ):
                prov = getattr(completion, "provider", None)
                prov_name = None
                try:
                    comp_dict2 = completion.to_dict()
                    prov_field = comp_dict2.get("provider")
                    if isinstance(prov_field, dict):
                        prov_name = (
                            prov_field.get("name")
                            if isinstance(prov_field.get("name"), str)
                            else None
                        )
                    else:
                        prov_name = None
                except Exception:
                    prov_name = None
                if isinstance(prov, dict) and isinstance(prov.get("name"), str):
                    prov_name = str(prov.get("name"))
                if prov_name and prov_name not in ignore_list:
                    ignore_list.append(prov_name)
                    provider_obj["ignore"] = ignore_list
                    extra_body["provider"] = provider_obj
                    if args.verbose:
                        print(
                            f"Empty content received. Ignoring provider '{prov_name}' and retrying...",
                            file=sys.stderr,
                        )
                    continue
            return content, usage, reasoning
        except Exception as e:
            last_exc = e
            if attempt >= args.max_retries:
                break
            backoff = min(
                args.backoff_max, (args.backoff_base**attempt)
            ) + random.uniform(0, args.jitter)
            if args.verbose:
                print(
                    f"Request failed (attempt {attempt+1}/{args.max_retries}). Retrying in {backoff:.2f}s. Error: {e}",
                    file=sys.stderr,
                )
            time.sleep(backoff)
    raise last_exc if last_exc else RuntimeError("Request failed with unknown error")


def process_one(row_idx: int, row_dict: dict) -> None:
    """Process a single dataset row: call model, build record, and buffer save."""
    try:
        if args.few_shot and fs_examples:
            messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
            for u, a in fs_examples:
                messages.append({"role": "user", "content": u})
                messages.append({"role": "assistant", "content": a})
            messages.append({"role": "user", "content": f"{row_dict['text_prompt']}"})
            # messages.append({"role": "assistant", "content": AI_MESSAGE_PREFIX})
        else:
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": f"{row_dict['text_prompt']}"},
                # {"role": "assistant", "content": AI_MESSAGE_PREFIX}
            ]
        response_raw, usage, reasoning_raw = call_with_retries(messages)
        # Preserve raw response exactly as returned
        if response_raw is None:
            response_clean = ""
        else:
            # Normalize to single line to avoid CSV multi-line cell issues
            response_clean = (
                str(response_raw)
                .replace("\r\n", "\n")
                .replace("\r", "\n")
                .replace("\n", " ")
                .strip()
            )
        token_count = extract_token_count(usage)
        usage_flat = _flatten_usage(usage)
        reasoning_clean = ""
        if reasoning_raw:
            reasoning_clean = (
                str(reasoning_raw)
                .replace("\r\n", "\n")
                .replace("\r", "\n")
                .replace("\n", " ")
                .strip()
            )
        # Never parse here; defer parsing to the end
        record = {
            **row_dict,
            "raw_response": response_clean,
            "tokens": token_count,
            "reasoning": reasoning_clean,
            **{f"usage.{k}": v for k, v in usage_flat.items()},
        }

        # If this is a retry fix (was previously empty), collect for upsert; else append normally
        if len(retry_set) > 0 and str(row_dict.get("text_prompt", "")) in retry_set:
            retry_updates_buffer.append(record)
            if len(retry_updates_buffer) >= args.save_interval:
                upsert_by_text_prompt(pd.DataFrame(retry_updates_buffer), file_path)
                retry_updates_buffer.clear()
        else:
            results_buffer.append(record)
            if len(results_buffer) >= args.save_interval:
                results_df = pd.DataFrame(results_buffer)
                safe_append_csv(results_df, file_path)
                results_buffer.clear()
    except Exception as e:
        if len(results_buffer) > 0:
            results_df = pd.DataFrame(results_buffer)
            safe_append_csv(results_df, file_path)
            results_buffer.clear()
        print(f"Error during completion: {e}", file=sys.stderr)
        raise e


if args.use_batch and safe_model.startswith("google__"):
    # Concurrent Google batch jobs: shard requests and submit up to batch_concurrency at a time
    model_name = (
        args.model_name
        if args.model_name.startswith("models/")
        else f"models/{args.model_name.split('/',1)[1]}"
    )
    thinking_cfg = _gemini_thinking_from_effort()
    response_schema = build_gemini_response_schema(TASK)

    # Materialize work rows and shard
    work_rows: list[dict] = list(df_work.to_dict(orient="records"))
    if len(work_rows) == 0:
        pass
    else:

        def build_inlined_requests(rows: list[dict]) -> list[dict]:
            reqs: list[dict] = []
            for row_dict in rows:
                reqs.append(
                    {
                        "model": model_name,
                        "contents": [
                            {
                                "role": "user",
                                "parts": [
                                    {"text": str(row_dict.get("text_prompt", ""))}
                                ],
                            }
                        ],
                        "config": gtypes.GenerateContentConfig(
                            temperature=0,
                            response_mime_type="application/json",
                            response_schema=response_schema,
                            thinking_config=thinking_cfg,
                            system_instruction=gtypes.Part.from_text(
                                text=SYSTEM_MESSAGE
                            ),
                            # max_output_tokens=128,
                        ),
                    }
                )
            return reqs

        chunks: list[list[dict]] = [
            work_rows[i : i + int(max(1, args.batch_size))]
            for i in range(0, len(work_rows), int(max(1, args.batch_size)))
        ]

        pending: list[tuple[int, list[dict]]] = list(enumerate(chunks))
        active: list[dict] = []

        def submit_next() -> None:
            if not pending:
                return
            if len(active) >= int(max(1, args.batch_concurrency)):
                return
            chunk_id, rows = pending.pop(0)
            inlined_requests = build_inlined_requests(rows)
            display_name = f"{Path(file_path).stem}-part-{chunk_id:04d}"
            job = google_client.batches.create(
                model=model_name,
                src=cast(Any, inlined_requests),
                config={"display_name": display_name},
            )
            if args.verbose:
                print(f"Created batch job {chunk_id}: {getattr(job, 'name', '')}")
            active.append({"job": job, "rows": rows, "chunk_id": chunk_id})

        # Prime initial submissions
        while len(active) < int(max(1, args.batch_concurrency)) and pending:
            submit_next()

        poll_interval = float(max(0.5, args.batch_poll_interval))
        while active or pending:
            # Poll active jobs
            i = 0
            while i < len(active):
                entry = active[i]
                job = entry.get("job")
                chunk_rows = entry.get("rows", [])
                chunk_id = entry.get("chunk_id")
                name = str(getattr(job, "name", "") or "")
                try:
                    job_cur = google_client.batches.get(name=name)
                except Exception as e:
                    raise RuntimeError(f"Failed to poll batch job {chunk_id}: {e}")
                state = getattr(job_cur, "state", None)
                if state in (
                    "JOB_STATE_SUCCEEDED",
                    "JOB_STATE_FAILED",
                    "JOB_STATE_CANCELLED",
                    "JOB_STATE_EXPIRED",
                ):
                    # Remove from active
                    active.pop(i)
                    if state != "JOB_STATE_SUCCEEDED":
                        raise RuntimeError(
                            f"Batch job {chunk_id} did not succeed. Final state: {state}"
                        )

                    # Collect results
                    results: list[dict] = []
                    inlined = getattr(job_cur, "dest", None)
                    if inlined and getattr(inlined, "inlined_responses", None):
                        for ir in inlined.inlined_responses:
                            try:
                                results.append(ir.response.model_dump())
                            except Exception:
                                results.append({})
                    else:
                        dest = getattr(job_cur, "dest", None)
                        file_name = getattr(dest, "file_name", None)
                        if not file_name:
                            raise RuntimeError(
                                f"No results found for batch job {chunk_id}"
                            )
                        buf = google_client.files.download(file=file_name)
                        text = (
                            buf.decode("utf-8") if hasattr(buf, "decode") else str(buf)
                        )
                        for line in text.splitlines():
                            if not line.strip():
                                continue
                            try:
                                results.append(json.loads(line))
                            except Exception:
                                results.append({})

                    # Map results to rows and write incrementally
                    out_rows: list[dict] = []
                    for row_dict, resp_obj in zip(chunk_rows, results):
                        finish_reasons = []
                        try:
                            text_parts = []
                            for cand in resp_obj.get("candidates") or []:
                                finish_reasons.append(cand.get("finish_reason"))
                                content = cand.get("content") or {}
                                for part in content.get("parts") or []:
                                    if isinstance(
                                        part.get("text"), str
                                    ) and not part.get("thought", False):
                                        text_parts.append(part["text"])
                            response_text = " ".join(text_parts).strip()
                        except Exception:
                            response_text = ""
                        reasoning_parts = []
                        try:
                            for cand in resp_obj.get("candidates") or []:
                                content = cand.get("content") or {}
                                for part in content.get("parts") or []:
                                    if part.get("thought", False) and isinstance(
                                        part.get("text"), str
                                    ):
                                        reasoning_parts.append(part["text"])
                        except Exception:
                            pass
                        reasoning_text = " ".join(
                            p.strip() for p in reasoning_parts if p
                        ).strip()
                        if (
                            response_text == ""
                            and finish_reasons
                            and finish_reasons[0] == "MAX_TOKENS"
                        ):
                            response_text = "Token limit reached"
                        usage_obj = resp_obj.get("usage_metadata", {})
                        prompt_tokens = usage_obj.get("prompt_token_count")
                        candidates_tokens = usage_obj.get("candidates_token_count")
                        thoughts_tokens = usage_obj.get("thoughts_token_count")
                        total_tokens = usage_obj.get("total_token_count")
                        usage = {
                            "usage_metadata": {
                                "prompt_token_count": prompt_tokens,
                                "candidates_token_count": candidates_tokens,
                                "thoughts_token_count": thoughts_tokens,
                                "total_token_count": total_tokens,
                            },
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": candidates_tokens,
                            "output_tokens": candidates_tokens,
                            "reasoning_tokens": thoughts_tokens,
                            "total_tokens": total_tokens,
                        }
                        token_count = extract_token_count(usage)
                        usage_flat = _flatten_usage(usage)
                        out_rows.append(
                            {
                                **row_dict,
                                "raw_response": response_text,
                                "tokens": token_count,
                                "reasoning": reasoning_text,
                                **{f"usage.{k}": v for k, v in usage_flat.items()},
                            }
                        )

                    if out_rows:
                        df_batch = pd.DataFrame(out_rows)
                        if len(df_batch) > 0:
                            if len(retry_set) > 0 and "text_prompt" in df_batch.columns:
                                mask_retry = (
                                    df_batch["text_prompt"].astype(str).isin(retry_set)
                                )
                                df_retry = df_batch[mask_retry]
                                df_new = df_batch[~mask_retry]
                            else:
                                df_retry = df_batch.iloc[0:0]
                                df_new = df_batch
                            if len(df_retry) > 0:
                                retry_updates_buffer.extend(
                                    df_retry.to_dict(orient="records")
                                )
                                upsert_by_text_prompt(
                                    pd.DataFrame(retry_updates_buffer), file_path
                                )
                                retry_updates_buffer.clear()
                            if len(df_new) > 0:
                                safe_append_csv(df_new, file_path)

                    # After processing, attempt to submit next pending chunk
                    submit_next()
                else:
                    i += 1

            if active or pending:
                time.sleep(poll_interval)
else:
    if args.verbose:
        tq_bar = tqdm(records_iter, desc="sample", leave=False, total=total_remaining)
        for row_idx, row_dict in enumerate(tq_bar):
            tq_bar.set_description(str(row_dict["text_prompt"])[:80])
            process_one(row_idx, row_dict)
        tq_bar.close()
    else:
        for row_idx, row_dict in enumerate(records_iter):
            process_one(row_idx, row_dict)
    if len(retry_updates_buffer) > 0:
        upsert_by_text_prompt(pd.DataFrame(retry_updates_buffer), file_path)
        retry_updates_buffer.clear()
    if len(results_buffer) > 0:
        results_df = pd.DataFrame(results_buffer)
        safe_append_csv(results_df, file_path)
        print(f"Saved raw responses to {file_path}")

print("Finished.")
