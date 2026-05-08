import os
from argparse import ArgumentParser
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from random import seed
from typing import Literal

import numpy as np
import polars as pl
from data_gen_utils import MockPool, save_df_phase
from tqdm import tqdm


def _process_chunk_interval(chunk_data):
    """Process a chunk of interval data efficiently."""
    rows_data, range_names = chunk_data
    results = []
    
    for row_dict in rows_data:
        list1_clean = row_dict['list1'].replace("'", "")
        list1_str = list1_clean[1:-1].split(", ")
        ref = row_dict['ref']
        list_len = int(row_dict['list_len'])
        position = int(row_dict['position'])
        
        text_prompt = f"What interval does x={ref} belong to? A: x < {list1_str[0]},"
        prompt = f"<|intva|>{ref}:"
        post = None
        for range_name, num in zip(range_names, list1_str):
            prev = post
            post = num
            if prev is not None:
                text_prompt += f" {range_name}: {prev} <= x < {post},"
            prompt += f" {range_name} {num}"
        text_prompt += f" {range_names[list_len]}: {list1_str[-1]} <= x"
        prompt += f" {range_names[list_len]}"
        answer = range_names[position]
        
        result = {**row_dict, "prompt": prompt, "text_prompt": text_prompt, "answer": answer}
        results.append(result)
    
    return results


def _process_chunk_mean(chunk_data):
    """Process a chunk of mean data efficiently."""
    rows_data = chunk_data
    results = []
    
    for row_dict in rows_data:
        list1_clean = row_dict['list1'].replace("'", "")
        text_prompt = f"What is the mean of the list {list1_clean}?"
        prompt = "<|mean|>" + list1_clean[1:-1]
        answer = row_dict["mean"]
        
        result = {**row_dict, "prompt": prompt, "text_prompt": text_prompt, "answer": answer}
        results.append(result)
    
    return results


def _process_chunk_std(chunk_data):
    """Process a chunk of std data efficiently."""
    rows_data = chunk_data
    results = []
    
    for row_dict in rows_data:
        list1_clean = row_dict['list1'].replace("'", "")
        text_prompt = f"What is the std of the list {list1_clean}?"
        prompt = "<|std|>" + list1_clean[1:-1]
        answer = row_dict["std"]
        
        result = {**row_dict, "prompt": prompt, "text_prompt": text_prompt, "answer": answer}
        results.append(result)
    
    return results


def convert_to_eval_prompts(df_phase: pl.DataFrame, phase: Literal["train", "val", "test"], path, seed=42, num_workers: int=1) -> pl.DataFrame:
    # --- Vectorized cases ---
    if "tnc" in path.name.lower():
        n = df_phase.height
        rng = np.random.default_rng(seed)
        ops = rng.choice(["<", ">"], size=n)
        op_texts = np.where(
            rng.random(n) < 0.5,
            np.where(ops == "<", "less than", "greater than"),
            ops
        )
        df_phase = df_phase.with_columns([
            pl.Series("operator", ops),
            pl.Series("text_operator", op_texts),
        ])
        df_phase = df_phase.with_columns([
            (pl.col("num1") + " " + pl.col("operator") + " " + pl.col("num2")).alias("prompt"),
            ("Is " + pl.col("num1") + " " + pl.col("text_operator") + " " + pl.col("num2") + "?").alias("text_prompt"),
            pl.when(pl.col("operator") == "<")
                .then(pl.col("num1").cast(float) < pl.col("num2").cast(float))
                .otherwise(pl.col("num1").cast(float) > pl.col("num2").cast(float))
                .alias("answer")
        ])
        return df_phase

    elif "add" in path.name.lower():
        df_phase = df_phase.with_columns([
            (pl.col("num1") + " " + pl.col("operator") + " " + pl.col("num2")).alias("prompt"),
            ("What is " + pl.col("num1") + " " + pl.col("operator") + " " + pl.col("num2") + "?").alias("text_prompt"),
            pl.col("sum").alias("answer")
        ])
        return df_phase

    elif "mul" in path.name.lower():
        df_phase = df_phase.with_columns([
            (pl.col("num1") + " * " + pl.col("num2")).alias("prompt"),
            ("What is " + pl.col("num1") + " * " + pl.col("num2") + "?").alias("text_prompt"),
            pl.col("prod").alias("answer")
        ])
        return df_phase

    elif "div" in path.name.lower():
        df_phase = df_phase.with_columns([
            (pl.col("num1") + " / " + pl.col("num2")).alias("prompt"),
            ("What is " + pl.col("num1") + " / " + pl.col("num2") + "?").alias("text_prompt"),
            pl.col("quot").alias("answer")
        ])
        return df_phase

    elif "exp" in path.name.lower():
        df_phase = df_phase.with_columns([
            (pl.col("num1") + " ^ " + pl.col("num2")).alias("prompt"),
            ("What is " + pl.col("num1") + " ^ " + pl.col("num2") + "?").alias("text_prompt"),
            pl.col("exp").alias("answer")
        ])
        return df_phase

    elif "repeat" in path.name.lower():
        df_phase = df_phase.with_columns([
            ("<|repeat|>" + pl.col("num1")).alias("prompt"),
            ("Repeat the number " + pl.col("num1") + ".").alias("text_prompt"),
            pl.col("num1").alias("answer")
        ])
        return df_phase

    elif "char_repeat" in path.name.lower():
        df_phase = df_phase.with_columns([
            ("<|char|>" + pl.col("digit_pos") + " " + pl.col("num1")).alias("prompt"),
            ("What is the " + pl.col("digit_pos") + ". digit of " + pl.col("num1") + "?").alias("text_prompt"),
            pl.col("digit").alias("answer")
        ])
        return df_phase

    elif "minmax" in path.name.lower():
        n = df_phase.height
        rng = np.random.default_rng(seed)
        ops = rng.choice(["minimum", "maximum"], size=n)
        df_phase = df_phase.with_columns([
            pl.Series("operator", ops),
            pl.col("list1").str.replace_all("'", "").alias("list1_clean"),
        ])
        df_phase = df_phase.with_columns([
            (pl.lit("<|") + pl.col("operator").str.slice(0, 3) + pl.lit("|>") + pl.col("list1_clean").str.replace("]", "").str.replace(r"\[", "")).alias("prompt"),
            ("What is the " + pl.col("operator") + " of the list " + pl.col("list1_clean") + "?").alias("text_prompt"),
            pl.when(pl.col("operator") == "minimum")
                .then(pl.col("minimum"))
                .otherwise(pl.col("maximum"))
                .alias("answer"),
            pl.when(pl.col("operator") == "minimum")
                .then(pl.col("minimum_difficulty"))
                .otherwise(pl.col("maximum_difficulty"))
                .alias("difficulty"),
            pl.when(pl.col("operator") == "minimum")
                .then(pl.col("minimum_difficulty_sd"))
                .otherwise(pl.col("maximum_difficulty_sd"))
                .alias("difficulty_sd")
        ])
        return df_phase

    elif "sort" in path.name.lower():
        n = df_phase.height
        rng = np.random.default_rng(seed)
        ops = rng.choice(["asc", "desc"], size=n)
        df_phase = df_phase.with_columns([
            pl.Series("operator", ops),
            pl.col("list1").str.replace_all("'", "").alias("list1_clean"),
        ])
        df_phase = df_phase.with_columns([
            (pl.lit("<|") + pl.col("operator") + pl.lit("|>") + pl.col("list1_clean").str.replace("]", "").str.replace(r"\[", "")).alias("prompt"),
            ("Sort the list " + pl.col("list1_clean") + " in " + pl.col("operator") + "ending order.").alias("text_prompt"),
            pl.when(pl.col("operator") == "asc")
                .then(pl.col("asc").str.replace("]", "").str.replace(r"\[", "").str.replace_all("'", ""))
                .otherwise(pl.col("desc").str.replace("]", "").str.replace(r"\[", "").str.replace_all("'", ""))
                .alias("answer")
        ])
        return df_phase

    # --- Non-vectorizable cases: fallback to row-wise (apply) ---
    # These cases involve randomness or complex logic
    if "interval" in path.name.lower():
        task_type = "interval"
    elif "mean" in path.name.lower():
        task_type = "mean"
    elif "std" in path.name.lower():
        task_type = "std"
    else:
        raise NotImplementedError()
        
    # For non-vectorizable cases, use row-wise apply with Pool/MockPool
    # Convert to efficient data structures for processing
    df_dicts = df_phase.to_dicts()
    
    # Determine chunk size and processing function
    chunk_size = max(1, len(df_dicts) // (num_workers * 4)) if num_workers > 1 else len(df_dicts)
    
    if task_type == "interval":
        process_func = _process_chunk_interval
        range_names = ["A", "B", "C", "D", "E", "F"]
        chunks = [(df_dicts[i:i + chunk_size], range_names) for i in range(0, len(df_dicts), chunk_size)]
    elif task_type == "mean":
        process_func = _process_chunk_mean
        chunks = [df_dicts[i:i + chunk_size] for i in range(0, len(df_dicts), chunk_size)]
    elif task_type == "std":
        process_func = _process_chunk_std
        chunks = [df_dicts[i:i + chunk_size] for i in range(0, len(df_dicts), chunk_size)]
    else:
        raise NotImplementedError()
    
    # Use Pool or MockPool based on num_workers
    pool = MockPool() if num_workers <= 1 else Pool(num_workers)
    chunk_results = list(tqdm(
        pool.imap(process_func, chunks),
        total=len(chunks),
        desc=f'Creating {phase} prompts'
    ))
    pool.close()
    pool.join()
    
    # Flatten results
    records = [record for chunk_result in chunk_results for record in chunk_result]
    df_phase_out = pl.DataFrame(records)
    return df_phase_out

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_train_path_glob", type=str, help="Path to CSV file with numbers")
    parser.add_argument("--output_folder", type=str, help="Output folder")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed(args.seed)

    args.output_folder = Path(args.output_folder)
    os.makedirs(args.output_folder, exist_ok=True)

    data_train_paths = glob(args.data_train_path_glob)
    if len(data_train_paths) == 0:
        raise ValueError(f"No files found for {args.data_train_path_glob}")
    print(f"Found {len(data_train_paths)} file(s) for {args.data_train_path_glob}")
    data_train_paths = [Path(path) for path in data_train_paths]
    for path in data_train_paths:
        df_train = pl.read_csv(path, infer_schema_length=0, encoding="utf8-lossy", rechunk=True)
        
        val_path = path.name.split("train")[0]
        val_path_glob = f"{path.parent}/{val_path}val*.csv.gz"
        # get the first matching file for validation
        val_files = glob(val_path_glob)
        if len(val_files) == 0:
            raise ValueError(f"No validation files found for {val_path_glob}")
        if len(val_files) > 1:
            print(f"Multiple validation files found for {val_path_glob}, using the first one: {val_files[0]}")
        data_val_path = Path(val_files[0])
        df_val = pl.read_csv(data_val_path, infer_schema_length=0, encoding="utf8-lossy", rechunk=True)

        test_path = path.name.split("train")[0]
        test_path_glob = f"{path.parent}/{test_path}test*.csv.gz"
        # get the first matching file for test
        test_files = glob(test_path_glob)
        if len(test_files) == 0:
            raise ValueError(f"No test files found for {test_path_glob}")
        if len(test_files) > 1:
            print(f"Multiple test files found for {test_path_glob}, using the first one: {test_files[0]}")
        data_test_path = Path(test_files[0])
        df_test = pl.read_csv(data_test_path, infer_schema_length=0, encoding="utf8-lossy", rechunk=True)

        paths: dict[Literal["train", "val", "test"], Path] = {
            'train': path,
            'val': data_val_path,
            'test': data_test_path.parent / data_test_path.stem # removes the .gz extension
        }
        for phase, df, path in tqdm(zip(paths.keys(), [df_train, df_val, df_test], [path, data_val_path, data_test_path]), total=3, desc="Converting to eval prompts"):
            df = convert_to_eval_prompts(df, phase, path, seed=args.seed)
            save_df_phase(df, phase, args.output_folder, path)
