import math
import multiprocessing as mp
import os
import sys

if True:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path
from time import time
from typing import Literal

import polars as pl
from convert_to_eval_prompts import convert_to_eval_prompts
from data_gen_utils import (
    Generation_settings,
    MockPool,
    SignificantDigitsDistribution,
    Task,
    get_strat_params,
    plot_data_distribution,
    plot_difficulty_histogram,
    save_df_phase,
)
from tasks import generate_dataset
from tqdm.auto import tqdm

tasks: list[Task] = [t for t in Task if t != Task.TEXT]

parser = ArgumentParser()
parser.add_argument("--save_dir", type=str, help="Path to save directory")
parser.add_argument("--num_train_samples", type=int, default=30_000_000, help="Number of train samples to generate. Default is 30 million.")
parser.add_argument("--num_val_samples", type=int, default=10_000, help="Number of validation samples to generate. Default is 10,000.")
parser.add_argument("--num_test_samples", type=int, default=10_000, help="Number of test samples to generate. Default is 10,000.")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--tasks", nargs="+", default=["all"], help="Tasks to generate. By default, generate all tasks. Use 'all' to generate all tasks or specify a list of tasks.")

parser.add_argument("--base", type=int, default=10, help="Base for the numbers. Default is 10. Use 2 for binary numbers.")
parser.add_argument("--min_exponent", type=int, default=-14, help="Minimum exponent for the given base for the numbers. If None, it will be set to min_num.")
parser.add_argument("--max_exponent", type=int, default=15, help="Maximum exponent for the numbers. If None, it will be set based on max_number.")
parser.add_argument("--min_number", type=float, default=1e-14, help="Minimum absolute number allowed to generate. All numbers are guaranteed to be within [min_number, max_number). Default is 1/1000.")
parser.add_argument("--max_number", type=int, default=1e15, help="Maximum absolute number allowed to generate. All numbers are guaranteed to be within [min_number, max_number). Default is 1000.")
parser.add_argument("--significant_bits", choices=[53, 24, 11], type=int, default=53, help="Number of significant bits (base 2) to use for floating point numbers. Default is 53 (double precision).")
parser.add_argument("--significant_digits_distribution", type=SignificantDigitsDistribution, default=SignificantDigitsDistribution.DECIMAL_UNIFORM, help="Distribution of significant digits for the numbers. 'uniform' means all digits are equally likely, 'triangular' means more digits are more likely, 'full' means all digits are used.")

parser.add_argument("--initial_num_samples", type=int, default=20, help="Initial number of samples to generate for estimating yield. Default is 20.")

parser.add_argument("--num_workers", type=int, default=math.ceil(mp.cpu_count()/2), help="Number of workers to use for multiprocessing. Default is half the number of CPU cores.")
parser.add_argument("--suffix", type=str, default="", help="Suffix to add to the save directory name. Useful for distinguishing between different runs.")
parser.add_argument("--phases", nargs="+", default=["train", "val", "test"], help="Phases to generate. Default is all three phases.")
parser.add_argument("--verbose", action="store_true", help="Generate verbose columns and images.")

args = parser.parse_args()
if  args.tasks[0]== "all":
    args.tasks = tasks
tasks = args.tasks
if args.min_exponent is None:
    args.min_exponent = int(math.floor(math.log(args.min_number, args.base)))
if args.max_exponent is None:
    args.max_exponent = int(math.ceil(math.log(args.max_number, args.base)))
args.significant_digits = math.floor(args.significant_bits * math.log(2, args.base))

os.makedirs(args.save_dir, exist_ok=True)


generation_settings = Generation_settings(args)

def iterative_generation(task: Task, num_samples_target: int, split: str, initial_num_samples: int = 20) -> pl.DataFrame:
    """
    Iterative dataset generation strategy:
    1. Generate small batch (N=20) to estimate yield per N
    2. Generate required samples estimated based on yield ratio
    3. Update yield ratio and repeat until target is reached or max iterations hit.
    """
    print(f"Generating dataset for {task} with target {num_samples_target} samples...")
    # Initialize with empty dataset
    df_combined = pl.DataFrame()
    
    iteration = 1
    max_iterations = 10  # Safety limit
    yield_ratio = 1  # Initial guess for yield ratio
    yield_momentum = 0  # Momentum factor for updating yield ratio
    N_iteration = initial_num_samples  # Start with small batch size
    
    while len(df_combined) < num_samples_target and iteration <= max_iterations:
        print(f"\nIteration {iteration}: Generating data with N={N_iteration}")
        
        batch_results = generate_dataset(
            task,
            split=split,
            N=N_iteration,
            gs=generation_settings,
            pool=MockPool() if args.num_workers <= 1 else mp.Pool(args.num_workers))
        num_interations: int = next(batch_results)
        
        # Process results and add to dataset
        batch_dataset = []
        for result in tqdm(batch_results, desc=f"Processing iteration {iteration}", total=num_interations):
            batch_dataset.extend(result)
        
        # Convert batch to DataFrame and remove duplicates within the batch
        df_batch = pl.DataFrame(batch_dataset)
        filter_candidates = ["num1", "num2", "list1", "ref"]
        filter_params = [col for col in filter_candidates if col in df_batch.columns]
        df_batch = df_batch.unique(subset=filter_params)
        
        # Get the number of unique samples in this batch
        unique_samples_batch = len(df_batch)
        
        # Store the count before merging
        total_samples_old = len(df_combined)
        
        # Combine with existing data and remove duplicates
        df_combined = pl.concat([df_combined, df_batch])
        filter_params = [col for col in filter_candidates if col in df_batch.columns]
        df_combined = df_combined.unique(subset=filter_params)
        
        # Calculate how many new unique samples were actually added
        new_samples_added = len(df_combined) - total_samples_old

        effective_yield = new_samples_added / N_iteration
        
        # Update yield ratio with a weighted average to adapt over time
        if iteration == 1:
            # First iteration, just use the actual yield
            yield_ratio = effective_yield
        else:
            # Update with momentum
            yield_ratio = yield_momentum * yield_ratio + (1-yield_momentum) * effective_yield
        if iteration>1:
            yield_ratio *= 1
        
        print(f"Batch produced {unique_samples_batch} unique samples ({effective_yield:.4f} per N)")
        print(f"Added {new_samples_added} new unique samples to dataset")
        print(f"Updated yield ratio: {yield_ratio:.4f}")
        remaining_samples_needed = max(0,num_samples_target - len(df_combined))
        print(f"Total dataset size: {len(df_combined)} / {num_samples_target} samples. Still need {remaining_samples_needed} samples")
        
        # Break if we're not making progress
        if new_samples_added == 0:
            print("Warning: No new samples added in last batch. Breaking loop.")
            break
            
        # Calculate N for next iteration based on yield ratio
        if yield_ratio > 0:
            N_iteration = math.ceil(remaining_samples_needed / yield_ratio) 
        else:
            N_iteration = remaining_samples_needed
            
        # Increase iteration counter
        iteration += 1
    
    # assert len(df_combined) <= num_samples_target * 1.4, f"Final dataset has too many samples: {len(df_combined)} > {num_samples_target * 1.1}"
    assert len(df_combined) >= num_samples_target, f"Could only generate {len(df_combined)} < {num_samples_target} samples"
    return df_combined.sample(n=min(num_samples_target, len(df_combined)), seed=args.seed)
    
        
for task in tasks:
    splits: dict[Literal["train", "val","test"], int] = {"test": args.num_test_samples, "val": args.num_val_samples,"train": args.num_train_samples}
    splits = {k: v for k, v in splits.items() if k in args.phases}
    for split, num_samples in splits.items():
        dataset = []
        print(f"\nGenerating {task} dataset for {split} split (target: {num_samples} samples)")
        start_time = time()
        df = iterative_generation(task, num_samples, split, initial_num_samples=args.initial_num_samples)
        print(f"Final dataset size: {len(df)}")

        # Print 5 random samples
        print(df.sample(5))

        # Format num_samples: XM for millions, Xk for thousands
        num_samples_str = f"{num_samples//1000000}M" if num_samples >= 1000000 else f"{num_samples//1000}k" if num_samples >= 1000 else str(num_samples)

        save_path = Path(f"{args.save_dir}/{task}_{args.significant_digits_distribution}_{args.suffix+'_' if args.suffix else ''}{split}_{num_samples_str}.csv.gz")

        end_time = time()
        formated_duration = str(timedelta(seconds=end_time - start_time))
        print(f"Generation time: {formated_duration} (HH:MM:SS)")

        if args.verbose:
            params = get_strat_params(save_path.name.split(".")[0])
            os.makedirs(save_path.parent / "images", exist_ok=True)
            plot_data_distribution(df, x_col=params[0], y_col=params[1], save_path=save_path.parent / "images" / f"{save_path.name.split(".")[0]}_distribution.png", title=f"{task} {split} distribution")

        df = convert_to_eval_prompts(df, split, save_path, seed=args.seed, num_workers=args.num_workers)
        
        if "difficulty" in df.columns:
            if args.verbose:
                plot_difficulty_histogram(df, file_path=save_path, difficulty_column="difficulty")
            print(df["difficulty"].describe())
            if split == "test" and args.significant_digits_distribution == SignificantDigitsDistribution.BINARY_UNIFORM:
                df = df.sort("difficulty")
        if "difficulty_sd" in df.columns:
            if args.verbose:
                plot_difficulty_histogram(df, file_path=save_path, difficulty_column="difficulty_sd")
            print(df["difficulty_sd"].describe())
            if split == "test" and args.significant_digits_distribution == SignificantDigitsDistribution.DECIMAL_UNIFORM:
                df = df.sort("difficulty_sd")

        if not args.verbose:
            df = df[["prompt", "text_prompt", "answer", "difficulty", "difficulty_sd"]]
        
        save_df_phase(df, split, args.save_dir, save_path)
