#!/usr/bin/env python3
"""
Script to analyze token counts from test prediction CSV files.

This script processes multiple test_predictions_*.csv files from a source folder,
computes input and output token counts using a specified tokenizer, and outputs
a combined CSV file with token count statistics per dataset.
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import yaml
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def extract_tokenizer_name_from_args(source_folder: Path) -> str:
    """
    Extract the tokenizer name from the args.yml file in the source folder.
    
    Args:
        source_folder (Path): Path to the source folder containing args.yml.
        
    Returns:
        str: The tokenizer name (e.g., 'fe_gpt2').
        
    Raises:
        FileNotFoundError: If args.yml is not found.
        KeyError: If tokenizer_dir is not found in args.yml.
        ValueError: If tokenizer path format is unexpected.
    """
    args_file = source_folder / "args.yml"
    
    if not args_file.exists():
        raise FileNotFoundError(f"args.yml not found in {source_folder}")
    
    with open(args_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'tokenizer_dir' not in config:
        raise KeyError("tokenizer_dir not found in args.yml")
    
    tokenizer_path = Path(config['tokenizer_dir'])
    
    # Extract tokenizer name from path like /path/to/tokenizers/num_text/fe_gpt2
    if tokenizer_path.parent.name == 'num_text' and tokenizer_path.parent.parent.name == 'tokenizers':
        return tokenizer_path.name
    else:
        raise ValueError(f"Unexpected tokenizer path format: {tokenizer_path}")


def load_tokenizer(tokenizer_dir: Path) -> PreTrainedTokenizerFast:
    """
    Load the tokenizer from the specified directory.
    
    Args:
        tokenizer_dir (Path): Path to the tokenizer directory.
        
    Returns:
        PreTrainedTokenizerFast: The loaded tokenizer.
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    tokenizer.padding_side = "left"
    return tokenizer


def count_tokens(text: str, tokenizer: PreTrainedTokenizerFast, is_prompt: bool = False) -> int:
    """
    Count the number of tokens in a given text.
    
    Args:
        text (str): The text to tokenize.
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use.
        is_prompt (bool): If True, adds 2 extra tokens to account for special tokens not visible in the raw text.
        
    Returns:
        int: Number of tokens in the text.
    """
    if pd.isna(text) or text == "":
        return 2 if is_prompt else 0
    
    tokens = tokenizer(text, padding="do_not_pad", return_tensors=None)
    token_count = len(tokens["input_ids"]) # pyright: ignore[reportArgumentType]
    
    # Add 2 extra tokens for prompts (e.g., BOS/EOS or other special tokens)
    if is_prompt:
        token_count += 2
    
    return token_count


def process_dataset(csv_path: Path, tokenizer: PreTrainedTokenizerFast) -> Tuple[str, int, int, int]:
    """
    Process a single dataset CSV file and compute token counts.
    
    Args:
        csv_path (Path): Path to the CSV file.
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use.
        
    Returns:
        Tuple[str, int, int, int]: Dataset name, total input tokens, total output tokens, number of samples.
    """
    # Extract dataset name from filename
    filename = csv_path.name
    if filename.startswith("test_predictions_") and filename.endswith(".csv"):
        dataset_name = filename[17:-4]  # Remove "test_predictions_" and ".csv"
    else:
        dataset_name = filename
    
    logging.info(f"Processing dataset: {dataset_name}")
    
    # Load only the required columns
    try:
        df = pd.read_csv(csv_path, usecols=['prompt', 'token prediction'], dtype=str)
    except ValueError as e:
        logging.error(f"Required columns not found in {csv_path}: {e}")
        return dataset_name, 0, 0, 0
    except Exception as e:
        logging.error(f"Error reading {csv_path}: {e}")
        return dataset_name, 0, 0, 0
    
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Process each row with progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Tokenizing {dataset_name}"):
        prompt = row['prompt']
        token_prediction = row['token prediction']
        
        # Count input tokens (prompt) - add 2 extra tokens for special tokens
        input_tokens = count_tokens(prompt, tokenizer, is_prompt=True)
        total_input_tokens += input_tokens
        
        # Count output tokens (token prediction)
        output_tokens = count_tokens(token_prediction, tokenizer, is_prompt=False)
        total_output_tokens += output_tokens
    
    num_samples = len(df)
    mean_tokens_per_sample = (total_input_tokens + total_output_tokens) / num_samples if num_samples > 0 else 0
    
    logging.info(f"Dataset {dataset_name}: {total_input_tokens} input tokens, {total_output_tokens} output tokens, {num_samples} samples, {mean_tokens_per_sample:.1f} tokens/sample")
    return dataset_name, total_input_tokens, total_output_tokens, num_samples


def append_to_output_csv(output_csv_path: Path, dataset_name: str, input_tokens: int, output_tokens: int, num_samples: int) -> None:
    """
    Append results to the output CSV file.
    
    Args:
        output_csv_path (Path): Path to the output CSV file.
        dataset_name (str): Name of the dataset.
        input_tokens (int): Total input tokens.
        output_tokens (int): Total output tokens.
        num_samples (int): Number of samples in the dataset.
    """
    file_exists = output_csv_path.exists()
    mean_input_tokens_per_sample = input_tokens / num_samples if num_samples > 0 else 0
    mean_output_tokens_per_sample = output_tokens / num_samples if num_samples > 0 else 0
    mean_total_tokens_per_sample = (input_tokens + output_tokens) / num_samples if num_samples > 0 else 0
    
    with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'dataset_name', 'input_tokens', 'output_tokens', 'num_samples',
            'mean_input_tokens_per_sample', 'mean_output_tokens_per_sample', 'mean_total_tokens_per_sample'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        # Write data row
        writer.writerow({
            'dataset_name': dataset_name,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'num_samples': num_samples,
            'mean_input_tokens_per_sample': round(mean_input_tokens_per_sample, 1),
            'mean_output_tokens_per_sample': round(mean_output_tokens_per_sample, 1),
            'mean_total_tokens_per_sample': round(mean_total_tokens_per_sample, 1)
        })


def find_test_prediction_files(source_folder: Path) -> List[Path]:
    """
    Find all test_predictions_*.csv files in the source folder.
    
    Args:
        source_folder (Path): Path to the source folder.
        
    Returns:
        List[Path]: List of test prediction CSV file paths.
    """
    pattern = "test_predictions_*.csv"
    csv_files = list(source_folder.glob(pattern))
    
    if not csv_files:
        logging.warning(f"No files matching pattern '{pattern}' found in {source_folder}")
    else:
        logging.info(f"Found {len(csv_files)} test prediction files")
    
    return csv_files


def main():
    """Main function to process all datasets and generate token count analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze token counts from test prediction CSV files"
    )
    parser.add_argument(
        "source_folder",
        type=Path,
        help="Path to the source folder containing test_predictions_*.csv files and args.yml"
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=Path,
        default=None,
        help="Path to the tokenizer directory (if not provided, will be auto-detected from args.yml)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV file path (default: token_counts_analysis.csv in source folder)"
    )
    
    args = parser.parse_args()
    
    # Set default output path to source folder if not specified
    if args.output is None:
        args.output = args.source_folder / "token_counts_analysis.csv"
    
    setup_logging()
    
    # Validate input paths
    if not args.source_folder.exists():
        logging.error(f"Source folder does not exist: {args.source_folder}")
        return 1
    
    # Determine tokenizer directory
    if args.tokenizer_dir is None:
        # Auto-detect tokenizer from args.yml
        try:
            tokenizer_name = extract_tokenizer_name_from_args(args.source_folder)
            # Construct path to local tokenizer directory
            script_dir = Path(__file__).parent
            args.tokenizer_dir = script_dir / "tokenizers" / "num_text" / tokenizer_name
            logging.info(f"Auto-detected tokenizer: {tokenizer_name}")
            logging.info(f"Using tokenizer directory: {args.tokenizer_dir}")
        except Exception as e:
            logging.error(f"Failed to auto-detect tokenizer from args.yml: {e}")
            logging.error("Please provide --tokenizer_dir argument manually")
            return 1
    
    if not args.tokenizer_dir.exists():
        logging.error(f"Tokenizer directory does not exist: {args.tokenizer_dir}")
        return 1
    
    # Load tokenizer
    logging.info(f"Loading tokenizer from {args.tokenizer_dir}")
    try:
        tokenizer = load_tokenizer(args.tokenizer_dir)
        logging.info(f"Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        return 1
    
    # Find test prediction files
    csv_files = find_test_prediction_files(args.source_folder)
    if not csv_files:
        logging.error("No test prediction files found")
        return 1
    
    # Remove existing output file to start fresh
    if args.output.exists():
        args.output.unlink()
        logging.info(f"Removed existing output file: {args.output}")
    
    # Process each dataset
    processed_count = 0
    for csv_file in csv_files:
        try:
            dataset_name, input_tokens, output_tokens, num_samples = process_dataset(csv_file, tokenizer)
            append_to_output_csv(args.output, dataset_name, input_tokens, output_tokens, num_samples)
            processed_count += 1
            logging.info(f"Successfully processed and saved results for {dataset_name}")
        except Exception as e:
            logging.error(f"Failed to process {csv_file}: {e}")
            continue
    
    
    # Display summary
    if args.output.exists():
        try:
            summary_df = pd.read_csv(args.output)
            logging.info("\nSummary:")
            logging.info(f"Total datasets: {len(summary_df)}")
            logging.info(f"Total input tokens: {summary_df['input_tokens'].sum():,}")
            logging.info(f"Total output tokens: {summary_df['output_tokens'].sum():,}")
            logging.info(f"Total samples: {summary_df['num_samples'].sum():,}")
            logging.info(f"Average input tokens per dataset: {summary_df['input_tokens'].mean():.0f}")
            logging.info(f"Average output tokens per dataset: {summary_df['output_tokens'].mean():.0f}")
            logging.info(f"Average input tokens per sample across all datasets: {summary_df['mean_input_tokens_per_sample'].mean():.2f}")
            logging.info(f"Average output tokens per sample across all datasets: {summary_df['mean_output_tokens_per_sample'].mean():.2f}")
            logging.info(f"Average total tokens per sample across all datasets: {summary_df['mean_total_tokens_per_sample'].mean():.2f}")
        except Exception as e:
            logging.warning(f"Could not display summary: {e}")
    
    logging.info(f"Processing complete. {processed_count}/{len(csv_files)} datasets processed successfully.")
    logging.info(f"Results saved to: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())