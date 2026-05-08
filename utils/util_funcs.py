import logging
import re
from argparse import Namespace
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from sys import maxsize
from typing import Callable, Optional, cast
from warnings import deprecated

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tap import Tap
from torch.cuda.amp import GradScaler
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from dataloader.curriculum_manager import CurriculumManager
from utils.enums import TrackedMetrics, TrackedMetricsDataframe, TrainMetrics
from utils.train_argument_parser import TrainArgumentParser
from utils.warm_start_lr_scheduler import WarmStartLrScheduler

NUMBER_PARSE_REGEX = r"[-]?(?:(?:0(?!\.[0-9]))|(?:[0-9]*[.][0-9]+)|(?:[1-9][0-9]*))"
ABSOLUTE_NUMBER_PARSE_REGEX = r"(?:(?:0(?!\.[0-9]))|(?:[0-9]*[.][0-9]+)|(?:[1-9][0-9]*))"


# https://huggingface.co/blog/codeparrot
# We don’t want to apply weight decay to biases and LayerNorm weights so we use a helper function to exclude those.
def get_grouped_params(model: nn.Module, args: Namespace, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        elif any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [{"params": params_with_wd, "weight_decay": args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},]

def print_and_save_arguments(args: Tap, save_dir: Path):
    """
    The function prints all arguments and their values to the console and writes them to a file named 'args.yml' in the specified directory.
    Only arguments with non-None values are written to the file.
    Args:
        args (Tap): An object containing the arguments to be printed and saved.
        save_dir (Path): The directory where the arguments file will be saved.
    """
    arg_dict = args._log_all()
    sorted_arg_dict = {k: arg_dict[k] for k in sorted(arg_dict) if not callable(arg_dict[k])}
    logging.info("Arguments:")
    logging.info("-----------------")
    for arg, value in sorted_arg_dict.items():
        logging.info(f"{arg}: {value}")
    logging.info("-----------------")

    with open(save_dir / "args.yml", "w") as f:
        for arg, value in sorted_arg_dict.items():
            if value is not None:
                f.write(f"{arg}: {value}\n")

def _load_args_to_dict(save_dir: Path) -> dict:
    with open(save_dir / "args.yml", 'r') as file:
        args = yaml.safe_load(file)
    return args

def save_checkpoint(
        model: PreTrainedModel,
        optimizer: Optimizer,
        lr_schedulers: list[LRScheduler],
        step: int,
        save_dir: Path,
        prefix: str = "",
        curriculum_manager: Optional["CurriculumManager"] = None,
    ):
    """
    Save the current state of the model, optimizers, learning rate schedulers.

    Args:
        model (PreTrainedModel): The model to be saved.
        optimizers (list[Optimizer]): The optimizers whose states will be saved.
        lr_schedulers (list[LRScheduler]): The learning rate schedulers whose states will be saved.
        step (int): The current training step, used for naming the checkpoint.
        save_dir (Path): The directory where the checkpoint will be saved.
        prefix (str, optional): An optional prefix for the checkpoint name. Defaults to "".
    """
    checkpoint_dir = save_dir / f"{prefix or step}_checkpoint"
    model.save_pretrained(checkpoint_dir)
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    
    # Save scheduler states
    for i, lr_scheduler in enumerate(lr_schedulers):
        torch.save(lr_scheduler.state_dict(), checkpoint_dir / f"scheduler{i}.pt")

    if curriculum_manager is not None:
        torch.save(curriculum_manager.get_state(), checkpoint_dir / "curriculum_manager.pt")
    # logging.info(f"Saved {prefix or step} checkpoint at step {step} to {save_dir}")

def reconstruct_larger_array(known_indices: list[int], small_array: list[float], B: int, N: int) -> list:
    """
    Reconstructs a larger array from a smaller array by placing the elements of the smaller array
    at specific intervals and interpolating the values in between.

    Args:
        small_array (list[float]): The smaller array containing known values.
        B (int): The interval between known values in the larger array.
        N (int): The size of the larger array to be reconstructed.

    Returns:
        list: The reconstructed larger array with interpolated values.
    """
    large_array = np.zeros(N)
    # known_indices = [B * (i + 1) - 1 for i in range(len(small_array)-1)]
    large_array[known_indices] = small_array
    for i in range(1, len(known_indices)):
        large_array[known_indices[i-1]:known_indices[i]] = np.linspace(
            large_array[known_indices[i-1]], large_array[known_indices[i]], 
            known_indices[i] - known_indices[i-1] + 1)[:-1]
    large_array[:B] = small_array[0]
    return large_array.tolist()

def load_metrics(load_dir: Path, save_dir: Path) -> tuple[TrackedMetricsDataframe, int, dict[int,int]]:
    max_step = maxsize if (load_dir.name.startswith("latest") or load_dir.name.startswith("best"))  else int(load_dir.name.split("_")[0])
    # Load metrics file
    metrics = pd.read_csv(load_dir.parent / "metrics.csv")
    # Remove all rows with step greater than max_step
    metrics = metrics[metrics["step"] <= max_step]
    # save to args.save_dir
    metrics.to_csv(save_dir / "metrics.csv", index=False)
    step = metrics["step"].iloc[-1] + 1
    num_tokens_per_step = dict(zip(metrics["step"].tolist(), metrics["num_tokens"].tolist()))
    return metrics, step, num_tokens_per_step

def load_train_metrics_from_csv(metrics_csv_path: Path) -> TrainMetrics:
    """
    Load a TrainMetrics object from a metrics CSV file.
    
    This function reconstructs a TrainMetrics object from a CSV file that was
    previously saved using the save_or_add_to_csv function with TrackedMetrics data.
    It handles the flattened structure and reconstructs nested dictionaries.
    
    Args:
        metrics_csv_path (Path): Path to the metrics.csv file.
    
    Returns:
        TrainMetrics: Reconstructed TrainMetrics object populated with data from the CSV.
        
    Example:
        >>> from pathlib import Path
        >>> train_metrics = load_train_metrics_from_csv(Path("trained/run/metrics.csv"))
        >>> print(train_metrics.val_gen_accs[-1])  # Last evaluation accuracies
    """
    df = pd.read_csv(metrics_csv_path)
    train_metrics = TrainMetrics()
    
    # Load basic metrics that are always present
    train_metrics.num_tokens_per_step = dict(zip(df["step"].tolist(), df["num_tokens"].tolist()))
    train_metrics.lr_updates = df["lr"].tolist() if "lr" in df.columns else []
    train_metrics.train_perplexities = df["train_perplexity"].tolist() if "train_perplexity" in df.columns else []
    train_metrics.train_token_accs = df["train_token_accs"].tolist() if "train_token_accs" in df.columns else []
    
    # Reconstruct nested dictionaries for train_losses
    train_loss_cols = [col for col in df.columns if col.startswith("train_loss_")]
    for col in train_loss_cols:
        loss_name = col.replace("train_loss_", "")
        train_metrics.train_losses[loss_name] = df[col].tolist()
    
    # Reconstruct nested dictionaries for val_gen_accs
    val_gen_acc_cols = [col for col in df.columns if col.startswith("val_gen_acc_")]
    val_gen_accs_dict: dict[str, list[float]] = {}
    for col in val_gen_acc_cols:
        dataset_name = col.replace("val_gen_acc_", "")
        val_gen_accs_dict[dataset_name] = df[col].tolist()
    
    # Convert to list of dicts format
    if val_gen_accs_dict:
        num_rows = len(df)
        train_metrics.val_gen_accs = [
            {name: vals[i] for name, vals in val_gen_accs_dict.items()}
            for i in range(num_rows)
        ]
    
    # Reconstruct nested dictionaries for val_gen_losses
    val_gen_loss_cols = [col for col in df.columns if col.startswith("val_gen_loss_")]
    val_gen_losses_dict: dict[str, list[float]] = {}
    for col in val_gen_loss_cols:
        dataset_name = col.replace("val_gen_loss_", "")
        val_gen_losses_dict[dataset_name] = df[col].tolist()
    
    if val_gen_losses_dict:
        num_rows = len(df)
        train_metrics.val_gen_losses = [
            {name: vals[i] for name, vals in val_gen_losses_dict.items()}
            for i in range(num_rows)
        ]
    
    # Reconstruct nested dictionaries for val_gen_perplexities
    val_gen_perplexity_cols = [col for col in df.columns if col.startswith("val_gen_perplexity_")]
    val_gen_perplexities_dict: dict[str, list[float]] = {}
    for col in val_gen_perplexity_cols:
        dataset_name = col.replace("val_gen_perplexity_", "")
        val_gen_perplexities_dict[dataset_name] = df[col].tolist()
    
    if val_gen_perplexities_dict:
        num_rows = len(df)
        train_metrics.val_gen_perplexities = [
            {name: vals[i] for name, vals in val_gen_perplexities_dict.items()}
            for i in range(num_rows)
        ]
    
    # Reconstruct additional_metrics (nested dict of dicts)
    additional_metrics_cols = [col for col in df.columns if col.startswith("additional_metrics_")]
    if additional_metrics_cols:
        num_rows = len(df)
        train_metrics.additional_metrics = []
        for i in range(num_rows):
            row_metrics: dict[str, dict[str, float] | None] = {}
            for col in additional_metrics_cols:
                # Format: additional_metrics_<dataset>_<metric>
                # The metric can be compound like "exact_number_acc" or simple like "logSMAPE"
                parts = col.replace("additional_metrics_", "")
                
                # Find the dataset name by looking for known dataset patterns
                # Split and reconstruct to find the dataset/metric boundary
                # The dataset name ends with .csv.gz or .txt typically
                if ".csv.gz_" in parts:
                    dataset_name, metric_full = parts.split(".csv.gz_", 1)
                    dataset_name += ".csv.gz"
                elif ".txt_" in parts:
                    dataset_name, metric_full = parts.split(".txt_", 1)
                    dataset_name += ".txt"
                else:
                    # Fallback to original logic if pattern doesn't match
                    split_parts = parts.rsplit("_", 1)
                    if len(split_parts) == 2:
                        dataset_name, metric_full = split_parts
                    else:
                        continue
                
                val = df[col].iloc[i]
                if pd.notna(val):
                    if dataset_name not in row_metrics:
                        row_metrics[dataset_name] = {}
                    metric_dict = row_metrics[dataset_name]
                    if isinstance(metric_dict, dict):
                        metric_dict[metric_full] = float(val)
            train_metrics.additional_metrics.append(row_metrics)
    
    return train_metrics

def load_checkpoint(
        optimizer: Optimizer,
        checkpoint_dir: Path,
        device: torch.device,
        lr_schedulers: Optional[list[LRScheduler]] = None,
        curriculum_manager: Optional["CurriculumManager"] = None,
        step: int = -1
):
    optimizer_path = checkpoint_dir / "optimizer.pt"
    if optimizer_path.exists():
        logging.info(f"Loading optimizer state from {optimizer_path}")
        optimizer_state = torch.load(optimizer_path, map_location=device)
        if isinstance(optimizer_state, list) and len(optimizer_state) == 1:
            optimizer.load_state_dict(optimizer_state[0])
        elif not isinstance(optimizer_state, list):
            optimizer.load_state_dict(optimizer_state)
        else:
            logging.warning(f"Optimizer state format mismatch. Expected single optimizer state, got {type(optimizer_state)}")
    else:
        logging.warning(f"No optimizer state found at {optimizer_path}")

    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    if lr_schedulers is not None:
        for i, lr_scheduler in enumerate(lr_schedulers):
            lr_scheduler.load_state_dict(torch.load(checkpoint_dir / f"scheduler{i}.pt"))
            # lr_scheduler.last_epoch = step+1

    if curriculum_manager is not None and (checkpoint_dir / "curriculum_manager.pt").exists():
        curriculum_manager_state_dict: dict = torch.load(checkpoint_dir / "curriculum_manager.pt")
        curriculum_manager.load_state(curriculum_manager_state_dict)


@deprecated("This function is deprecated. Use `load_checkpoint` instead.")
def load_checkpoint_legacy(
        tokenizer: PreTrainedTokenizerFast,
        save_dir: Path,
        args: TrainArgumentParser,
        step: int,
        device: torch.device,
        get_model: Callable,
    ) -> tuple[PreTrainedModel, Optimizer, list[LRScheduler], GradScaler, pd.DataFrame, int]:
    """
    Load the latest checkpoint for a given model, optimizer, learning rate scheduler, and gradient scaler.

    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use.
        save_dir (Path): The directory where the checkpoint is saved.
        args (TrainArgumentParser): The training arguments.
        step (int): The current training step.
        device (torch.device): The device to load the model onto.
        get_model (Callable): A callable that returns the model with the loaded checkpoint.

    Returns:
        tuple: A tuple containing the following:
            - model (PreTrainedModel): The model with the loaded checkpoint.
            - optimizer (Optimizer): The optimizer with the loaded state.
            - lr_schedulers (list[LRScheduler]): The learning rate scheduler with the loaded state.
    """
    logging.info(f"Loading checkpoint from {save_dir}")
    model = get_model(
        tokenizer=tokenizer,
        args=args,
        pretrained_model_dir=save_dir,
        device=device
    )
    optimizer = AdamW(get_grouped_params(model, cast(Namespace, args)), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            WarmStartLrScheduler(optimizer, total_iters=args.num_warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_train_steps-args.num_warmup_steps),
        ],
        [args.num_warmup_steps]
    )
    lr_schedulers = [lr_scheduler]
    old_args = _load_args_to_dict(save_dir.parent)
    # Load the optimizer and scheduler state
    if args.continue_from:
        optimizer.load_state_dict(torch.load(save_dir / "optimizer.pt"))
        for i, lr_scheduler in enumerate(lr_schedulers):
            lr_scheduler.load_state_dict(torch.load(save_dir / f"scheduler{i}.pt"))
            lr_scheduler.last_epoch = step+1
        args.seed = int(old_args["seed"])
        args.seed = args.seed or old_args["seed"]

    # Set seed to the old seed
    # Verify that the arguments match
    for arg, value in old_args.items():
        if arg in ["save_dir", "reproducibility", "continue_from", "verbose", "tqdm", "from_pretrained", "save_checkpoint_steps", "seed", "compile"]:
            continue
        try:
            v = getattr(args, arg)
        except AttributeError:
            logging.warning(f"Argument '{arg}' not found in the current arguments")
            continue
        v_type = type(v)
        if isinstance(value, v_type):
            value = v_type(value)

        if getattr(args, arg) != value:
            if arg in ["num_epochs", "max_train_steps", "train_set_path", "val_set_path", "eval_steps", "max_eval_steps", "num_workers", "unique_samples", "num_warmup_steps", "lr", "weight_decay", "batch_size"]:
                if not args.from_pretrained:
                    logging.warning(f"Argument '{arg}' does not match: {getattr(args, arg)} != {value}")
                continue
            else:
                raise AssertionError(f"Argument '{arg}' does not match: {getattr(args, arg)} != {value}")

    logging.info(f"Loaded latest checkpoint at step {step} from {save_dir}")
    return model, optimizer, lr_schedulers

def parse_numbers_from_text(text: str, split_large_numbers=False) -> list[float]:
    """
    Parse numbers from a text.
    Args:
        text (str): The text to parse numbers from
    Returns:
        list[float]: The list of numbers
    """
    numbers = []
    
    for match in re.finditer(NUMBER_PARSE_REGEX, text):
        num_str = match.group().strip()
        # Process the number string to handle precision
        if split_large_numbers:
            original = Decimal(num_str)
            back_and_forth = Decimal(float(num_str))
            if original != back_and_forth:
                extracted_nums = split_large_number(num_str)
                numbers.extend(extracted_nums)
            else:
                numbers.append(float(num_str))
        else:
            numbers.append(float(num_str))
    return numbers

def split_large_number(num_str: str):
    """Split numbers with more than 15 significant digits into multiple numbers."""
    # Use Decimal for precise representation
    sign = '-' if num_str.startswith('-') else ''
    num_str = num_str.lstrip(' -+').rstrip().removesuffix(".")
    if "." in num_str:
        num_str = num_str.rstrip("0")
    num_str = num_str.removesuffix(".")

    numbers = []
    
    while num_str != "":
        if num_str.startswith("."):
            integer_part, decimal_part = "", num_str.removeprefix(".")
        else:
            integer_part, decimal_part = num_str.split(".") if "." in num_str else (num_str, "")
        if integer_part != "" and integer_part != "0":
            num_str = num_str.lstrip("0")
            chunk_length = 15 if "." not in num_str[:15] else 16
            if len(num_str.rstrip("0").removesuffix(".").rstrip("0")) <= 15:
                chunk_length = maxsize
            num_chunk = num_str[:chunk_length]
            num_remainder = num_str[chunk_length:]
            num_remainder_replacement = "0" * len(num_remainder.split(".")[0] if "." not in num_chunk else "")
            num = float(sign + num_chunk+num_remainder_replacement)
            assert not np.isinf(num), f"Number {num_str} is too large to be represented as a float"
            numbers.append(num)
            if num_remainder != "":
                logging.warning(f"Number {num_str} is too large to be represented as a float. Remainder: {num_remainder}")
                if num_chunk.startswith("."):
                    num_chunk_int, num_chunk_dec = "", num_chunk.removeprefix(".")
                else:
                    num_chunk_int, num_chunk_dec = num_chunk.split(".",1) if "." in num_chunk else (num_chunk, "")
                if num_chunk_dec != "":
                    num_chunk_dec = "0" * len(num_chunk_dec)
                    num_str = "." + num_chunk_dec + num_remainder
                else:
                    num_str = num_remainder
            else:
                break
        else:
            # Only decimal part
            # find beginning of significant digits
            start_index = next(re.finditer(r"[1-9]", num_str), None)
            if start_index is None:
                return [0]
            start_index = start_index.start()
            num_chunk = num_str[start_index:start_index + 15]
            num_remainder = num_str[start_index + 15:]
            numbers.append(float(sign + num_str[:start_index + 15]))
            num_chunk_replacement = "0" * len(num_chunk)
            if num_remainder != "":
                logging.warning(f"Number {num_str} is too large to be represented as a float. Remainder: {num_remainder}")
                num_str = num_str[:start_index] + num_chunk_replacement + num_remainder
            else:
                break
    return numbers

def replace_num_from_text(text: str, num_tokens: list[str], num_token_bins: list[float], num_prefix_token: str="", allow_negative=True, split_large_numbers=False) -> tuple[str, torch.FloatTensor]:
    """
    Replace numbers in a text with a token and return the numbers.
    Args:
        text (str): The text to replace numbers in
        num_token (str): The token to replace numbers with
    Returns:
        tuple[str, list[float]]: The text with numbers replaced and the list of numbers
    
    TODO Support INT and FLOAT types
    """
    numbers = []
    parts = []
    last_end = 0
    regex = NUMBER_PARSE_REGEX if allow_negative else ABSOLUTE_NUMBER_PARSE_REGEX

    for match in re.finditer(regex, text):
        num_str = match.group().lower().strip()
        if split_large_numbers:
            original = Decimal(num_str)
            back_and_forth = Decimal(float(num_str))
            if original != back_and_forth:
                extracted_nums = split_large_number(num_str)
        extracted_nums = [float(num_str)]
        numbers.extend(extracted_nums)
        parts.append(text[last_end:match.start() if num_str[0] != ' ' else match.start() + 1])
        for _ in extracted_nums[1:]:
            parts.append("<|overflow|>")
        last_end = match.end()

    parts.append(text[last_end:])
    num_token_indices = np.digitize(numbers, num_token_bins)
    new_text = "".join(f"{part}{num_prefix_token}{num_tokens[num_token_indices[i]-1] if i<len(num_token_indices) else ''}" for i, part in enumerate(parts))
    return new_text, torch.tensor(numbers, dtype=torch.float64) if numbers else torch.tensor([])

def to_flat_dict(data: TrackedMetrics | dict) -> dict:
    if not isinstance(data, dict):
        data_dict = asdict(data)
    else:
        data_dict = data
    new_data: dict = dict()
    for key, value in data_dict.items():
        if isinstance(value, dict):
            for k, v in value.items():
                new_data[f"{key}_{k}"] = v
        else:
            new_data[key] = value
    return new_data

def get_num_token_mask(tokens: torch.LongTensor, num_tokens: torch.LongTensor) -> torch.BoolTensor:
    """
    Get the mask for the number tokens in the input tokens.
    Args:
        tokens (torch.LongTensor): The input tokens
        num_tokens (torch.LongTensor): The number tokens
    Returns:
        mask (torch.BoolTensor): The mask for the number tokens
    """
    # Create a view of t as a 1D tensor without copying data
    t_flat = tokens.contiguous().view(-1)
    # This checks for each element in t_flat if it is in v
    mask_flat = torch.isin(t_flat, num_tokens)
    # Reshape the mask back to t's original shape
    mask = mask_flat.view(tokens.shape)
    return mask


def get_num_token_mask_numpy(tokens: np.ndarray, num_tokens: np.ndarray | list) -> np.ndarray:
    """
    Get the mask for the number tokens in the input tokens.
    Args:
        tokens (np.ndarray): The input token indices
        num_tokens (np.ndarray): The number tokens indices
    Returns:
        mask (np.ndarray): The mask for the number tokens
    """
    # Create a view of t as a 1D tensor without copying data
    t_flat = tokens.reshape(-1)
    # This checks for each element in t_flat if it is in v
    mask_flat = np.isin(t_flat, num_tokens)
    # Reshape the mask back to t's original shape
    mask = mask_flat.reshape(tokens.shape)
    return mask

def replace_in_order(text: str, substrings: list[str], replacements: list[str]) -> str:
    # Iterator for replacements
    replacement_iter = iter(replacements)
    occurrences=list()
    # Process each substring
    for sub in substrings:
        # Find all occurrences of the substring in the text
        occurrences.extend([(m.start(), m.end()) for m in re.finditer(re.escape(sub), text)])
    # Sort occurrences by start index
    occurrences.sort(key=lambda x: x[0])
    offset = 0
        
    # Replace each occurrence of the substring in order
    for s,e in occurrences:
        # Find the next replacement from the list
        try:
            replacement = next(replacement_iter)
            text = text[:s+offset] + replacement + text[e+offset:]
            offset += len(replacement) - (e - s)
        except StopIteration:
            # If we run out of replacements, stop replacing
            break
    
    return text

def check_intervals_all_true(bool_tensor: torch.BoolTensor, indices: torch.LongTensor) -> torch.BoolTensor:
    """
    Check if all intervals between indices in a boolean tensor are True.
    Args:
        bool_tensor (torch.BoolTensor): The boolean tensor
        indices (torch.LongTensor): The indices to check
    Returns:
        ret (torch.BoolTensor): A tensor of booleans indicating for each interval, whether they are all True
    """
    intervals = [bool_tensor[i:j] for i, j in zip([0] + indices[:-1].tolist(), indices)]
    return torch.tensor([torch.all(interval) for interval in intervals])
