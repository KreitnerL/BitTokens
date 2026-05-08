import math
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns

sns.set_theme()

class BetterEnum(Enum):
    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.value == other.value
        elif isinstance(other, str):
            return self.value == other
        elif hasattr(other, 'value') and hasattr(other, '__class__'):
            # Handle cases where we have multiple enum instances with same value
            return (self.__class__.__name__ == other.__class__.__name__ and 
                    self.value == other.value)
        return NotImplemented
    
    def __str__(self) -> str:
        return str(self.value)
    
    def __hash__(self) -> int:
        return self.value.__hash__()
    def __repr__(self) -> str:
        return self.value.__repr__()
    
class SignificantDigitsDistribution(BetterEnum):
    BINARY_UNIFORM = "binary_uniform"
    BINARY_TRIANGULAR = "binary_triangular"
    BINARY_EXPONENTIAL = "binary_exponential"
    FULL = "full"
    DECIMAL_UNIFORM = "decimal_uniform"
    DECIMAL_TRIANGULAR = "decimal_triangular"
    DECIMAL_EXPONENTIAL = "decimal_exponential"

class Task(BetterEnum):
    ADDITION = "Addition"
    MULTIPLICATION = "Multiplication"
    DIVM = "DivM"
    DIVISION = "Division"
    EXPONENTIATION = "Exponentiation"
    MEAN = "Mean"
    STD = "Std"
    MIN_MAX = "MinMax"
    SORTING = "Sorting"
    INTERVAL = "Interval"
    TEXT = "Text"

@dataclass
class Generation_settings:
    """Settings for number generation."""
    base: int
    min_exponent: int
    max_exponent: int
    max_number: float
    min_number: float
    max_number_shifted: int
    min_number_shifted: int
    significant_digits: int
    number_of_total_digits: int
    min_number_exponent: int
    max_number_exponent: int
    significant_bits: Literal[53, 24, 11]
    significant_digits_distribution: SignificantDigitsDistribution

    def __init__(self, args):
        self.base = args.base
        self.max_number = args.max_number
        self.min_number = args.min_number
        self.min_number_exponent = int(math.floor(math.log(self.min_number, self.base)))
        self.max_number_exponent = int(math.ceil(math.log(self.max_number, self.base)))
        self.min_exponent = args.min_exponent
        self.max_exponent = min(args.max_exponent, self.max_number_exponent)
        self.max_number_shifted = self.max_number * args.base ** -self.min_exponent
        self.min_number_shifted = math.ceil(args.min_number * args.base ** -self.min_exponent)
        self.significant_digits = args.significant_digits
        self.significant_bits = args.significant_bits
        self.number_of_total_digits = self.max_exponent - self.min_exponent
        # Ensure significant_digits_distribution is always an enum, not a string
        if isinstance(args.significant_digits_distribution, str):
            # Find the matching enum instance by value to ensure we get the canonical instance
            for enum_value in SignificantDigitsDistribution:
                if enum_value.value == args.significant_digits_distribution:
                    self.significant_digits_distribution = enum_value
                    break
            else:
                raise ValueError(f"Unknown significant digits distribution: {args.significant_digits_distribution}")
        else:
            self.significant_digits_distribution = args.significant_digits_distribution
        print(f"Generation settings initialized: {self}")

    def patch(self, **args) -> 'Generation_settings':
        """Patch settings with new values."""
        from types import SimpleNamespace
        combined_dict = {**vars(self), **args}
        return Generation_settings(SimpleNamespace(**combined_dict))
    
class MockPool:
    _processes=1
    class MockResult:
        def __init__(self, value):
            self.value = value
        def get(self):
            return self.value
        def read(self):
            return True
    def apply_async(self, func, args):
        return self.MockResult(func(*args))
    def imap_unordered(self, func, iterable, chunksize=None):
        for args in iterable:
            yield func(args)
    def imap(self, func, iterable, chunksize=None):
        for args in iterable:
            yield func(args)
    def close(self):
        pass
    def join(self):
        pass

def get_strat_params(filename: str) -> list[str]:
    """
    Extracts the parameters from the filename.

    Args:
        filename (str): The name of the file.

    Returns:
        list[str]: List of parameters extracted from the filename.
    """
    if "TNC_FreqUniform" in filename:
        return ["diff_exp", "num1_exp", "num2_exp"]
    elif "TNC_DecimalShift" in filename:
        return ["num1_integers", "num2_integers", "num1_decimals", "num2_decimals"]
    elif "TNC_Random" in filename:
        return ["num1_integers", "num1_decimals", "num2_integers", "num2_decimals"]
    elif "TNC_DigitFlip" in filename:
        return ["num_integers", "num_decimals", "num_digit_diff"]
    elif "TNC_Mixed" in filename:
        return ["num1_exp", "diff_exp"]
    elif "Char_Repeat" in filename:
        return ["num_integers","num_decimals"]
    elif "Repeat" in filename:
        return ["num1_integers","num1_decimals"]
    elif "MinMax" in filename:
        return ["exp", "spread","operator", "list_len"] 
    elif "Interval" in filename:
        return ["exp", "spread", "list_len"]
    elif "Mean" in filename:
        return ["exp","spread", "list_len"]
    elif "Std" in filename:
        return ["exp","spread", "list_len"]
    elif "Sorting" in filename:
        return ["exp","spread", "operator","list_len"]
    elif "Add" in filename:
        return ["num1_exp", "num2_exp"]
    elif "Mul" in filename:
        return ["num1_exp", "num2_exp"]
    elif "Div" in filename:
        return ["num1_exp", "num2_exp"]
    elif "Exponent" in filename:
        return ["num1_exp", "num2_exp"]
    else:
        raise ValueError(f"Unknown task: {filename}")

def plot_data_distribution(data: pl.DataFrame, x_col: str, y_col: str, save_path: Path, title="Data Distribution"):
    """
    Plots the distribution of data as a heatmap and saves it to the specified path.

    Args:
        data (pl.DataFrame): DataFrame containing the data to plot.
        x_col (str): Column name for the x-axis.
        y_col (str): Column name for the y-axis.
        title (str): Title of the plot.
        save_path (Path): Path where the plot will be saved.
    """
    plt.figure(figsize=(10, 6))
    # Group by x_col and y_col, count occurrences
    grouped = data.group_by([x_col, y_col]).len().rename({"len": "count"}).sort([y_col, x_col])
    # Get sorted unique x and y labels
    x_labels = sorted(data[x_col].unique().to_list())
    y_labels = sorted(data[y_col].unique().to_list())
    # Pivot to wide format for heatmap
    heatmap_data = grouped.pivot(
        values="count",
        index=y_col,
        on=x_col
    ).fill_null(0)
    # Reindex rows and columns to ensure correct order
    heatmap_data = heatmap_data.sort(y_col)
    # Ensure columns are in the correct order (skip y_col in columns)
    col_order = [y_col] + [str(x_label) for x_label in x_labels]
    heatmap_data = heatmap_data.select(col_order)
    # Convert to numpy array for seaborn, and get sorted axes
    heatmap_array = heatmap_data.select(pl.exclude(y_col)).to_numpy()
    x_labels = heatmap_data.columns[1:]  # skip y_col
    y_labels = heatmap_data[y_col].to_list()
    sns.heatmap(heatmap_array, cbar_kws={'label': 'Count'}, xticklabels=x_labels, yticklabels=y_labels)
    plt.title(title)
    plt.xlabel(x_col)
    plt.xticks(rotation=45)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution plot to {save_path}.")

def plot_accuracy_distribution(data: pd.DataFrame, x_col: str, y_col: str, acc_col: str, save_path: Path, title="Accuracy Distribution"):
    """
    Plots the mean accuracy as a heatmap for each (x_col, y_col) combination and saves it.

    Args:
        data (pd.DataFrame): DataFrame containing the data to plot.
        x_col (str): Column name for the x-axis.
        y_col (str): Column name for the y-axis.
        acc_col (str): Column name containing accuracy values.
        title (str): Title of the plot.
        save_path (Path): Path where the plot will be saved.
    """
    plt.figure(figsize=(10, 6))
    # Group by x_col and y_col, compute mean accuracy
    heatmap_data = data.groupby([y_col, x_col])[acc_col].mean().unstack()
    # Sort the index (y-axis) and columns (x-axis) using natsort
    try:
        heatmap_data = heatmap_data.loc[[str(k) for k in sorted(heatmap_data.index.astype(int))], [str(k) for k in sorted(heatmap_data.columns.astype(float).astype(int))]]
    except KeyError:
        heatmap_data = heatmap_data.loc[[str(k) for k in sorted(heatmap_data.index.astype(int))], [str(k) for k in sorted(heatmap_data.columns.astype(float))]]
    sns.heatmap(heatmap_data,cbar_kws={'label': f'Mean {acc_col}'},vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel(x_col)
    plt.xticks(rotation=45)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_difficulty_histogram(df: pd.DataFrame | pl.DataFrame, file_path: Path, difficulty_column="difficulty") -> None:
    """
    Plot a histogram of the difficulty column in the dataframe and save it as a PNG file.

    Args:
        df (DataFrame): DataFrame containing the difficulty column.
        file_path (Path): Path to save the histogram image.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[difficulty_column], bins=len(df[difficulty_column].unique()), kde=True)  # pyright: ignore[reportOperatorIssue, reportArgumentType]
    plt.title("Histogram of difficulty samples")
    plt.xlabel(difficulty_column)
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.xlim(df[difficulty_column].min(),df[difficulty_column].max())
    image_save_path = file_path.parent / "images" / (str(file_path.stem).removesuffix("".join(file_path.suffixes[:-1])) + f"_{difficulty_column}.png")
    plt.savefig(image_save_path, bbox_inches='tight')
    print(f"Saved histogram to {image_save_path}.")

def save_df_phase(df_phase: pl.DataFrame, phase: str, output_folder: str | Path, path: Path):
    file_name = path.name.split(".")[0]
    extension = "csv" if phase == "test" else "csv.gz"
    out_path = f"{output_folder}/{file_name}.{extension}"
    if phase == "test":
        df_phase.write_csv(out_path)
    else:
        # Write uncompressed first, then compress with external tool
        temp_path = f"{output_folder}/{file_name}.csv"
        out_path = f"{output_folder}/{file_name}.csv.gz"
        
        df_phase.write_csv(temp_path)
        # Use pigz (parallel gzip) if available, otherwise gzip
        try:
            subprocess.run(['pigz', '-1', '-f', temp_path], check=True)
            # pigz renames file.csv to file.csv.gz automatically
        except FileNotFoundError:
            subprocess.run(['gzip', '-1', '-f', temp_path], check=True)
    print(f"Saved {out_path}")
