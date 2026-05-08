import csv
import datetime
from itertools import cycle
from math import ceil
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from utils.enums import TrackedMetrics, TrainMetrics
from utils.util_funcs import to_flat_dict

sns.set_theme()

def _prepare_fig(N: int, base_width:float = 5, base_height: float = 5, width_ratio=2.0) -> tuple[Figure, list[Axes]]:
    """
    Prepare a figure and axes for plotting N subplots.
    N subplots arranged to match roughly a 16:9 aspect ratio with automatically calculated figure size.

    Args:
    ----------
    - N (int): Number of subplots to create
    - base_width (float): Base width of each subplot
    - base_height (float): Base height of each subplot
    - width_ratio (float): Desired aspect ratio (width/height) for the overall layout

    Returns:
    ----------
    - axes (list[Axes]): List of matplotlib Axes objects for each subplot
    """
    if N == 1:
        cols, rows = 1, 1
    else:
        # Find the best grid configuration by trying different options
        # Start with the initial estimate
        initial_cols = ceil(np.sqrt(N * width_ratio))
        
        # Try a few configurations around the initial estimate
        candidates = []
        for cols in range(max(1, initial_cols - 2), initial_cols + 3):
            rows = ceil(N / cols)
            total_subplots = rows * cols
            unused_subplots = total_subplots - N
            aspect_ratio = cols / rows
            
            # Score based on: minimizing unused subplots and matching desired aspect ratio
            unused_penalty = unused_subplots
            aspect_penalty = abs(aspect_ratio - width_ratio)
            score = unused_penalty + 2.1 * aspect_penalty  # Increased weight for aspect ratio
            
            candidates.append((score, cols, rows, unused_subplots))
        
        # Choose the configuration with the best score
        _, cols, rows, _ = min(candidates, key=lambda x: x[0])
    
    # Scale figure size according to grid dimensions
    # For reference, a figure with 1 row and 2 columns should be about (8,4)
    figsize = (base_width * cols, base_height * rows)
    # Create figure and axes with increased vertical spacing
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Adjust spacing between subplots to prevent title/label overlap
    if rows > 1:
        fig.subplots_adjust(hspace=0.4)  # Increase vertical spacing between subplots
    if cols > 1:
        fig.subplots_adjust(wspace=0.3)  # Maintain horizontal spacing
    if rows > 1 and cols > 1:
        fig.subplots_adjust(hspace=0.4, wspace=0.3)  # Both spacings for grid layouts
    
    # Flatten axes array for easier indexing
    if rows > 1 and cols > 1:
        axes = axes.flatten()
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    elif cols == 1:
        axes = axes.reshape(-1)
    return fig, axes

def plot_losses(train_metrics: TrainMetrics, save_path: str|Path, weights_: Optional[list[float]]=None, skip_k_steps: int =1):
    """
    Plot the losses of the model.

    Args:
    ----------
    - train_metrics (TrainMetrics): The metrics of the model
    - save_path (str): The path to save the plot
    - weights_ (list[float]): The weights of the losses
    """
    # Count legend items to determine if we need extra space for external legends
    ax1_legend_count = len(train_metrics.train_losses) - 1 + 1  # -1 for skipped first item, +1 for "train loss"
    ax2_legend_count = len(train_metrics.val_gen_losses[0]) + 2  # val_gen_losses columns + best value line + best point
    
    # Adjust figure width if legends will be external
    base_width = 12
    extra_width = 0
    if ax1_legend_count > 4:
        extra_width += 3  # Extra space for ax1 external legend
    if ax2_legend_count > 4:
        extra_width += 3  # Extra space for ax2 external legend
    
    loss_plot_fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2, figsize=(base_width + extra_width, 6))
    ax1: Axes
    ax2: Axes
    
    # Adjust spacing between subplots if external legends are used
    wspace_needed = 0.2  # Default spacing
    if ax1_legend_count > 4 or ax2_legend_count > 4:
        wspace_needed = 0.8  # Increase spacing when legends are external
    loss_plot_fig.subplots_adjust(wspace=wspace_needed)
    
    if weights_ is None:
        weights: list[float] = [1] * max(len(train_metrics.train_losses)+1, len(train_metrics.val_gen_losses[0])+1)
    else:
        weights = weights_

    steps_full =np.linspace(0, len(train_metrics.train_losses["train_total"]), len(train_metrics.val_gen_accs)+1)
    x_full = np.linspace(0, train_metrics.num_tokens_per_step[len(train_metrics.train_token_accs)], len(train_metrics.val_gen_accs)+1)
    x = x_full[skip_k_steps:]
    # Average train_losses for each step range in x
    for i, (k, v) in enumerate(train_metrics.train_losses.items()):
        if i == 0:
            continue
        v_mean = [np.nanmean(v[int(steps_full[j-1]):int(steps_full[j])]) if len(v[int(steps_full[j-1]):int(steps_full[j])])>0 and not np.isnan(v[int(steps_full[j-1]):int(steps_full[j])]).all() else np.nan for j in range(1, len(steps_full))]
        y = [vm*weights[i] for vm in v_mean[skip_k_steps:]]
        ax1.plot(x[1:], y, label=f"{k.removesuffix('_train_30M')}={round(v[-1],3)}") # pyright: ignore[reportArgumentType]
    v = list(train_metrics.train_losses.values())[0]
    v_mean = [np.nanmean(v[int(steps_full[j-1]):int(steps_full[j])]) if int(steps_full[j-1]) < int(steps_full[j]) and int(steps_full[j]) <= len(v) else np.nan for j in range(1, len(steps_full))]
    ax1.plot(x[1:], [vm*weights[0] for vm in v_mean[skip_k_steps:]], label="train loss", linestyle="--")  # pyright: ignore[reportArgumentType]
    ax1.set_title("Train loss")
    ax1.set_xlabel("tokens")
    ax1.set_ylabel("loss")
    
    # Position ax1 legend
    if ax1_legend_count > 4:
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax1.legend(loc="upper right")
    
    val_gen_losses_df = pd.DataFrame(train_metrics.val_gen_losses[skip_k_steps:], index=x[1:])
    # Remove suffix from column names
    val_gen_losses_df.columns = [col.replace("_val_10k.csv.gz", "") for col in val_gen_losses_df.columns]
    val_gen_losses_df = val_gen_losses_df * weights[1:1+val_gen_losses_df.shape[-1]]
    val_gen_losses_df.plot(ax=ax2, label="val gen loss")
    ax2.set_title("Validation loss")
    ax2.set_xlabel("tokens")
    ax2.set_ylabel("loss")
    val_gen_losses_array = [list(d.values())[0]*weights[1] for d in train_metrics.val_gen_losses[skip_k_steps:]]
    best_value = min(val_gen_losses_array)
    best_x = x[1:][val_gen_losses_array.index(best_value)]
    best_line_label = f"Min val loss={round(best_value, 3)}"
    ax2.axhline(y=best_value, color='black', linestyle='dashed', label=best_line_label)
    ax2.plot(best_x, best_value, marker="*", markersize=10, color="red", label=f"Best step={int(best_x)}")
    
    # Position ax2 legend
    if ax2_legend_count > 4:
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax2.legend(loc="upper right")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(loss_plot_fig)

def plot_difficulty_acc(
        eval_difficulty_acc_agg: dict[str, np.ndarray],
        maximum_difficulty: dict[str, list[int]],
        tokens: int,
        save_path: str|Path,
        min_difficulty: dict[str,int],
        eval_difficulty_ratios: dict[str, np.ndarray],
        advancement_thresholds: dict[str, np.ndarray],
    ):
    n_steps = list(eval_difficulty_acc_agg.values())[0].shape[1]
    # Create x-axis values in token space for proper scaling
    x_tokens = np.linspace(tokens // n_steps, tokens, n_steps, endpoint=True)


    num_tasks = len(eval_difficulty_acc_agg)
    
    # Arrange pairs in a grid layout inspired by _prepare_fig
    # Each pair has 2 subplots (acc + ratio) stacked vertically
    width_ratio = 3/2
    base_width, base_height = 12, 9  # Keep original width for each pair
    
    # Calculate grid dimensions for pairs
    cols = ceil(np.sqrt(num_tasks * width_ratio))
    rows = ceil(num_tasks / cols)
    
    # Grid has cols columns and (rows * 2) rows (2 subplots per pair)
    grid_cols = cols
    grid_rows = rows * 3
    
    figsize = (base_width * cols, base_height * rows)
    fig, axes = plt.subplots(
        nrows=grid_rows,
        ncols=grid_cols,
        figsize=figsize,
        # sharex=True,
        constrained_layout=True
    )
    
    # Ensure axes is always 2D for easier indexing
    if grid_rows == 1 and grid_cols == 1:
        axes_array = np.array([[axes]])
    elif grid_rows == 1:
        axes_array = np.array(axes).reshape(1, -1)
    elif grid_cols == 1:
        axes_array = np.array(axes).reshape(-1, 1)
    else:
        # axes is already 2D
        axes_array = np.array(axes)
    
    # Plot accuracy and ratio pairs for each task
    assert len(eval_difficulty_acc_agg) == len(maximum_difficulty) == len(min_difficulty) == len(eval_difficulty_ratios) == len(advancement_thresholds), (
        f"Mismatched lengths: {len(eval_difficulty_acc_agg)} accuracy, {len(maximum_difficulty)} max difficulty, {len(min_difficulty)} min difficulty, {len(eval_difficulty_ratios)} ratios, {len(advancement_thresholds)} thresholds"
    )
    for i, (k, diff_acc, max_diff, min_diff, advancement_threshold) in enumerate(zip(eval_difficulty_acc_agg.keys(), eval_difficulty_acc_agg.values(), maximum_difficulty.values(), min_difficulty.values(), advancement_thresholds.values())):
        # Calculate position in grid
        pair_col = i % cols
        pair_row_start = (i // cols) * 3  # Each triplet takes 3 rows
        acc_row = pair_row_start
        thresh_row = pair_row_start + 1
        ratio_row = pair_row_start + 2
        
        # Accuracy plot
        acc_ax: Axes = axes_array[acc_row, pair_col]
        im_acc = acc_ax.imshow(np.flip(diff_acc, axis=0), aspect='auto', interpolation='nearest', cmap='flare_r', vmin=0, vmax=1, 
                               extent=(x_tokens[0] if len(x_tokens)>1 else 0, x_tokens[-1], 0, diff_acc.shape[0]))
        acc_ax.set_title(f"{k} - Accuracy")
        acc_ax.set_ylabel("difficulty")
        acc_ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: str(int(val + min_diff))))
        
        # Plot max difficulty line using token values
        max_diff_x = np.linspace(x_tokens[0], x_tokens[-1], len(max_diff))
        acc_ax.plot(max_diff_x, max_diff, color='white', linestyle='dashed', label=f"max_difficulty={max_diff[-1]+ min_difficulty[k]}")
        acc_ax.legend(loc='lower left')
        
        # Advancement threshold plot
        thresh_ax: Axes = axes_array[thresh_row, pair_col]
        im_thresh = thresh_ax.imshow(np.flip(advancement_threshold, axis=0), aspect='auto', interpolation='nearest', cmap='flare_r', vmin=0, vmax=1,
                                  extent=(x_tokens[0] if len(x_tokens)>1 else 0, x_tokens[-1], 0, advancement_threshold.shape[0]))
        thresh_ax.set_title(f"{k} - Advancement Threshold")
        thresh_ax.set_ylabel("difficulty")
        thresh_ax.set_xlabel("tokens")  # Set xlabel on threshold plots (bottom of each pair)
        thresh_ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: str(int(val + min_diff))))
        thresh_ax.plot(max_diff_x, max_diff, color='white', linestyle='dashed', label=f"max_difficulty={max_diff[-1]+ min_difficulty[k]}")
        thresh_ax.legend(loc='lower left')

        # Ratio plot
        ratio_ax: Axes = axes_array[ratio_row, pair_col]
        diff_ratio = eval_difficulty_ratios[k]
        im_ratio = ratio_ax.imshow(np.flip(diff_ratio, axis=0), aspect='auto', interpolation='nearest', cmap='viridis', vmin=0,
                                  extent=(x_tokens[0] if len(x_tokens)>1 else 0, x_tokens[-1], 0, diff_ratio.shape[0]))
        ratio_ax.set_title(f"{k} - Difficulty Ratio")
        ratio_ax.set_ylabel("difficulty")
        ratio_ax.set_xlabel("tokens")  # Set xlabel on ratio plots (bottom of each pair)
        ratio_ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: str(int(val + min_diff))))
        ratio_ax.plot(max_diff_x, max_diff, color='white', linestyle='dashed', label=f"max_difficulty={max_diff[-1]+ min_difficulty[k]}")
        ratio_ax.legend(loc='lower left')

        acc_cax = acc_ax.inset_axes((1.01, 0.05, 0.025, 0.9))  # Colorbar for accuracy
        fig.colorbar(im_acc, cax=acc_cax, label="LogSMAPE")

        thresh_cax = thresh_ax.inset_axes((1.01, 0.05, 0.025, 0.9))  # Colorbar for threshold
        fig.colorbar(im_thresh, cax=thresh_cax, label="advancement threshold")
        
        ratio_cax = ratio_ax.inset_axes((1.01, 0.05, 0.025, 0.9))  # Colorbar for ratio
        fig.colorbar(im_ratio, cax=ratio_cax, label="difficulty ratio")
    
    # Hide unused subplots if any
    for i in range(num_tasks, rows * cols):
        col = i % cols
        row_start = (i // cols) * 3
        if row_start < grid_rows:
            axes_array[row_start, col].set_visible(False)
        if row_start + 1 < grid_rows:
            axes_array[row_start + 1, col].set_visible(False)
        if row_start + 2 < grid_rows:
            axes_array[row_start + 2, col].set_visible(False)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_freq_loss_and_weights(
        train_metrics: dict,
        tokens: int,
        save_path: str|Path,
    ):
    """
    Creates a 2d plot with subplots for available data:
    1. Loss per frequency over time (tokens) - if num_loss_per_freq exists.
    2. Frequency loss weights over time (tokens) - if train_metrics has content.
    """
    num_loss_per_freq = train_metrics.pop("num_loss_per_freq", None)
    train_metrics.pop("freq_loss_weights", None)

    # Determine which plots to create
    has_freq_loss = num_loss_per_freq is not None and num_loss_per_freq.size > 0
    has_other_metrics = len(train_metrics) > 0
    
    # Count number of subplots needed
    n_subplots = sum([has_freq_loss, has_other_metrics])
    
    if n_subplots == 0:
        # Create an empty figure if no data to plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Frequency Loss and Weights")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return

    # Create figure with appropriate number of subplots
    fig, axes = plt.subplots(
        nrows=n_subplots, ncols=1, figsize=(12, 4 * n_subplots), 
        sharex=True, constrained_layout=True
    )
    
    # Ensure axes is always a list
    if n_subplots == 1:
        axes = [axes]
    
    current_ax = 0
    
    if has_freq_loss:
        n_steps = num_loss_per_freq.shape[0]
        # Create x-axis values in token space for proper scaling
        x_tokens = np.linspace(tokens // n_steps, tokens, n_steps, endpoint=True)

        # --- Loss per frequency plot ---
        ax = axes[current_ax]
        im1 = ax.imshow(np.flip(num_loss_per_freq.T, axis=0), aspect='auto', interpolation='nearest', cmap='flare', vmin=0, vmax=1,
                       extent=(x_tokens[0], x_tokens[-1], 0, num_loss_per_freq.shape[1]))
        ax.set_ylabel("low frequencies → high frequencies")
        ax.set_title("Loss per frequency")

        # Add a compact colorbar as an inset axis
        cax = ax.inset_axes([1.01, 0.05, 0.025, 0.9])  # [x0, y0, width, height]
        fig.colorbar(im1, cax=cax, label="Loss")
        current_ax += 1
    
    if has_other_metrics:
        # Create x-axis values in token space for line plots
        if len(list(train_metrics.values())[0]) > 0:
            n_steps = len(list(train_metrics.values())[0])
            x_tokens = np.linspace(tokens // n_steps, tokens, n_steps, endpoint=True)
        else:
            x_tokens = []

        # --- Other metrics plot ---
        ax = axes[current_ax]
        for i, (k, v) in enumerate(train_metrics.items()):
            if len(v) > 0 and len(x_tokens) > 0:
                ax.plot(x_tokens, v, label=k)
            else:
                ax.plot(v, label=k)  # Fallback to index-based plotting
        ax.set_title("Losses")
        ax.set_ylabel("loss")
        ax.legend()
        current_ax += 1

    # Set xlabel only on the last subplot
    if n_subplots > 0:
        axes[-1].set_xlabel("tokens")

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def plot_accuracy(train_metrics: TrainMetrics, save_path: str|Path, acc_funcs: list[Callable]):
    """
    Plot the accuracy of the model.

    Args:
    ----------
    - train_metrics (TrainMetrics): The metrics of the model
    - save_path (str): The path to save the plot
    - acc_funcs (list[Callable]): The accuracy functions
    """
    x_full = np.linspace(0, train_metrics.num_tokens_per_step[len(train_metrics.train_token_accs)], len(train_metrics.val_gen_accs)+1)
    x = x_full[1:]
    N = len(train_metrics.val_gen_accs[-1])
    fig, axes = _prepare_fig(N)
    val_accs_per_task: dict[str, list[float]] = {k.replace(".csv.gz", "").replace("_val_10k", ""): [step_accs[k] for step_accs in train_metrics.val_gen_accs] for k in train_metrics.val_gen_accs[0].keys()}
    additional_acc_per_task: dict[str, list[dict[str, float]]] = {k.replace(".csv.gz", "").replace("_val_10k", ""): [step_accs[k] for step_accs in train_metrics.additional_metrics] for k in train_metrics.additional_metrics[0].keys()}
    DEFAULT_COLORS = sns.color_palette()
    for i, (k, v) in enumerate(val_accs_per_task.items()):
        color_iter = cycle(DEFAULT_COLORS)
        color = next(color_iter)
        k2 = [_ for _ in additional_acc_per_task.keys() if k.startswith(_)]
        if len(k2)>0:
            axes[i].plot(x, v, label=f"{k.removeprefix(k2[0])}={round(v[-1],3)}", color=color)
            axes[i].set_title(k2[0])
            for metric_name in additional_acc_per_task[k2[0]][0].keys():
                additional_values = [step[metric_name] for step in additional_acc_per_task[k2[0]]]
                axes[i].plot(x, additional_values, label=f"{metric_name}={round(additional_values[-1],3)}", linestyle="--", color=next(color_iter))
        else:
            axes[i].set_title(k)
            axes[i].plot(x, v, label=f"{k}={round(v[-1],3)}", color=color)
        best_value = max(v)
        best_x = x[v.index(best_value)]
        best_line_label = f"max acc={round(best_value, 3)}"
        axes[i].axhline(y=best_value, color='black', linestyle='dashed', label=best_line_label)
        axes[i].plot(best_x, best_value, marker="*", markersize=10, color="red", label=f"Best step={int(best_x)}")
        axes[i].legend(fontsize='small')
        axes[i].set_xlabel("tokens")
        axes[i].set_ylabel(str(next(iter(acc_funcs[i:]), 'harmonic mean')))
        axes[i].set_ybound(-0.1, 1.1)

    for ax in axes[N:]:
        ax.set_visible(False)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def plot_dict(data: dict[int, dict[str, float]], save_path: Path, title: str, xlabel: str, ylabel: str, log=False):
    """
    Plot the accuracies of the model.

    Parameters:
    ----------
    - data: The data to plot. The keys are the steps and the values are dictionaries with the accuracies
    - save_path: The path to save the plot
    """
    accuracies_fig, accuracies_axes = plt.subplots(1,1, figsize=(6,5))
    accuracies_axes: Axes
    accuracies_axes.clear()
    accuracies_axes.set_title(title)
    accuracies_axes.set_xlabel(xlabel)
    accuracies_axes.set_ylabel(ylabel)
    if log:
        accuracies_axes.set_yscale("log")
    for label in list(data.values())[0].keys():
        steps: list[int] = data.keys()
        accuracies_axes.plot(steps, [d.get(label, np.nan) for d in data.values()], label=f"{label}={list(data.values())[-1].get(label, np.nan):.3f}", marker="." if len(data)<100 else None)
    accuracies_axes.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside the plot
    # accuracies_axes.grid(axis='y')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_lr_update(save_dir: Path, lr_updates: list[float], tokens: Optional[list[int]] = None, suffix: str = ""):
    """
    Plot the learning rate update.

    Parameters:
    ----------
    - lr_update: The learning rate update for each lr scheduler
    - epoch: The current epoch
    """
    # Generate loss input
    lr_fig, lr_fig_axes = plt.subplots(1,1, figsize=(6,5))
    lr_fig_axes: Axes
    lr_fig_axes.clear()
    lr_fig_axes.set_title("learning rate")
    lr_fig_axes.set_xlabel("steps")
    lr_fig_axes.plot(lr_updates)
    # lr_fig_axes.grid(axis='y')
    lr_fig_axes.set_ylim(bottom=None, top=None)
    if tokens is not None:
        ax2 = lr_fig_axes.twiny()
        ax2.plot(tokens, np.ones_like(tokens)+1) # Create a dummy plot
        ax2.set_xlabel("tokens")
    plt.savefig(save_dir / f'lr_update{suffix}.png', bbox_inches='tight')
    plt.close()

def plot_interval_duration(interval_durations: list[float], tokens: list[int], total_duration: float, save_path: Path, batch_size: int):
    """
    Plot the duration of each interval.

    Parameters:
    ----------
    - interval_durations: The duration of each interval
    - save_path: The path to save the plot
    """
    interval_duration_fig, interval_duration_axes = plt.subplots(1,1, figsize=(6,5))
    interval_duration_axes: Axes
    interval_duration_axes.clear()
    total_duration_str = str(datetime.timedelta(seconds=int(total_duration)))
    interval_duration_axes.set_title(f"Total duration={total_duration_str}")
    interval_duration_axes.set_xlabel("tokens")
    interval_duration_axes.plot(tokens, interval_durations, label=f"latest duration = {interval_durations[-1]:.3f}s")
    # interval_duration_axes.grid(axis='y')
    interval_duration_axes.set_ylim(bottom=0, top=None)
    interval_duration_axes.set_ylabel("duration (s/1M tokens)")
    interval_duration_axes.legend()
    plt.savefig(save_path / "durations.png", bbox_inches='tight')
    plt.close()

def save_or_add_to_csv(data: TrackedMetrics | dict, path: Path):
    """
    Saves the given dictionary of lists to an existing csv file, or creates a new one if no file exists yet.
    """
    data = to_flat_dict(data)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(path, "w", newline='')
        writer = csv.writer(csv_file)
        writer.writerow(data.keys())
    else:
        csv_file = open(path, "a", newline='')
        writer = csv.writer(csv_file)
    
    writer.writerow(list(data.values()))

    csv_file.flush()
    csv_file.close()
