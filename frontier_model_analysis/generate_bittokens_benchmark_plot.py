# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle

# File paths (absolute as requested)
path_multi = "./frontier_model_analysis/results/analysis/multitask.csv"
path_solo = "./frontier_model_analysis/results/analysis/solotask.csv"

# Read CSVs
df_multi = pd.read_csv(path_multi, dtype=str)
df_solo = pd.read_csv(path_solo, dtype=str)

# Clean column names and values
df_multi.columns = df_multi.columns.str.strip()
df_solo.columns = df_solo.columns.str.strip()
for df in (df_multi, df_solo):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

# Identify model columns from the header order
fixed_cols = ["Task", "Metric"]
model_cols_multi = [c for c in df_multi.columns if c not in fixed_cols]
model_cols_solo = [c for c in df_solo.columns if c not in fixed_cols]

# Preserve order: start from multitask order, then add any solo-only models at the end
model_order = list(model_cols_multi) + [
    m for m in model_cols_solo if m not in model_cols_multi
]


# Melt to long format and convert scores to numeric
def melt_scores(df, source_label):
    long_df = df.melt(
        id_vars=["Task", "Metric"],
        value_vars=[c for c in df.columns if c not in fixed_cols],
        var_name="Model",
        value_name="Score",
    )
    long_df["Score"] = pd.to_numeric(long_df["Score"], errors="coerce")
    long_df["source"] = source_label
    return long_df.dropna(subset=["Score"])


long_multi = melt_scores(df_multi, "multi")
long_solo = melt_scores(df_solo, "solo")

# Task sets and plotting order
multi_tasks = {"Min/Max", "Interval", "Sorting", "Add", "Mult", "Div", "Fineweb"}
solo_tasks = {"Exp", "Mean", "Std"}
tasks_order = [
    "Add",
    "Mult",
    "Div",
    "Mean",
    "Std",
    "Min/Max",
    "Interval",
    "Sorting",
    "Fineweb",
    "Exp",
]

# Select correct source per task
df_all = pd.concat(
    [
        long_multi[long_multi["Task"].isin(multi_tasks)],
        long_solo[long_solo["Task"].isin(solo_tasks)],
    ],
    ignore_index=True,
)

# Compute display value: percentages for non-Perplexity metrics
is_perplexity = df_all["Metric"].str.contains("Perplexity", case=False, na=False)
df_all["ValuePlot"] = np.where(is_perplexity, df_all["Score"], df_all["Score"] * 100.0)

# Drop models with no data across selected tasks
present_models = (
    df_all.groupby("Model")["ValuePlot"]
    .apply(lambda s: np.isfinite(s).any())
    .pipe(lambda s: s[s].index.tolist())
)
model_order = [m for m in model_order if m in present_models]

# --- Replace your palette logic with this ---
OURS = "BitToken (Ours)"

# Palette A (Okabe–Ito inspired)
model_to_color = {
    "Subword": "#0072B2",
    "Single Digit": "#56B4E9",
    "xVal": "#E69F00",
    "FoNE": "#FFBE7D",
    OURS: "#CC79A7",
}

# Keep only colors for present models; preserve your existing model_order
model_to_color = {m: model_to_color[m] for m in model_order if m in model_to_color}

# Styling similar to tokens_vs_performance_matrix.png
sns.set_theme(context="talk", style="whitegrid", font_scale=1.2)
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.titlepad": 12,
        "axes.labelpad": 8,
    }
)

# Create 2x5 subplots
fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=False, sharey=False)
axes = axes.flatten()

# Background color for solo-task subplots
solo_bg = "#f0f0f0"

# Sets for y-axis label logic
ylabel_logsmape_tasks = {"Add", "Exp"}
ylabel_remove_tasks = {"Mult", "Mean", "Std", "Interval", "Sorting", "Div"}

# Plot per task
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 18
SUBPLOT_TITLE_SIZE = 18
for idx, task in enumerate(tasks_order):
    ax = axes[idx]
    sub = df_all[df_all["Task"] == task].copy()
    if sub.empty:
        ax.set_title(task)
        ax.axis("off")
        continue

    # Ensure models are in consistent order and present
    sub["Model"] = pd.Categorical(sub["Model"], categories=model_order, ordered=True)
    sub = sub.sort_values("Model")

    # Grey background for solo tasks
    # if task in solo_tasks:
    # ax.patch.set_facecolor(solo_bg)

    # Barplot: one bar per model
    sns.barplot(
        data=sub,
        x="Model",
        hue="Model",
        y="ValuePlot",
        legend=False,
        order=model_order,
        palette=model_to_color,
        ax=ax,
        edgecolor="k",
        linewidth=0.5,
    )

    # Titles
    long_titles = {
        "Add": "Addition",
        "Div": "Division",
        "Exp": "Exponentiation",
        "Interval": "Interval",
        "Mean": "Mean",
        "Min/Max": "Min/Max",
        "Mult": "Multiplication",
        "Sorting": "Sorting",
        "Std": "Std",
        "Fineweb": "Fineweb",
    }

    if task == "Fineweb":
        ax.set_title(task + "⬇", fontsize=SUBPLOT_TITLE_SIZE)
    else:
        ax.set_title(long_titles[task], fontsize=SUBPLOT_TITLE_SIZE)
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    # Y label rules

    if task == "Fineweb":
        ax.set_ylabel("Perplexity", fontsize=AXIS_LABEL_SIZE)
    elif task in ylabel_logsmape_tasks:
        ax.set_ylabel("log-sMAPE (%)", fontsize=AXIS_LABEL_SIZE)
    elif task in ylabel_remove_tasks:
        ax.set_ylabel("")
    elif task == "Min/Max":
        ax.set_ylabel("Exact match\naccuracy (%)", fontsize=AXIS_LABEL_SIZE)
    else:
        ax.set_ylabel("Performance (%)")

    # X tick labels: insert newline for BitToken (Ours)
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    tick_labels = [
        lbl.replace("BitToken (Ours)", "BitToken") if isinstance(lbl, str) else lbl
        for lbl in tick_labels
    ]
    texts = ax.set_xticklabels(tick_labels, rotation=90, ha="center")
    for text in texts:
        if text.get_text() == "BitToken":
            text.set_fontweight("bold")

    # Grid style
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)

# Turn off any unused axes (safety)
for j in range(len(tasks_order), len(axes)):
    axes[j].axis("off")

# Build legend beneath the plot
# handles = [Patch(facecolor=model_to_color[m], edgecolor="k", label=m) for m in model_order]
# fig.legend(
#     handles=handles,
#     loc="lower center",
#     bbox_to_anchor=(0.5, -0.1),
#     ncol=min(len(model_order), 5),
#     frameon=True,
#     title="Models",
#     fontsize=16,
#     title_fontsize=16,
# )
# Remove legend

fig.tight_layout()
fig.subplots_adjust(bottom=0.22)  # make room for legend
# Shade the entire subplot slot for the solo tasks (Mean, Std, Exp)
# solo_indices = [tasks_order.index(t) for t in ["Mean", "Std", "Exp"]]
# for i in solo_indices:
#     bbox = axes[i].get_position()  # in figure (0-1) coords
#     fig.add_artist(Rectangle(
#         (bbox.x0, bbox.y0), bbox.width, bbox.height,
#         transform=fig.transFigure,
#         facecolor=solo_bg, edgecolor="none", zorder=0
#     ))
#     axes[i].set_zorder(1)  # keep the axes above the background patch


highlight_axes = [axes[i] for i in [0, 4, 5, 9]]

# Get bounding boxes in figure coordinates
bboxes = [ax.get_position() for ax in highlight_axes]

# Compute combined bounds
x0 = min(b.x0 for b in bboxes) - 0.07
y0 = min(b.y0 for b in bboxes) - 0.2
x1 = max(b.x1 for b in bboxes) + 0.01
y1 = max(b.y1 for b in bboxes) + 0.08

# Create rectangle in figure coordinates
rect = Rectangle(
    (x0, y0),
    x1 - x0,
    y1 - y0,
    transform=fig.transFigure,  # use figure coordinates
    fill=False,
    color="grey",
    linewidth=2,
    linestyle="--",
)


# Add label in bottom-right corner of the rectangle
fig.text(
    x1 - 0.01,  # a little inside the right edge
    y1 + 0.01,  # a little above the bottom edge
    "Solo-task",
    ha="right",
    va="bottom",
    fontsize=20,
    color="black",
    fontweight="bold",
)

# Add label in bottom-right corner of the rectangle
fig.text(
    x0 + 0.01,  # a little inside the right edge
    y1 + 0.01,  # a little above the bottom edge
    "Multi-task",
    ha="left",
    va="bottom",
    fontsize=20,
    color="black",
    fontweight="bold",
)
# Add rectangle to figure
# fig.add_artist(rect)

from matplotlib.lines import Line2D

tl_x = axes[8].get_position().x0 - 0.04
tr_x = axes[8].get_position().x1 + 0.01
t_y = axes[8].get_position().y1 + 0.08

line = Line2D(
    [tl_x, tr_x],
    [t_y, t_y],
    transform=fig.transFigure,
    color="grey",
    linewidth=2,
    linestyle="--",
)
fig.add_artist(line)

line = Line2D(
    [tl_x, tl_x],
    [t_y, y1],
    transform=fig.transFigure,
    color="grey",
    linewidth=2,
    linestyle="--",
)
fig.add_artist(line)

line = Line2D(
    [tr_x, tr_x],
    [t_y, y0],
    transform=fig.transFigure,
    color="grey",
    linewidth=2,
    linestyle="--",
)
fig.add_artist(line)


plt.tight_layout()
plt.savefig(
    "./frontier_model_analysis/results/analysis/results.pdf",
    bbox_inches="tight",
)
plt.savefig(
    "./frontier_model_analysis/results/analysis/results.png",
    bbox_inches="tight",
)
