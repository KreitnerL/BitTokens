# BitTokens: Efficient numeracy in language models through single-token number embeddings

<p align="center">
    <a href="https://arxiv.org/abs/2510.06824" target="_blank" rel="noopener"><img src="https://img.shields.io/badge/arXiv-2510.06824-b31b1b.svg" alt="arxiv paper"></a>
    <a href="https://openreview.net/forum?id=Bh4Ubk80M8" target="_blank" rel="noopener"><img src="https://img.shields.io/badge/ICML 2026-Spotlight-gold" alt="ICML 2026 Spotlight"></a>
    <a href="https://kreitnerl.github.io/BitTokens/" target="_blank" rel="noopener"><img src="https://img.shields.io/badge/Website-BitTokens-32a852" alt="BitTokens Website"></a>
    <a href="bittokens_notebook.ipynb" target="_blank" rel="noopener"><img src="https://img.shields.io/badge/Jupyter-Notebook-436972" alt="Jupyter notebook (browser)"></a>
    <a href="https://huggingface.co/datasets/KreitnerL/BitTokens-dataset" target="_blank" rel="noopener"><img src="https://img.shields.io/badge/HuggingFace-Dataset-ffd21e" alt="Hugging Face dataset"></a>
    <a href="LICENSE" target="_blank" rel="noopener"><img src="https://img.shields.io/badge/License-MIT-blue" alt="MIT License"></a>
</p>


LLMs perform poorly on arithmetic tasks, requiring excessive reasoning tokens to achieve good performance. We propose BitTokens,
a novel encoding strategy that represents any number as a single token using its IEEE 754 binary floating-point representation. This single-token number encoding allows language models to solve arithmetic tasks both effectively and efficiently.
![Figure 1](/images/fig1.png)


## How to use BitTokens
To get started check out our interactive [Jupyter notebook](bittokens_notebook.ipynb).

A more detailed implementation of BitTokens can be found in the [bittoken_embedding.py](networks/number_embedding_modules/bittoken_embedding.py) file.


## Setup
### Package manager UV
> [!TIP]
> We recommend using the fast package manager uv for dependency management, but you may use any other package manager. We provide an additional `requirements.txt` file for this. Replace `uv run` with `python` in the commands.

1. Download and install the fast package manager [UV](https://docs.astral.sh/uv/#highlights). 
    ```sh
    # Download and install uv with python version >=3.13
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2. Sync uv environment
    ```sh
    # Installs Python 3.13, PyTorch 2.11 + CUDA 12.6, FlashAttention, and other dependencies
    uv sync
    source .venv/bin/activate
    ```

`uv sync` installs the pinned, pre-built FlashAttention wheel for Linux x86_64, Python 3.13, PyTorch 2.11, CUDA 12, and
the C++11 ABI; the matching PyTorch CUDA 12.6 wheel; and Muon from a pinned Git commit. Do not replace either GPU package with a different CUDA build unless you deliberately update the compatible pins together.

Verify the installation:

```sh
uv run python -c 'import torch, flash_attn, muon; print(torch.__version__, torch.version.cuda, flash_attn.__version__)'
```



### Prepare Environment
1. Create an `.env` file and define the following variables:
   ```sh
   PROJECT_PATH=... # Absolute path to the 'BitTokens/' folder
   DATA_PATH=...    # Absolute path to data folder

   # [Optional] If you want to use the eval_scripts
   OPENROUTER_API_KEY=...
   ```

2. For convenience, load the `.env` file to execute the next commands.
    ```sh
    source .env
    ```

### Get the datasets
#### Exact paper dataset
To reproduce the manuscript-style training commands below, download the exact synthetic number-problem dataset used by the paper. Set `DATA_PATH` to the directory where the files should be placed, then run:

```sh
uv run --with huggingface_hub hf download KreitnerL/BitTokens-dataset --repo-type dataset --local-dir "$DATA_PATH"
```

Dataset page: https://huggingface.co/datasets/KreitnerL/BitTokens-dataset

The dataset contains all synthetic number-problem CSV files referenced by the BitToken configs and the FoNE, xVal, significant-digit, token-digit, and base-10 baseline configs. It includes the standard arithmetic tasks plus the hard tasks: Exponentiation, Mean, and Std. It also includes the binary-uniform curriculum files used by BitTokens where referenced by the configs.

The hosted dataset has 37 CSV files: 14 train CSVs, 14 validation CSVs, and 9 test CSVs. It intentionally does not include FineWeb-derived `.txt` files; those should be downloaded from the public FineWeb dataset instead.

The hosted CSV files keep only the columns required for training and evaluation: `prompt`, `text_prompt`, `answer`, `difficulty`, and `difficulty_sd`.

#### FineWeb text data
The multitask configs mix the synthetic number-problem data with text data. Download FineWeb from its original public Hugging Face dataset rather than from this repo:

```sh
uv run --with huggingface_hub hf download HuggingFaceFW/fineweb \
  --repo-type dataset \
  --include "sample/10BT/*.parquet" \
  --local-dir "$DATA_PATH"
```

Decode the downloaded parquet files to text files:

```sh
uv run $PROJECT_PATH/data_generation/decode_fineweb.py \
  --folder_dir "$DATA_PATH/sample/10BT/" \
  --save_path "$DATA_PATH/"
```

The training configs expect the FineWeb text files at `$DATA_PATH/000_00000_train.txt` and `$DATA_PATH/val_text.txt`. If your decoded files have different names, create those train/validation text files under `$DATA_PATH` before launching training.

#### Regenerate synthetic number problems
You can also generate fresh number problems locally. This is useful for development, but it will not produce the exact same examples used in the paper, so training results may differ.

1. Generate the number problems for each task for each phase:
    ```sh
    # Decimal version (used for all base-10 baselines and for testing)
    uv run $PROJECT_PATH/data_generation/data_generation_v2.py --save_dir $DATA_PATH
    # Binary version (used for BitToken training)
    uv run $PROJECT_PATH/data_generation/data_generation_v2.py --save_dir $DATA_PATH --significant_digits_distribution binary_uniform
    ```
2. Download and decode FineWeb as described above if you want to run the mixed numeric/text multitask configs.


## Evaluating the pretrained model
We offer the checkpoint of a pretrained multiTask BitToken model under [releases](https://github.com/KreitnerL/BitTokens/releases). Download the and extraxt the zip folder and place the output in the project directory. You can use the preconfigured validation datasets like this:

```sh
uv run eval.py --load_config_from $DATA_PATH/configs/config_bittoken_multiTask.py --tqdm --verbose --num_workers 16 --model_dir $DATA_PATH/trained/multiTask/bittoken/2026-03-29_19-08-00-833751_961/best_checkpoint
```

Please note that this checkpoint is not identical to the one referenced in the paper, as we recently renamed several model layers—which breaks loading compatibility for older weights. However, this updated model yields very similar performance.

## Running experiments
To recreate a BitToken model in a multiTask setting similar to the manuscript, run:
```sh
uv run $PROJECT_PATH/train.py --load_config_from $PROJECT_PATH/configs/config_bittoken_multiTask.py --tqdm --verbose --deterministic --seed 999
```
> [!NOTE]
> 
> The first run has a longer startup time because it tokenizes the entire dataset first and stores it in a cache directory under `$DATA_PATH/`.

This has been tested on a `Nvidia DGX A100 80GB` GPU. The results will be stored in the folder `$PROJECT_PATH/trained`.

## Citation
If you find our work useful, please cite our ICML 2026 paper:
```bibtex
@inproceedings{
    kreitner2026bittokens,
    title={Efficient numeracy in language models through single-token number embeddings},
    author={Linus Kreitner and Paul Hager and Jonathan Mengedoht and Georgios Kaissis and Daniel Rueckert and Martin J. Menten},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026},
    url={https://openreview.net/forum?id=Bh4Ubk80M8}
}
```
