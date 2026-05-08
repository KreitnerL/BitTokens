# BitTokens: Efficient numeracy in language models through single-token number embeddings

<p align="center">
<a href="https://arxiv.org/abs/2510.06824"><img src="https://img.shields.io/badge/arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arxiv version"></a>
<img src="https://icml.cc/static/core/img/ICML-logo.svg" height="28x">
<a href="https://scholar.google.com/citations?user=huPvQJIAAAAJ"><img src="https://img.shields.io/badge/Google%20Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white"/></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"></a>
</p>

LLMs perform poorly on arithmetic tasks, requiring excessive reasoning tokens to achieve good performance. We propose BitTokens,
a novel encoding strategy that represents any number as a single token using its IEEE 754 binary floating-point representation. This single-token number encoding allows language models to solve arithmetic tasks both effectively and efficiently.
![Figure 1](/images/fig1.png)


## BitTokens
The implementation of BitTokens can be found in the [float64.embedding.py](networks/number_embedding_modules/float64_embedding.py) file.


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
    # Installs python 3.13, torch 2.11, and other dependencies
    uv sync
    ```

### Install remaining dependencies:
> [!NOTE]
> 
> At the time of writing, there exists no official pre-built wheel for [FlashAttention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) with `torch=2.11` and `python=3.13`. You can use [this](https://github.com/Dao-AILab/flash-attention/issues/2425#issue-4196009498) approach instead.

> [!TIP]
> Sometimes [FlashAttention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) causes trouble when installing. If you run into an error, please refer to the official install guide.
    
```sh
uv pip install flash-attn==2.8.3 --no-build-isolation
uv pip install git+https://github.com/KellerJordan/Muon
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
1. Generate the number problems for each task for each phase (roughly 20G):
    ```sh
    # Decimal version (used for all base-10 baselines and for testing)
    uv run $PROJECT_PATH/data_generation/data_generation_v2.py --save_dir $DATA_PATH
    # Binary version (used for BitToken training)
    uv run $PROJECT_PATH/data_generation/data_generation_v2.py --save_dir $DATA_PATH --significant_digits_distribution binary_uniform
    ```
2. Download the fineweb text data
    Download the fineweb_10BT subset from https://huggingface.co/datasets/HuggingFaceFW/fineweb and save it under `$DATA_PATH/`
3. Decode fineweb to `.txt` (roughly 2.4G)
    ```sh
    uv run $PROJECT_PATH/data_generation/decode_fineweb.py --folder_dir $DATA_PATH/sample/10BT/ --save_path $DATA_PATH/
    ```


## Running experiments
To recreate a BitToken model in a multiTask setting similar to the manuscript, run:
```sh
uv run $PROJECT_PATH/train.py --load_config_from $PROJECT_PATH/configs/config_fe_multiTask.py --tqdm --verbose --deterministic --seed 999
```
This has been tested on a `Nvidia DGX A100 80GB` GPU.

The results will be stored in the folder `$PROJECT_PATH/trained`.

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