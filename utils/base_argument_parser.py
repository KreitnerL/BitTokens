import logging
from abc import ABC
from argparse import Namespace
from pathlib import Path
from sys import argv
from typing import Literal, Optional, TypeVar, override

import torch
from tap import Tap

from utils.enums import COMBINE_STRATEGY, MODEL_TYPE, NUMBER_HEAD, POSITION_EMBEDDING

TapType = TypeVar("TapType", bound="Tap")

def cast_to_torch_dtype(value: str) -> torch.dtype:
    return getattr(torch, value.removeprefix("torch."))

class BaseArgumentParser(Tap, ABC):
    load_config_from: Optional[Path] = None # Load configuration from a file
    save_dir: Optional[Path] = None           # Directory to save the experiment artifacts
    tokenizer_dir: Path                      # Path to tokenizer directory

    combine_strategy: COMBINE_STRATEGY = "sum" # Strategy to combine number encoding with input embeddings
    num_embedding_type: Literal["fone", "xval","float64", "base10"] = "float64" # Type of loss to use for numerical values
    float_type: Literal["float32", "float64"] = "float64" # Type of float to use for numerical values
    base: int = 2                     # Base for number encoding
    add_reciprocal: bool = False        # Whether to include 1/x in the embedding
    normalize_num_embedding: bool = False # Whether to normalize the number embedding to [-1, 1]
    dropout: float = 0 # Dropout for the model. By default, 0.1 dropout is used
    num_loss_type: str = "mse" # Type of loss to use for numerical values
    num_head_type: NUMBER_HEAD = NUMBER_HEAD.LINEAR # Type of number prediction head
    norm: Literal["rms", "layer"] = "rms" # Normalization function to use in attention layers
    precision_type: Literal["float16", "bfloat16", "float32", "float64"] = "float64" # Precision type for number embeddings


    model: Optional[MODEL_TYPE] = "rope_gpt2" # Model to use
    position_embedding: Optional[POSITION_EMBEDDING] = None # Position embedding to use
    n_embd: int = 768             # Dimension of the embeddings
    n_head: int = 6               # Number of heads
    n_layer: int = 6              # Number of layers
    tie_word_embeddings: bool = False        # Tie word embeddings with output layer
    context_length: int = 1024     # Context length
    cache_base_path: Optional[Path]=None    # Path to cache directory. Only used for pretokenized datasets
    attn_implementation: Optional[Literal["flash_attention_2", "sdpa"]] = "flash_attention_2" # Attention implementation to use
    difficulty_column: str = "difficulty" # Column name for difficulty in curriculum datasets
    
    device_batch_size: int = 48   # Batch size
    default_vram: int = 50_905_677_824      # GPU VRAM in byte for batch_size
    num_workers: int = 0          # Number of workers for dataloader
    compile: bool = False                   # Compile the model
    
    verbose: bool = False                   # Print verbose output
    tqdm: bool = False                      # Use tqdm for progress bar
    device: torch.device = torch.device("cuda:0") # Device to use
    data_type: str = str(torch.bfloat16) # Data type. By default, bfloat16 is used. Set to turn off amp
    _config_name: str = "base_config"       # Name of the configuration object in config files

    def verify_arguments(self):
        if self.verbose:
            logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)

        match self.model:
            case "gpt2" | "stem":
                if self.position_embedding != "ape":
                    if self.position_embedding == "rope" or self.position_embedding == "rope_legacy":
                        logging.warning(f"Warning: 'rope' position embedding is not supported for {self.model}. Using 'ape' instead")
                    self.position_embedding = "ape"

            case "rope_gpt2" | "rope_stem" | "modded_nanoGPT_stem" | "modded_nanoGPT":
                if self.position_embedding not in ["rope", "rope_legacy"]:
                    if self.position_embedding == "ape":
                        logging.warning(f"Warning: 'ape' position embedding is not supported for {self.model}. Using 'rope' instead")
                    self.position_embedding = "rope"
        self.data_type = cast_to_torch_dtype(self.data_type)
        self.precision_type = cast_to_torch_dtype(f"torch.{self.precision_type}")
        
        assert torch.cuda.is_available(), "CUDA is not available"
        vram = torch.cuda.get_device_properties(0).total_memory
        if vram > 1.6 * self.default_vram:
            self.device_batch_size = self.device_batch_size * 2
            print(f"VRAM: {vram} bytes. Doubling batch size: {self.device_batch_size}")

    @override
    def parse_args(self, args = None, known_only = False, legacy_config_parsing=False) -> "BaseArgumentParser": # pyright: ignore[reportRedeclaration]
        if "--load_config_from" in argv:
            config_path = argv[argv.index("--load_config_from") + 1]
            config_vars = {}
            exec(open(config_path).read(), config_vars)
            config: Namespace = config_vars[self._config_name]
            for k,v in vars(config).items():
                if f"--{k}" in argv:
                    continue
                elif isinstance(v, bool):
                    if v:
                        argv.append(f"--{k}")
                elif isinstance(v,list):
                    argv.append(f"--{k}")
                    argv.extend([str(vv) for vv in v])
                else:
                    argv.extend([f"--{k}", str(v)])
        args: BaseArgumentParser = super().parse_args(argv[1:], known_only, legacy_config_parsing)
        args.verify_arguments()
        return args
