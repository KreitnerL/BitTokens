import logging
import os
from math import ceil  # type: ignore
from pathlib import Path
from random import randint
from random import seed as r_seed
from sys import maxsize
from typing import Literal, Optional, TypeVar, override

import torch
from numpy.random import seed as np_seed
from tap import Tap
from torch import manual_seed

from utils.base_argument_parser import BaseArgumentParser
from utils.enums import DATASET_CURRICULUM_TYPE, DATASET_TYPE
from utils.metrics import MetricFunction

TapType = TypeVar("TapType", bound="Tap")

class TrainArgumentParser(BaseArgumentParser):
    run_name: str = "" # Name of the run. If set, the run name is used instead of the default name
    train_set_paths: list[Path]=None        # Paths to training csv or txt files
    train_cache_paths: list[Path] = None # Paths to cache files. If set, the cache files are used instead of the training files
    train_dataset_type: DATASET_TYPE="pretokenized_number" # Dataset type of the training set
    train_dataset_curriculum_types: list[DATASET_CURRICULUM_TYPE] = None # Curriculum type of the training sets
    train_set_tokens: list[int] = None  # Number of tokens per training set
    train_set_ratios: list[float] = None # Ratios for each training set
    optimize_last: bool = False # Optimize the last training set via ratio sampling. Default is False
    train_set_loss_weights: list[float] = None # Loss weights for each training set
    loss_norm_magnitude: float = 10 # Magnitude of the loss norm. By default, the loss norm is set to 1
    loss_weight_momentum: float = 1 # Momentum for loss weight update. By default, 100% of the previous loss weight is retained
    task_loss_exponent: float = -1 # Exponent for the generaized mean objective. By default, the exponent is set to -1, which is equivalent to the harmonic mean
    reset_loss_after_warmup: bool = False # Reset loss weights after warmup. By default, the loss weights are not reset
    task_ratio_cap: list[float] = None # Maximum task sampling ratio for each task. If set, the task sampling ratios are capped at the specified values
    online_weighting_warmup_tokens: int = 96*384*1024 # Number of steps to warmup the online weighting. By default, no warmup is used
    num_loss_weight: float = 10     # Number of loss weights to use
    frequency_weight_slope: float = 0. # Slope of the linear frequency weight. By default, no frequency weighting is used
    deterministic: bool = False # Use deterministic computation. This may slow down training, but ensures reproducibility
    optimizer: Literal["AdamW", "Muon"] = "Muon" # Optimizer to use
    
    val_set_paths: list[Path]=None           # List of paths to validation csv files
    val_cache_paths: list[Path] = None # Paths to cache files. If set, the cache files are used instead of the validation files
    val_dataset_types: list[DATASET_TYPE]   # Dataset type of the validation set
    val_set_tokens: list[int] = None  # Number of tokens per val set
    val_set_metrics: list[MetricFunction] = None # Metrics to use for validation
    val_additional_metrics: list[MetricFunction] = None # Additional metrics to use for validation
    save_dir: str                           # Directory to save the experiment artifacts

    # Curriculum learning parameters
    use_curriculum: bool = False            # Enable curriculum learning
    curriculum_max_difficulty_fraction: float = 0.15     # Initial maximum difficulty level
    curriculum_advancement_threshold: float = 0.9 # Accuracy threshold to advance difficulty
    curriculum_new_difficulty_ratio: float = 0.3 # Ratio of data from new difficulties
    curriculum_generalized_mean_power: float = -1.0 # Power for generalized mean calculation (default: -1.0 for harmonic mean)
    curriculum_performance_history_size: int = 3 # Number of evaluations to track for advancement
    curriculum_advancement_difficulty_window_size: int = 3 # Number of evaluations to consider for advancement
    curriculum_advancement_performance_window_size: int = 1 # Number of evaluations to track for advancement
    curriculum_step_size: float = 0.02 # Number of difficulty levels to advance at once as a fraction of the total difficulty range
    preview_difficulty_ratio: float = 0.2  # Fraction of task ratio for preview difficulties (0.0 = disabled)
    preview_exponential_decay: float = 0.8 # Exponential decay rate for preview difficulties
    endgame_switch_step_fraction: float = 0.5 # Fraction of training steps to switch to endgame mode (>=1 = disabled)
    endgame_switch_lr_fraction: float = 0.5 # Fraction of learning rate to switch to in endgame mode (0 = disabled)
    min_difficulty_ratio: float = 0.5 # Minimum difficulty ratio to avoid zero probabilities (default: 1% of effective batch size)

    continue_from: Optional[Path] = None    # Path to the save dir of a pretrained model to continue training. Note that the latest checkpoint is being used.
    from_pretrained: Optional[Path] = None  # Path to the save dir of a pretrained model to start training from scratch
    lr: float = 0.02              # Learning rate
    adamW_lr: float = 0.02     # Baser learning rate for other parameters whe using AdamW optimizer
    effective_batch_size: int = 384 # Batch size including gradient accumulation
    weight_decay: float = 0.    # Weight decay
    lr_scheduler_type: Literal["cosine", "plateau"] = "cosine"       # Learning rate scheduler type
    grad_clip: float = -1          # Gradient clipping
    muon_momentum_warmup: int = 300 # Number of steps to warmup the muon momentum. By default, 300 steps are used

    num_warmup_steps: int = 1     # Number of warmup steps. If num_warmup_tokens is set, this is ignored
    num_warmup_tokens: int = 0    # Number of warmup tokens. If set, num_warmup_steps is ignored
    unique_samples: int = maxsize # Number of unique samples in the train set
    num_epochs: int = 5           # Number of training epochs. If max_train_steps or train_token_budget is set, this is ignored
    max_train_steps: int = maxsize# Maximum number of training steps. If train_token_budget is set, this is ignored.
    train_token_budget: int = 10_000_000_000 # Maximum number of tokens to train on. If set, max_train_steps and num_epochs is ignored
    stop_after: int = maxsize      # Stop training after n steps

    eval_every_k_steps: int = 0  # Evaluate every k steps.
    eval_every_k_tokens: int = 16*384*1024  # Evaluate every k tokens. If set, eval_every_k_steps is ignored    
    max_eval_steps: int = 2      # Maximum number of evaluation steps of size effective_batch_size.
    max_milestone_eval_steps: int = 50 # Maximum number of milestone evaluation steps
    max_milestone_eval_tokens: int = 0 # Maximum number of milestone evaluation tokens. If set, max_milestone_eval_steps is ignored
    save_checkpoint_steps: int = maxsize # Save checkpoint every n steps
    save_checkpoint_tokens: int = 0 # Save checkpoint every n tokens. If set, save_checkpoint_steps is ignored
    no_save_latest: bool = False            # Do not save the latest checkpoint

    seed: int = None              # Manual seed for reproducibility. By default, a random seed is used
    wandb: bool = False               # Use wandb to log training progress
    wandb_project: Optional[str] = None     # Wandb project name
    wandb_group: Optional[str] = None       # Wandb group name
    wandb_sweep_id: Optional[str] = False   # Wandb sweep ID
    _config_name: str = "train_config"      # Name of the configuration object in config files

    @override
    def verify_arguments(self):
        super().verify_arguments()
        assert self.effective_batch_size % self.device_batch_size == 0, f"Effective batch size ({self.effective_batch_size}) must be a multiple of device batch size ({self.device_batch_size})"
        assert self.save_checkpoint_steps == maxsize or self.save_checkpoint_steps % self.eval_every_k_steps == 0, "save_checkpoint_steps must be a multiple of eval_steps"

        assert len(self.val_set_paths) == len(self.val_dataset_types), f"Number of validation set paths ({len(self.val_set_paths)}) must be equal to the number of validation dataset types ({len(self.val_dataset_types)})"

        match self.model:
            case "stem" | "rope_stem" | "modded_nanoGPT_stem":
                assert self.train_dataset_type not in ["pretokenized", "efficient_prompt"], f"Train dataset type {self.train_dataset_type} is not supported for {self.model}. Use number datasets instead."
                for val_dataset_type in self.val_dataset_types:
                    assert val_dataset_type not in ["pretokenized", "efficient_prompt"], f"Val dataset type {val_dataset_type} is not supported for {self.model}. Use number datasets instead."
            case "gpt2" | "rope_gpt2" | "modded_nanoGPT":
                assert self.train_dataset_type not in ["pretokenized_number", "efficient_number_prompt", "pretokenized_number_pos", "efficient_number_prompt_pos"], f"Train dataset type {self.train_dataset_type} is not supported for {self.model}. Use prompt datasets instead."
                for val_dataset_type in self.val_dataset_types:
                    assert val_dataset_type not in ["pretokenized_number", "efficient_number_prompt", "pretokenized_number_pos", "efficient_number_prompt_pos"], f"Val dataset type {val_dataset_type} is not supported for {self.model}. Use prompt datasets instead."

        self.seed = self.seed or randint(0, 1000)
        # Additional reproducibility settings for PyTorch
        os.environ["PYTHONHASHSEED"] = "0"
        r_seed(self.seed)
        manual_seed(self.seed)
        np_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)  # For multi-GPU setups
        if self.deterministic:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        
        if self.num_warmup_tokens>0:
            self.num_warmup_steps = ceil((self.num_warmup_tokens / self.context_length) / self.device_batch_size)
            logging.info(f"Setting num_warmup_steps={self.num_warmup_steps} based on num_warmup_tokens")
        else:
            assert self.num_warmup_steps > 0, "num_warmup_steps or num_warmup_tokens must be set"
            self.num_warmup_tokens = self.num_warmup_steps * self.context_length * self.device_batch_size
            logging.warning(f"Setting num_warmup_tokens={self.num_warmup_tokens} based on num_warmup_steps. Please use num_warmup_tokens for comparability with other experiments")

        if self.eval_every_k_tokens>0:
            self.eval_every_k_steps = ceil((self.eval_every_k_tokens / self.context_length) / self.device_batch_size)
            logging.info(f"Setting eval_every_k_steps={self.eval_every_k_steps} based on eval_every_k_tokens")
        else:
            assert self.eval_every_k_steps > 0, "eval_every_k_steps or eval_every_k_tokens must be set"
            self.eval_every_k_tokens = self.eval_every_k_steps * self.context_length * self.device_batch_size
            logging.warning(f"Setting eval_every_k_tokens={self.eval_every_k_tokens} based on eval_every_k_steps. Please use eval_every_k_tokens for comparability with other experiments")
        assert self.eval_every_k_steps % (self.effective_batch_size // self.device_batch_size) == 0, f"eval_every_k_steps ({self.eval_every_k_steps}) must be a multiple of gradient accumulation steps ({self.effective_batch_size // self.device_batch_size})"

        if self.save_checkpoint_tokens>0:
            self.save_checkpoint_steps = ceil((self.save_checkpoint_tokens / self.context_length) / self.device_batch_size)
            self.save_checkpoint_steps = int(round(self.save_checkpoint_steps / self.eval_every_k_steps)) * self.eval_every_k_steps
            logging.info(f"Setting save_checkpoint_steps={self.save_checkpoint_steps} based on save_checkpoint_tokens")
        else:
            assert self.save_checkpoint_steps > 0, "save_checkpoint_steps or save_checkpoint_tokens must be set"
            assert self.save_checkpoint_steps == maxsize or self.save_checkpoint_steps % self.eval_every_k_steps == 0, "save_checkpoint_steps must be a multiple of eval_steps"
            self.save_checkpoint_tokens = self.save_checkpoint_steps * self.context_length * self.device_batch_size
            logging.warning(f"Setting save_checkpoint_tokens={self.save_checkpoint_tokens} based on save_checkpoint_steps. Please use save_checkpoint_tokens for comparability with other experiments")

        if self.train_dataset_curriculum_types is None:
            if self.use_curriculum:
                logging.warning(f"No curriculum types specified for training sets. Setting all to {DATASET_CURRICULUM_TYPE.CURRICULUM}.")
            self.train_dataset_curriculum_types = [DATASET_CURRICULUM_TYPE.CURRICULUM if self.use_curriculum else DATASET_CURRICULUM_TYPE.STANDARD] * len(self.train_set_paths)
        else:
            if DATASET_CURRICULUM_TYPE.CURRICULUM in self.train_dataset_curriculum_types:
                assert self.use_curriculum, f"{DATASET_CURRICULUM_TYPE.CURRICULUM} type specified for training sets, but use_curriculum is False. Please set use_curriculum to True."
        if self.train_set_ratios is None:
            num_tasks = sum([1 for d in self.train_dataset_curriculum_types if d != DATASET_CURRICULUM_TYPE.ENDGAME])
            self.train_set_ratios = [1.0 / num_tasks if d!= DATASET_CURRICULUM_TYPE.ENDGAME else 0 for d in self.train_dataset_curriculum_types]
        else:
            assert len(self.train_set_ratios) == len(self.train_set_paths), f"Number of training set tokens ({len(self.train_set_ratios)}) must be equal to the number of training sets ({len(self.train_set_paths)})"

        if self.task_ratio_cap is not None:
            assert len(self.task_ratio_cap) == len(self.train_set_paths), f"Number of task ratio caps ({len(self.task_ratio_cap)}) must be equal to the number of training sets ({len(self.train_set_paths)})"
            for cap in self.task_ratio_cap:
                assert 0 < cap <= 1, f"Each task ratio cap must be in (0,1], got {cap}"
            total_cap = sum(self.task_ratio_cap)
            assert total_cap >= 1.0, f"Sum of task ratio caps must be at least 1.0, got {total_cap}"
            self.task_ratio_cap = torch.tensor(self.task_ratio_cap, dtype=torch.float32)
        else:
            self.task_ratio_cap = torch.ones(len(self.train_set_paths), dtype=torch.float32)

        if self.train_set_tokens is None:
            self.train_set_tokens = [-1] * len(self.train_set_paths)
        if self.train_cache_paths is None:
            self.train_cache_paths = [None] * len(self.train_set_paths)
        if self.val_set_tokens is None:
            self.val_set_tokens = [-1] * len(self.val_set_paths)
        if self.val_cache_paths is None:
            self.val_cache_paths = [None] * len(self.val_set_paths)
        
        if self.train_set_loss_weights is None:
            self.train_set_loss_weights = torch.tensor([1.] * len(self.train_set_paths))
        else:
            assert len(self.train_set_loss_weights) == len(self.train_set_paths), f"Number of training set loss weights ({len(self.train_set_loss_weights)}) must be equal to the number of training sets ({len(self.train_set_paths)})"
            self.train_set_loss_weights = torch.tensor(self.train_set_loss_weights, dtype=torch.float32)
        if self.val_set_metrics is None:
            self.val_set_metrics = [MetricFunction(MetricFunction.TOKEN_EQUALITY)] * len(self.val_set_paths)
        else:
            assert len(self.val_set_metrics) == len(self.val_set_paths), f"Number of validation set metrics ({len(self.val_set_metrics)}) must be equal to the number of validation sets ({len(self.val_set_paths)}"
        
        if self.loss_weight_momentum < 0 or self.loss_weight_momentum > 1:
            raise ValueError(f"loss_weight_momentum must be between 0 and 1. Got {self.loss_weight_momentum}")
        elif self.loss_weight_momentum<1:
            assert self.reset_loss_after_warmup, "loss_weight_momentum < 1 requires reset_loss_after_warmup to be True"
        if self.wandb_sweep_id or self.wandb_group:
            assert self.wandb_project is not None, "wandb_project must be set if wandb_sweep_id or wandb_group is set"

        # Curriculum learning validation
        if self.use_curriculum:
            assert 0 < self.curriculum_advancement_threshold <= 1, f"curriculum_advancement_threshold must be in (0,1], got {self.curriculum_advancement_threshold}"
            assert 0 < self.curriculum_new_difficulty_ratio <= 1, f"curriculum_new_difficulty_ratio must be in (0,1], got {self.curriculum_new_difficulty_ratio}"
            assert self.curriculum_max_difficulty_fraction >= 0, f"curriculum_max_difficulty must be >= 0, got {self.curriculum_max_difficulty_fraction}"
            assert self.curriculum_performance_history_size > 0, f"curriculum_performance_history_size must be > 0, got {self.curriculum_performance_history_size}"
            assert 0 <= self.preview_difficulty_ratio <= 1, f"preview_difficulty_ratio must be in [0,1], got {self.preview_difficulty_ratio}"
            assert 0 < self.preview_exponential_decay <= 1, f"preview_exponential_decay must be in (0,1], got {self.preview_exponential_decay}"

        return self

    @override
    def parse_args(self, args = None, known_only = False, legacy_config_parsing=False) -> "TrainArgumentParser":
        return super().parse_args(args, known_only, legacy_config_parsing)
