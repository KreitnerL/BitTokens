from argparse import Namespace
from typing import cast

from dotenv import load_dotenv

from utils.base_argument_parser import BaseArgumentParser
from utils.enums import DATASET_CURRICULUM_TYPE
from utils.eval_argument_parser import EvalArgumentParser
from utils.metrics import MetricFunction
from utils.train_argument_parser import TrainArgumentParser

load_dotenv()
import os  # noqa: E402

PROJECT_PATH = os.getenv("PROJECT_PATH")
DATA_PATH = os.getenv("DATA_PATH")
############################################
# Base configuration
############################################
base_config: BaseArgumentParser = Namespace()

# Model architecture parameters
base_config.tokenizer_dir = f"{PROJECT_PATH}/tokenizers/num_text/td_gpt2"
base_config.model = "rope_gpt2"

base_config.dropout = 0
base_config.n_embd = 768
base_config.n_head = 6
base_config.n_layer = 6

# Dataset and caching parameters
base_config.cache_base_path = f"{DATA_PATH}/cache"

# Resource allocation parameters
base_config.compile = True
base_config.num_workers = 16
base_config.verbose = True
base_config.difficulty_column = "difficulty_sd"


############################################
# Training configuration
############################################
train_config = cast(TrainArgumentParser, Namespace(**vars(base_config)))
# Training data parameters
train_set_paths_and_curriculum_types = {
    f"{DATA_PATH}/Exponentiation_decimal_uniform_train_30M.csv.gz": (DATASET_CURRICULUM_TYPE.CURRICULUM, f"{DATA_PATH}/cache/td_gpt2_88525419/13731216"),
}

train_config.train_set_paths = list(train_set_paths_and_curriculum_types.keys())
train_config.train_dataset_curriculum_types = [v[0] for v in train_set_paths_and_curriculum_types.values()]
train_config.train_cache_paths = [v[1] for v in train_set_paths_and_curriculum_types.values()]

train_config.use_curriculum = True
train_config.optimize_last = True
train_config.weight_decay = 0
train_config.lr = 0.02
train_config.curriculum_generalized_mean_power = 0
train_config.device_batch_size = 48
train_config.effective_batch_size = 192
train_config.lr_scheduler_type = "cosine"


# Data mixing parameters
train_config.train_set_ratios = [1]
train_config.train_dataset_type = "pretokenized"

# Validation data parameters
train_paths_metrics_dataset_types: dict[str, tuple[str, str]] = {
    f"{DATA_PATH}/Exponentiation_decimal_uniform_val_10k.csv.gz": (MetricFunction.LOG_SMAPE, "curriculum"),
}
train_config.val_set_paths = list(train_paths_metrics_dataset_types.keys())
train_config.val_set_metrics = [v[0] for v in train_paths_metrics_dataset_types.values()]
train_config.val_dataset_types = [v[1] for v in train_paths_metrics_dataset_types.values()]

train_config.val_additional_metrics = [
    MetricFunction.EXACT_NUMBER_ACC,
]
# Training hyperparameters
train_config.save_dir = f"{PROJECT_PATH}/trained/soloTask/td"
train_config.train_token_budget = 10_000_000_000  # Rougly equals 3 epochs (4_717_802_025)
train_config.num_warmup_tokens = train_config.train_token_budget//20
train_config.eval_every_k_tokens = 16*384*1024  # 100 steps or 1% of the token budget
train_config.max_eval_steps = 2
train_config.no_save_latest = True

# Dynamic loss weighting parameters
train_config.loss_weight_momentum = 1
train_config.online_weighting_warmup_tokens = 1600*64*1024 
train_config.reset_loss_after_warmup = False #True
train_config.grad_clip = -1

# WandB parameters
train_config.wandb_project = "STEM"
train_config.wandb_group = "soloTask_td"


############################################
# Evaluation configuration
############################################
# Add evaluation configuration if needed, similar to the reference
eval_config = cast(EvalArgumentParser, Namespace(**vars(base_config)))
test_paths_metrics_dataset_types_save_pred: dict[str, tuple[str, str, bool]] = {
    f"{DATA_PATH}/Exponentiation_decimal_uniform_val_10k.csv.gz": (MetricFunction.LOG_SMAPE, "efficient_prompt", True),
}
eval_config.test_set_paths = list(test_paths_metrics_dataset_types_save_pred.keys())
eval_config.test_set_metrics = [v[0] for v in test_paths_metrics_dataset_types_save_pred.values()]
eval_config.test_dataset_types = [v[1] for v in test_paths_metrics_dataset_types_save_pred.values()]
eval_config.save_testset_predictions = [v[2] for v in test_paths_metrics_dataset_types_save_pred.values()]

eval_config.additional_metrics = [
    MetricFunction.LOG_SMAPE,
    MetricFunction.EXACT_NUMBER_ACC,
]