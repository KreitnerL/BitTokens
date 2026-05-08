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
base_config.tokenizer_dir = f"{PROJECT_PATH}/tokenizers/num_text/fe_gpt2"
base_config.model = "rope_stem"
base_config.num_embedding_type = "float64"
base_config.normalize_num_embedding = False
base_config.add_reciprocal = True
base_config.combine_strategy = "zero_pad"
base_config.num_loss_type = "bce"

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
base_config.difficulty_column = "difficulty"


############################################
# Training configuration
############################################
train_config = cast(TrainArgumentParser, Namespace(**vars(base_config)))
# Training data parameters
train_set_paths_and_curriculum_types = {
    f"{DATA_PATH}/Addition_decimal_uniform_train_30M.csv.gz": (DATASET_CURRICULUM_TYPE.STANDARD, f"{DATA_PATH}/cache/fe_gpt2_47200109/21903304"),
    f"{DATA_PATH}/Multiplication_binary_uniform_train_30M.csv.gz": (DATASET_CURRICULUM_TYPE.CURRICULUM, f"{DATA_PATH}/cache/fe_gpt2_47200109/62268622"),
    f"{DATA_PATH}/Multiplication_decimal_uniform_train_30M.csv.gz": (DATASET_CURRICULUM_TYPE.ENDGAME, f"{DATA_PATH}/cache/fe_gpt2_47200109/94881869"),
    f"{DATA_PATH}/DivM_binary_uniform_train_30M.csv.gz": (DATASET_CURRICULUM_TYPE.CURRICULUM, f"{DATA_PATH}/cache/fe_gpt2_47200109/62072059"),
    f"{DATA_PATH}/Division_decimal_uniform_train_30M.csv.gz": (DATASET_CURRICULUM_TYPE.ENDGAME, f"{DATA_PATH}/cache/fe_gpt2_47200109/26549495"),
    f"{DATA_PATH}/Exponentiation_binary_uniform_train_30M.csv.gz": (DATASET_CURRICULUM_TYPE.CURRICULUM, f"{DATA_PATH}/cache/fe_gpt2_47200109/28310628"),
    f"{DATA_PATH}/Exponentiation_decimal_uniform_train_30M.csv.gz": (DATASET_CURRICULUM_TYPE.ENDGAME, f"{DATA_PATH}/cache/fe_gpt2_47200109/13731216"),
    f"{DATA_PATH}/MinMax_decimal_uniform_train_30M.csv.gz": (DATASET_CURRICULUM_TYPE.STANDARD, f"{DATA_PATH}/cache/fe_gpt2_47200109/21268272"),
    f"{DATA_PATH}/Interval_decimal_uniform_train_30M.csv.gz": (DATASET_CURRICULUM_TYPE.STANDARD, f"{DATA_PATH}/cache/fe_gpt2_47200109/15330129"),
    f"{DATA_PATH}/Sorting_decimal_uniform_train_30M.csv.gz": (DATASET_CURRICULUM_TYPE.STANDARD, f"{DATA_PATH}/cache/fe_gpt2_47200109/25805100"),
    f"{DATA_PATH}/Mean_binary_uniform_train_30M.csv.gz": (DATASET_CURRICULUM_TYPE.CURRICULUM, f"{DATA_PATH}/cache/fe_gpt2_47200109/49984459"),
    f"{DATA_PATH}/Mean_decimal_uniform_train_30M.csv.gz": (DATASET_CURRICULUM_TYPE.ENDGAME, f"{DATA_PATH}/cache/fe_gpt2_47200109/73768784"),
    f"{DATA_PATH}/Std_binary_uniform_train_30M.csv.gz": (DATASET_CURRICULUM_TYPE.CURRICULUM, f"{DATA_PATH}/cache/fe_gpt2_47200109/80076284"),
    f"{DATA_PATH}/Std_decimal_uniform_train_30M.csv.gz": (DATASET_CURRICULUM_TYPE.ENDGAME, f"{DATA_PATH}/cache/fe_gpt2_47200109/69357492"),
    f"{DATA_PATH}/000_00000_train.txt": (DATASET_CURRICULUM_TYPE.STANDARD,  f"{DATA_PATH}/cache/fe_gpt2_47200109/13591814")
}
train_config.train_set_paths = list(train_set_paths_and_curriculum_types.keys())
train_config.train_dataset_curriculum_types = [v[0] for v in train_set_paths_and_curriculum_types.values()]
# train_config.train_cache_paths = [v[1] for v in train_set_paths_and_curriculum_types.values()]

# Data mixing parameters
TASK_RATIO = 1
NUM_TASKS = len([d for d in train_config.train_dataset_curriculum_types if d != DATASET_CURRICULUM_TYPE.ENDGAME])
train_config.train_set_ratios = [TASK_RATIO / NUM_TASKS if d != DATASET_CURRICULUM_TYPE.ENDGAME else 0 for d in train_config.train_dataset_curriculum_types]
train_config.num_loss_weight = 10
train_config.train_dataset_type = "pretokenized_number"

train_config.optimize_last = True
train_config.use_curriculum = True
train_config.weight_decay = 0
train_config.lr = 0.02
train_config.curriculum_generalized_mean_power = 0
train_config.device_batch_size = 48
train_config.effective_batch_size = 192
train_config.lr_scheduler_type = "cosine"

# Validation data parameters
train_paths_metrics_dataset_types: dict[str, tuple[str, str]] = {
    f"{DATA_PATH}/Addition_decimal_uniform_val_10k.csv.gz": (MetricFunction.LOG_SMAPE, "efficient_number_prompt"),
    f"{DATA_PATH}/Multiplication_binary_uniform_val_10k.csv.gz": (MetricFunction.LOG_SMAPE, "curriculum_number"),
    f"{DATA_PATH}/Multiplication_decimal_uniform_val_10k.csv.gz": (MetricFunction.LOG_SMAPE, "efficient_number_prompt"),
    f"{DATA_PATH}/DivM_binary_uniform_val_10k.csv.gz": (MetricFunction.LOG_SMAPE, "curriculum_number"),
    f"{DATA_PATH}/Division_decimal_uniform_val_10k.csv.gz": (MetricFunction.LOG_SMAPE, "efficient_number_prompt"),
    f"{DATA_PATH}/Exponentiation_binary_uniform_val_10k.csv.gz": (MetricFunction.LOG_SMAPE, "curriculum_number"),
    f"{DATA_PATH}/Exponentiation_decimal_uniform_val_10k.csv.gz": (MetricFunction.LOG_SMAPE, "efficient_number_prompt"),
    f"{DATA_PATH}/MinMax_decimal_uniform_val_10k.csv.gz": (MetricFunction.EXACT_NUMBER_ACC, "efficient_number_prompt"),
    f"{DATA_PATH}/Interval_decimal_uniform_val_10k.csv.gz": (MetricFunction.NORMALIZED_QUINT_CLASS_ACC, "efficient_number_prompt"),
    f"{DATA_PATH}/Sorting_decimal_uniform_val_10k.csv.gz": (MetricFunction.EXACT_NUMBER_ACC, "efficient_number_prompt"),
    f"{DATA_PATH}/Mean_binary_uniform_val_10k.csv.gz": (MetricFunction.LOG_SMAPE, "curriculum_number"),
    f"{DATA_PATH}/Mean_decimal_uniform_val_10k.csv.gz": (MetricFunction.LOG_SMAPE, "efficient_number_prompt"),
    f"{DATA_PATH}/Std_binary_uniform_val_10k.csv.gz": (MetricFunction.LOG_SMAPE, "curriculum_number"),
    f"{DATA_PATH}/Std_decimal_uniform_val_10k.csv.gz": (MetricFunction.LOG_SMAPE, "efficient_number_prompt"),
    f"{DATA_PATH}/val_text.txt": (MetricFunction.SCALED_PPL, "pretokenized_number", False)
}
train_config.val_set_paths = list(train_paths_metrics_dataset_types.keys())
train_config.val_set_metrics = [v[0] for v in train_paths_metrics_dataset_types.values()]
train_config.val_dataset_types = [v[1] for v in train_paths_metrics_dataset_types.values()]

train_config.val_additional_metrics = [
    MetricFunction.LOG_SMAPE,
    MetricFunction.EXACT_NUMBER_ACC,
]
# Training hyperparameters
train_config.save_dir = f"{PROJECT_PATH}/trained/multiTask_hard/fe"
train_config.train_token_budget = 10_000_000_000
train_config.num_warmup_tokens = train_config.train_token_budget//10
train_config.eval_every_k_tokens = 32*384*1024
train_config.max_eval_steps = 2
train_config.no_save_latest = True

# Dynamic loss weighting parameters
train_config.loss_weight_momentum = 0.5
train_config.online_weighting_warmup_tokens = 192*384*1024
train_config.reset_loss_after_warmup = True
train_config.grad_clip = -1

# WandB parameters
train_config.wandb_project = "STEM"
train_config.wandb_group = "multi_task_hard_fe"



############################################
# Evaluation configuration
############################################
eval_config = cast(EvalArgumentParser, Namespace(**vars(base_config)))
# Evaluation data parameters
test_paths_metrics_dataset_types_save_pred: dict[str, tuple[str, str, bool]] = {
    f"{DATA_PATH}/Addition_decimal_uniform_test_10k.csv": (MetricFunction.LOG_SMAPE, "efficient_number_prompt", True),
    f"{DATA_PATH}/Multiplication_decimal_uniform_test_10k.csv": (MetricFunction.LOG_SMAPE, "efficient_number_prompt", True),
    f"{DATA_PATH}/Division_decimal_uniform_test_10k.csv": (MetricFunction.LOG_SMAPE, "efficient_number_prompt", True),
    f"{DATA_PATH}/Exponentiation_decimal_uniform_test_10k.csv": (MetricFunction.LOG_SMAPE, "efficient_number_prompt", True),
    f"{DATA_PATH}/MinMax_decimal_uniform_test_10k.csv": (MetricFunction.EXACT_NUMBER_ACC, "efficient_number_prompt", True),
    f"{DATA_PATH}/Interval_decimal_uniform_test_10k.csv": (MetricFunction.NORMALIZED_QUINT_CLASS_ACC, "efficient_number_prompt", True),
    f"{DATA_PATH}/Sorting_decimal_uniform_test_10k.csv": (MetricFunction.EXACT_NUMBER_ACC, "efficient_number_prompt", True),
    f"{DATA_PATH}/Mean_decimal_uniform_test_10k.csv": (MetricFunction.LOG_SMAPE, "efficient_number_prompt", True),
    f"{DATA_PATH}/Std_decimal_uniform_test_10k.csv": (MetricFunction.LOG_SMAPE, "efficient_number_prompt", True),
    f"{DATA_PATH}/val_text.txt": (MetricFunction.SCALED_PPL, "pretokenized_number", False)
}
eval_config.test_set_paths = list(test_paths_metrics_dataset_types_save_pred.keys())
eval_config.test_set_metrics = [v[0] for v in test_paths_metrics_dataset_types_save_pred.values()]
eval_config.test_dataset_types = [v[1] for v in test_paths_metrics_dataset_types_save_pred.values()]
eval_config.save_testset_predictions = [v[2] for v in test_paths_metrics_dataset_types_save_pred.values()]

eval_config.additional_metrics = [
    MetricFunction.LOG_SMAPE,
    MetricFunction.EXACT_NUMBER_ACC,
]
