from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Literal,
    NotRequired,
    Optional,
    OrderedDict,
    TypeAlias,
    TypedDict,
    Union,
)

import torch
from numpy import ndarray
from pandas import Series
from torch import DoubleTensor
from tqdm.asyncio import tqdm_asyncio
from transformers import (
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)
from transformers.generation.utils import (
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
GenerateBeamOutput = Union[GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput]
GenerateOutput = Union[GenerateNonBeamOutput, GenerateBeamOutput]

COMBINE_STRATEGY: TypeAlias = Literal["sum", "sum_scaled", "prod", "concat", "zero_pad", "weighted", "weighted_sum"]
MODEL_TYPE: TypeAlias = Literal["gpt2", "rope_gpt2", "stem", "rope_stem", "modded_nanoGPT", "modded_nanoGPT_stem"]
POSITION_EMBEDDING: TypeAlias = Literal["ape", "rope", "rope_legacy"]
DATASET_TYPE: TypeAlias = Literal["efficient_prompt", "efficient_number_prompt", "pretokenized", "pretokenized_number", "pretokenized_number_pos", "efficient_number_prompt_pos", "curriculum_number", "curriculum", "curriculum_number_pos"]

class BetterEnum(Enum):
    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.value == other.value
        elif isinstance(other, str):
            return self.value == other
        return NotImplemented
    
    def __str__(self) -> str:
        return str(self.value)
    
    def __hash__(self) -> int:
        return self.value.__hash__()
    def __repr__(self) -> str:
        return self.value.__repr__()


class DATASET_CURRICULUM_TYPE(BetterEnum):
    STANDARD = "standard"        # Standard dataset without curriculum learning
    CURRICULUM = "curriculum"    # Dataset with curriculum learning based on difficulty levels
    ENDGAME = "endgame"          # Dataset with endgame curriculum learning, focusing on the hardest samples at the end of training
    STANDBY = "standby"          # Dataset with standby curriculum learning, maintaining a small portion of samples even after switching to endgame

class NUMBER_HEAD(BetterEnum):
    LINEAR = "linear"            # Linear layer for number prediction
    MLP = "mlp"                  # Multi-layer perceptron for number prediction
    NONE = "None"                # No number prediction head

class CausalLMOutputWithCrossAttentionsAndNumbers(CausalLMOutputWithCrossAttentions):
    """
    `OrderedDict` that extends the `CausalLMOutputWithCrossAttentions` class
    with an additional field `numbers` to store the generated numbers.
    """
    numbers: Optional[DoubleTensor] = None
    additional_train_losses: Optional[dict[str, float | torch.FloatTensor]] = field(default_factory=dict)

class GenerateDecoderOnlyOutputWithNumbers(GenerateDecoderOnlyOutput):
    numbers: Optional[DoubleTensor] = None
class GenerateEncoderDecoderOutputWithNumbers(GenerateEncoderDecoderOutput):
    numbers: Optional[DoubleTensor] = None
class GenerateBeamDecoderOnlyOutputWithNumbers(GenerateBeamDecoderOnlyOutput):
    numbers: Optional[DoubleTensor]
class GenerateBeamEncoderDecoderOutputWithNumbers(GenerateBeamEncoderDecoderOutput):
    numbers: Optional[DoubleTensor]
GenerateNonBeamOutputWithNumbers = Union[GenerateDecoderOnlyOutputWithNumbers, GenerateEncoderDecoderOutputWithNumbers]
GenerateBeamOutputWithNumbers = Union[GenerateBeamDecoderOnlyOutputWithNumbers, GenerateBeamEncoderDecoderOutputWithNumbers]
GenerateOutputWithNumbers = Union[GenerateNonBeamOutputWithNumbers, GenerateBeamOutputWithNumbers]

@dataclass
class SamplesDict():
    input: list[str]                                        # List of input strings for the samples
    label: list[str]                                        # List of label strings for the samples
    pred: list[str]                                         # List of predicted strings for the samples
    correct: list[bool]                                     # List of booleans indicating whether each sample is correct
    additional_metrics: Optional[dict[str, list[float]]] = None # Additional metrics for the samples

@dataclass
class EvalOutput():
    acc: float                                              # Overall metric accuracy of the generated samples between 0 and 1
    correct_acc: float                                       # Accuracy of entirely correct samples
    loss: float                                             # Overall loss of the generated samples
    num_loss: float                                         # Loss of the generated samples for the number prediction task
    perplexity: float                                       # Perplexity of the generated samples
    text_perplexity: float                                  # Perplexity of the generated samples
    numeric_text_perplexity: float                          # Perplexity of the numeric parts of the generated samples
    per_sample_acc: list[float]                             # List of metric accuracies for each generated sample
    per_sample_correct: list[bool]                          # List of booleans indicating whether each generated sample is correct
    samples_dict: Optional[SamplesDict]                     # Dictionary containing input, label, prediction, correctness, and additional metrics for a small set of samples
    gen_predictions: list[str]                              # List of generated predictions
    num_preds: Optional[list[str]] = None                     # Predictions of the generated samples for the number prediction task
    num_trues: Optional[list[str]] = None                     # True values of the generated samples for the number prediction task
    additional_metrics: dict[str, float] = None   # Additional metrics for the evaluation output
    per_sample_additional_metrics: dict[str, list[float]] = None  # List of additional metrics for each generated sample
    additional_signals: Optional[dict[str, float|ndarray]] = None   # Additional signals for the evaluation output, e.g., for debugging or analysis
    per_difficulty_acc: dict[int, float] = None  # Dictionary mapping difficulty levels to lists of accuracies for each level

@dataclass
class EvalMetrics():
    val_gen_accs: list[dict[str, float]] = field(default_factory=list)
    val_gen_losses: list[dict[str, float]] = field(default_factory=list)
    val_gen_perplexities: list[dict[str, float]] = field(default_factory=list)
    additional_metrics: list[dict[str, dict[str, float] | None]] = field(default_factory=list)


class EvalBatch(TypedDict):
    input_ids: torch.LongTensor
    attention_mask: torch.BoolTensor
    position_ids: torch.LongTensor
    labels: torch.LongTensor
    input_numbers: Optional[torch.DoubleTensor]
    label_numbers: Optional[torch.DoubleTensor]
    cu_seq_lens: Optional[torch.LongTensor]
    max_seq_length: Optional[int]
    orig_answers: Optional[list[str]]
    orig_prompts: Optional[list[str]]
    difficulties: Optional[torch.LongTensor]
    answer_lengths: NotRequired[torch.Tensor]

@dataclass
class TrainMetrics(EvalMetrics):
    num_tokens_per_step: dict[int,int] = field(default_factory=lambda: {0:0})
    lr_updates: list[float] = field(default_factory=list)
    train_losses: dict[str, list[float]] = field(default_factory=dict)
    train_perplexities: list[float] = field(default_factory=list)
    train_token_accs: list[float] = field(default_factory=list)

@dataclass
class TrackedMetrics():
    step: int
    num_tokens: int
    val_gen_loss: dict[str, float]
    val_gen_perplexity: dict[str, float]
    val_gen_acc: dict[str, float]
    additional_metrics: dict[str, dict[str, float] | None]
    train_loss: dict[str, float] = None
    train_perplexity: Optional[float] = None
    train_token_accs: Optional[float] = None
    lr: Optional[float] = None
    duration: Optional[float] = None
    max_difficulty: Optional[dict[str, int]] = None

@dataclass
class TrackedMetricsDataframe():
    step: Series
    num_tokens: Series
    train_loss: Series
    train_perplexity: Series
    train_token_accs: Series
    val_gen_loss: Series
    val_gen_perplexity: Series
    val_gen_acc: Series
    lr: Series
    duration: Series

@dataclass
class Trainer():
    save_dir: Path
    model: GPT2LMHeadModel
    optimizer: torch.optim.Optimizer
    tt: Optional["tqdm_asyncio[int]"]
    step: int
    train_loader: torch.utils.data.DataLoader
    val_loaders: OrderedDict[str, torch.utils.data.DataLoader]
    lr_schedulers: list[torch.optim.lr_scheduler.LRScheduler]
    tokenizer: PreTrainedTokenizerFast
    curriculum_manager: Optional["CurriculumManager"] = None # type: ignore  # noqa: F821

    eval_steps: list = field(default_factory=list)
    interval_durations: list = field(default_factory=list)
    total_duration = 0
    COMPARE_METRIC_INDEX = -1
    WARMUP_LOSS_SCALE: torch.Tensor = None

    train_loss_per_freq: list[ndarray] = field(default_factory=list)
    train_loss_per_freq_agg: list[ndarray] = field(default_factory=list)
    freq_loss_weights_agg: list[ndarray] = field(default_factory=list)
    freq_lr_updates: list[float] = field(default_factory=list)
    additional_train_losses: dict[str, list[float]] = field(default_factory=dict)
    additional_train_losses_agg: dict[str, list[float]] = field(default_factory=dict)

    additional_eval_losses: dict[str, list[float]] = field(default_factory=dict)
    additional_eval_losses_agg: dict[str, list[float]] = field(default_factory=dict)
    eval_loss_per_freq: list[ndarray] = field(default_factory=list)
    eval_loss_per_freq_agg: list[ndarray] = field(default_factory=list)
    eval_use_linear_prob: dict[str, list[float]] = field(default_factory=dict)
    eval_use_linear_prob_agg: dict[str, list[float]] = field(default_factory=dict)
    eval_difficulty_acc: dict[str, dict[int, list[float]]] = field(default_factory=dict)
    eval_difficulty_acc_agg: dict[str, dict[int, list[float]]] = field(default_factory=dict)
    eval_difficulty_ratios: dict[str, dict[int, list[float]]] = field(default_factory=dict)
    advancement_thresholds: dict[str, dict[int, list[float]]] = field(default_factory=dict)
    maximum_difficulty: dict[str, list[int]] = field(default_factory=dict)

    accuracies: dict[int, dict[str, float]] = field(default_factory=dict)
    loss_weights: dict[int, dict[str, float]] = field(default_factory=dict)
    train_ratios: dict[int, dict[str, float]] = field(default_factory=dict)

    difficulties: dict[int, list[int]] = field(default_factory=dict)
    difficulties_agg: dict[str, list[dict[int, float]]] = field(default_factory=dict)
