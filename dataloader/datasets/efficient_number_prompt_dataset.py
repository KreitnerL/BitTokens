from math import inf
from pathlib import Path
from sys import maxsize
from typing import Optional
from warnings import deprecated

import torch
from pandas import DataFrame
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from dataloader.datasets.efficient_prompt_dataset import (
    EfficientPromptBatch,
    EfficientPromptEvalDataset,
    EfficientPromptTrainDataset,
)
from utils.util_funcs import replace_num_from_text


class EfficientNumberPromptBatch(EfficientPromptBatch):
    input_numbers: Tensor
    label_numbers: Tensor

class EfficientNumberPromptEvalDataset(EfficientPromptEvalDataset):
    def __init__(self, dataset_path: Optional[Path], tokenizer: PreTrainedTokenizerFast, unique_samples: int = maxsize, shuffle: bool=False, allow_negative: bool=True, df: Optional[DataFrame] = None, additional_columns: list[str]=[], load_all_columns=False):
        super().__init__(dataset_path, tokenizer, unique_samples, shuffle, df=df, additional_columns=additional_columns, load_all_columns=load_all_columns)
        assert tokenizer.init_kwargs["model_specific_special_tokens"]["num_token"] is not None, "Tokenizer must have a num_token"
        self.num_token_bins = tokenizer.init_kwargs["model_specific_special_tokens"].get("num_token_bins", [-inf, inf])
        self.num_tokens = tokenizer.init_kwargs["model_specific_special_tokens"]["num_token"]
        if not isinstance(self.num_tokens, list):
            self.num_tokens = [self.num_tokens]
        self.num_prefix_token = tokenizer.init_kwargs["model_specific_special_tokens"].get("num_prefix_token", "")
        self.allow_negative = allow_negative


    def __getitem__(self, idx) -> EfficientNumberPromptBatch:
        row = self.df.iloc[idx]
        input_str,input_numbers = replace_num_from_text(
            f"{self.tokenizer.bos_token}{row['prompt']}{self.tokenizer.init_kwargs["model_specific_special_tokens"]['eoq_token']}",
            self.num_tokens,
            self.num_token_bins,
            self.num_prefix_token,
            allow_negative=self.allow_negative
        )
        label_str, label_numbers = replace_num_from_text(
            f"{row['answer']}{self.tokenizer.eos_token}",
            self.num_tokens,
            self.num_token_bins,
            self.num_prefix_token,
            allow_negative=self.allow_negative
        )

        ret = {
            "idx": idx,
            **self.encode(input_str, label_str),
            "input_numbers": input_numbers,
            "orig_prompt": row['prompt'],
            "orig_answer": row['answer']
        }
        if self.allow_negative:
            ret["label_numbers"] = label_numbers
        return ret

@deprecated("This class is deprecated. Use `PretokenizedNumberTrainDataset` instead. You may use EfficientNumberPromptEvalDataset for evaluation.")
class EfficientNumberPromptTrainDataset(EfficientPromptTrainDataset):
    """
    Dataset for training a model on a prompt completion task.
    The input_ids contain the prompt and the answer. The label is equal to the input_ids with the prompt replaced by -100.
    No padding is applied.

    Args:
        df (DataFrame): The dataframe containing the prompts and answers
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use
    """
    def __init__(self, dataset_set_path: Path, tokenizer: PreTrainedTokenizerFast, unique_samples: int = maxsize, shuffle: bool=False, allow_negative: bool=True):
        super().__init__(dataset_set_path, tokenizer, unique_samples, shuffle)
        assert tokenizer.init_kwargs["model_specific_special_tokens"]["num_token"] is not None, "Tokenizer must have a num_token"
        self.num_token_bins = tokenizer.init_kwargs.get("num_token_bins", [-inf, inf])
        self.num_tokens = tokenizer.init_kwargs["model_specific_special_tokens"]["num_token"]
        if not isinstance(self.num_tokens, list):
            self.num_tokens = [self.num_tokens]
        self.num_prefix_token = tokenizer.init_kwargs.get("num_prefix_token", "")
        self.allow_negative = allow_negative
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_str,input_numbers = replace_num_from_text(
            f"{self.tokenizer.bos_token}{row['prompt']}{self.tokenizer.init_kwargs["model_specific_special_tokens"]['eoq_token']}",
            self.num_prefix_token,
            self.num_token_bins,
            self.num_prefix_token,
            allow_negative=self.allow_negative
        )
        label_str, label_numbers = replace_num_from_text(
            f"{row['answer']}{self.tokenizer.eos_token}",
            self.num_prefix_token,
            self.num_token_bins,
            self.num_prefix_token,
            allow_negative=self.allow_negative
        )
        return {
            "idx": idx,
            **self.encode(input_str, label_str),
            "input_numbers": torch.cat((input_numbers, label_numbers)),
            "orig_prompt": row['prompt'],
            "orig_answer": row['answer']
        }
