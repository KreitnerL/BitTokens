from pathlib import Path
from sys import maxsize
from typing import Optional, TypedDict
from warnings import deprecated

import torch
from pandas import DataFrame, read_csv
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


class EfficientPromptBatch(TypedDict):
    idx: int
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    answer_length: int
    orig_prompt: str
    orig_answer: str

class EfficientPromptEvalDataset(Dataset):
    """
    Dataset for evaluating a model on a prompt completion task.
    The input_ids only contain the prompt. The answer is stored under the label key.
    No padding is applied.

    Args:
        dataset_set_path (Path): The path to the dataset containing the prompts and answers
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use
    """
    def __init__(self, dataset_set_path: Optional[Path], tokenizer: PreTrainedTokenizerFast, unique_samples: int = maxsize, shuffle: bool=False, df: Optional[DataFrame] = None, additional_columns: list[str]=[], load_all_columns=False):
        assert (dataset_set_path is None) != (df is None), "Either dataset_set_path or df must be provided, not both."
        if dataset_set_path is not None:
            if load_all_columns:
                self.df: DataFrame = read_csv(dataset_set_path, dtype=str, nrows=unique_samples)
            else:
                self.df: DataFrame = read_csv(dataset_set_path, dtype=str, nrows=unique_samples, usecols=["prompt", "answer"] + additional_columns)
        elif df is not None:
            self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)
    
    def encode(self, prompt: str, answer: str):
        prompt_encoding = self.tokenizer.__call__(prompt, return_tensors="pt", padding="do_not_pad")
        answer_encoding = self.tokenizer.__call__(answer, return_tensors="pt", padding="do_not_pad")
        input_ids: torch.LongTensor = prompt_encoding["input_ids"]
        attention_mask: torch.BoolTensor = prompt_encoding["attention_mask"]
        labels: torch.LongTensor =  answer_encoding["input_ids"]
        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
            "answer_length": len(labels[0]),
            "orig_prompt": prompt,
            "orig_answer": answer
        }

    def __getitem__(self, idx) -> EfficientPromptBatch:
        row = self.df.iloc[idx]
        return {
            "idx": idx,
            **self.encode(f"{self.tokenizer.bos_token}{row['prompt']}{self.tokenizer.init_kwargs['eoq_token']}", f"{row['answer']}{self.tokenizer.eos_token}")
        }
@deprecated("This class is deprecated. Use `PretokenizedTrainDataset` instead. You may use EfficientPromptEvalDataset for evaluation.")
class EfficientPromptTrainDataset(Dataset):
    """
    Dataset for training a model on a prompt completion task.
    The input_ids contain the prompt and the answer. The label is equal to the input_ids with the prompt replaced by -100.
    No padding is applied.

    Args:
        df (DataFrame): The dataframe containing the prompts and answers
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use
    """
    def __init__(self, dataset_set_path: Path, tokenizer: PreTrainedTokenizerFast, unique_samples: int = maxsize, shuffle: bool=False, additional_columns: list[str]=[]):
        self.df: DataFrame = read_csv(dataset_set_path, dtype=str, nrows=unique_samples, usecols=["prompt", "answer"] + additional_columns)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)
    
    def encode(self, prompt: str, answer: str):
        prompt_encoding = self.tokenizer.__call__(prompt, return_tensors="pt", padding="do_not_pad")
        answer_encoding = self.tokenizer.__call__(answer, return_tensors="pt", padding="do_not_pad")
        prompt_input_ids: torch.LongTensor = prompt_encoding["input_ids"]
        prompt_attention_mask: torch.BoolTensor = prompt_encoding["attention_mask"]
        answer_input_ids: torch.LongTensor = answer_encoding["input_ids"]
        answer_attention_mask: torch.BoolTensor = answer_encoding["attention_mask"]
        input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1).squeeze(0)
        attention_mask = torch.cat([prompt_attention_mask, answer_attention_mask], dim=1).squeeze(0)
        labels = torch.cat([torch.full_like(prompt_input_ids, -100), answer_input_ids], dim=1).squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "answer_length": len(answer_input_ids[0]),
            "orig_prompt": prompt,
            "orig_answer": answer
        }

    def __getitem__(self, idx) -> EfficientPromptBatch:
        row = self.df.iloc[idx]
        return {
            "idx": idx,
            **self.encode(f"{self.tokenizer.bos_token}{row['prompt']}{self.tokenizer.init_kwargs['eoq_token']}", f"{row['answer']}{self.tokenizer.eos_token}")
        }
