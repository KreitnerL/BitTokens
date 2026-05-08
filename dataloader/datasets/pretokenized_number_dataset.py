from pathlib import Path

if __name__=="__main__":
    import sys
    # Add the root directory of your project to the PYTHONPATH
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import logging
from abc import ABC
from math import inf
from random import randint
from sys import maxsize
from time import time
from typing import Optional, Sequence, override
from warnings import deprecated

import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from dataloader.datasets.efficient_number_prompt_dataset import (
    EfficientNumberPromptBatch,
)
from dataloader.datasets.memmap_concatenator import MemmapConcatenator
from dataloader.datasets.pretokenized_dataset import (
    DF_CHUNK_SIZE,
    IGNORE_INDEX,
    IGNORE_INDEX_UINT16,
    TEXT_CHUNK_SIZE,
    _PretokenizedDataset,
)
from utils.util_funcs import get_num_token_mask_numpy, replace_num_from_text

tqdm.pandas()

class _PretokenizedNumberDataset(_PretokenizedDataset, ABC):
    """
    Dataset for evaluating a model on a prompt completion task.
    The input_ids only contain the prompt. The answer is stored under the label key.
    No padding is applied.
    """
    def __init__(
        self,
        dataset_paths: Sequence[Path],
        tokenizer: PreTrainedTokenizerFast,
        cache_base_path: Path,
        cached_paths: list[Path | None]=[],
        num_tokens: Optional[Sequence[int]]=None,
        context_length: int = 1024,
        unique_samples: int = maxsize,
        shuffle: bool=False,
        tokenizer_path: Optional[Path]=None,
        allow_negative: bool=True,
        additional_columns: list[str]=[]
    ):
        """
        Args:
            dataset_paths (Sequence[Path]): Paths to the dataset sets. Each dataset set must be a csv file with columns "prompt" and "answer" or a txt file with text.
            tokenizer (PreTrainedTokenizerFast): Tokenizer to use.
            cache_base_path (Path): Path to the cache directory.
            cached_paths (Optional[Sequence[Path]]): Paths to the cached dataset directories. If provided, the dataset sets are loaded from these paths.
            num_tokens (Optional[Sequence[int]]): Number of tokens to load from each dataset set. By default, the entire dataset set is loaded.
            context_length (int): The length of the context.
            unique_samples (int): The number of unique samples to use.
            shuffle (bool): Whether to shuffle the samples
            tokenizer_path (Path): Path to the tokenizer directory. If provided, the tokenizer is loaded from this path.
        """
        self.num_token_bins = tokenizer.init_kwargs.get("num_token_bins", [-inf, inf])
        self.allow_negative = allow_negative
        super().__init__(
            dataset_paths,
            tokenizer,
            cache_base_path,
            cached_paths=cached_paths,
            num_tokens=num_tokens,
            context_length=context_length,
            unique_samples=unique_samples,
            shuffle=shuffle,
            tokenizer_path=tokenizer_path,
            additional_columns=additional_columns
        )
        self.numbers = MemmapConcatenator(self.numbers_l, context_length)
        del self.numbers_l
    
    @staticmethod
    @override
    def _save_cache(data: np.ndarray, labels: Optional[np.ndarray], numbers: np.ndarray, cache_path: Path, dataset_path: Path, tokenizer_path: Path, length: int=0):
        _PretokenizedDataset._save_cache(data, labels, cache_path, dataset_path, tokenizer_path, length=length)
        np.memmap(cache_path/"numbers.memmap", mode="w+" if length==0 else "r+", shape=(length + len(numbers),), dtype=np.float64)[-len(numbers):] = numbers

    @override
    def _preprocess(self, *, tokenizer: PreTrainedTokenizerFast, cache_path: Path, dataset_path: Path, tokenizer_path: Path, df: Optional[DataFrame]=None, text_dataset: Optional[str]=None):
        logging.info("Preprocessing data...")
        s = time()
        num_tokens = tokenizer.init_kwargs["num_token"]
        if not isinstance(num_tokens, list):
            num_tokens = [num_tokens]
        num_token_ids = [tokenizer.convert_tokens_to_ids(num_token) for num_token in num_tokens]
        bos_token: str = tokenizer.bos_token
        eoq_token = tokenizer.init_kwargs["eoq_token"]
        eos_token = tokenizer.eos_token
        length=0
        if df is not None:
            for i in tqdm(range(0, len(df), DF_CHUNK_SIZE), desc="Tokenizing dataframe"):
                chunk = df.iloc[i:i+DF_CHUNK_SIZE]
                full_text = "".join((bos_token + chunk["prompt"].astype(str) + eoq_token + chunk["answer"].astype(str) + eos_token).astype(str).values)
                text, numbers = replace_num_from_text(full_text, num_tokens, self.num_token_bins, allow_negative=self.allow_negative)
                text_tokens: dict[str, np.ndarray] = tokenizer.__call__(text, padding="do_not_pad", return_tensors="np", return_attention_mask=False)
                data = text_tokens["input_ids"].astype(np.uint16).reshape(-1)
                numbers_full = np.zeros_like(data, dtype=np.float64)
                numbers_full[get_num_token_mask_numpy(data, num_token_ids)] = numbers
                bos_mask = data==self.bos_token_id
                eoq_mask = data==self.eoq_token_id
                prompt_mask = (np.cumsum(bos_mask) > np.cumsum(eoq_mask)) | eoq_mask
                labels = np.copy(data)
                labels[prompt_mask] = IGNORE_INDEX_UINT16
                
                _PretokenizedNumberDataset._save_cache(
                    data,
                    labels,
                    numbers_full,
                    cache_path,
                    dataset_path,
                    tokenizer_path,
                    length=length
                )
                length += len(data)
        elif text_dataset is not None:
            for i in tqdm(range(0, len(text_dataset), TEXT_CHUNK_SIZE), desc="Tokenizing text"):
                text_dataset_i = text_dataset[i:i+TEXT_CHUNK_SIZE]
                text, text_numbers = replace_num_from_text(text_dataset_i, num_tokens, self.num_token_bins, allow_negative=self.allow_negative)
                text_tokens: dict[str, np.ndarray] = tokenizer.__call__(text, padding="do_not_pad", return_tensors="np", return_attention_mask=False)

                numbers = np.zeros_like(text_tokens["input_ids"], dtype=np.float64)
                numbers[get_num_token_mask_numpy(text_tokens["input_ids"], num_token_ids)] = text_numbers

                data = text_tokens["input_ids"].astype(np.uint16).reshape(-1)
                _PretokenizedNumberDataset._save_cache(
                    data,
                    None,
                    numbers.reshape(-1),
                    cache_path,
                    dataset_path,
                    tokenizer_path,
                    length=length
                )
                length += len(data)
        with open(cache_path.parent/"info.txt", "a") as f:
            f.write(f"\n{cache_path.name}")
            f.write(f"\n\tDataset path: {dataset_path}")
            f.write(f"\n\tnum_tokens: {length+len(data)}\n")
        
        logging.info(f"Finished preprocessing data in {time()-s:.2f}s")

    @override
    def _init_vars(self, context_length, tokenizer):
        super()._init_vars(context_length, tokenizer)
        self.numbers_l = list()

    @override
    def _load_cache(self, cache_path: Path, num_tok: int=-1) -> int:
        num_tok = super()._load_cache(cache_path, num_tok)
        try:
            self.numbers_l.append(np.memmap(cache_path/"numbers.memmap", mode="r", dtype=np.float64, shape=(num_tok,)))
        except ValueError as e:
            logging.error(f"Could not load numbers from {cache_path/'numbers.memmap'}. The cache is likely corrupted. Please delete the cache directory and try again.")
            raise e

        return num_tok

class PretokenizedNumberTrainDataset(_PretokenizedNumberDataset):
    def __init__(
        self,
        dataset_paths: Sequence[Path],
        tokenizer: PreTrainedTokenizerFast,
        cache_base_path: Path,
        cached_paths: list[Path | None]=[],
        num_tokens: Optional[Sequence[int]]=None,
        context_length: int = 1024,
        unique_samples: int = maxsize,
        shuffle: bool=False,
        tokenizer_path: Optional[Path]=None,
        rand_offset=True,
        allow_negative: bool=True
    ):
        super().__init__(dataset_paths, tokenizer, cache_base_path, cached_paths=cached_paths, num_tokens=num_tokens, context_length=context_length, unique_samples=unique_samples, shuffle=shuffle, tokenizer_path=tokenizer_path, allow_negative=allow_negative)
        self.rand_offset = rand_offset
        
    def __len__(self):
        return len(self.data) // self.context_length - (1 if self.rand_offset else 0)
    
    def __getitem__(self, context_idx) -> EfficientNumberPromptBatch:
        offset = randint(0, self.context_length) if self.rand_offset else 0
        idx = context_idx*self.context_length + offset
        data = self.data[idx:idx+self.context_length].astype(np.int64)
        pad_mask: np.ndarray = (data != self.pad_token_id)
        eos_indices = (data == self.eos_token_id).nonzero()[0]
        reps = np.concatenate([eos_indices[:1]+1, eos_indices[1:] - eos_indices[:-1], self.context_length-eos_indices[-1:]-1])
        position_ids = np.arange(self.context_length)
        if reps.size > 0:
            position_ids -= np.repeat(np.concatenate([np.array([0]), eos_indices+1], axis=0), reps)

        labels = self.labels[idx:idx+self.context_length].astype(np.int64)
        labels[labels == IGNORE_INDEX_UINT16] = IGNORE_INDEX
        return {
            "idx": idx,
            "input_ids": np.array(data),
            "attention_mask": pad_mask.astype(np.int64),
            "labels": np.array(labels),
            "position_ids": position_ids,
            "input_numbers": np.array(self.numbers[idx:idx+self.context_length]),
            "sample_idx": self.data.get_dataset_idx(idx)
        }

@deprecated("Use PretokenizedNumberTrainDataset instead")
class PretokenizedNumberEvalDataset(_PretokenizedNumberDataset):
    def __init__(
            self,
            dataset_paths: Sequence[Path],
            tokenizer: PreTrainedTokenizerFast,
            cache_base_path: Path,
            cached_paths: list[Path | None]=[],
            num_tokens: Optional[Sequence[int]]=None,
            context_length: int = 1024,
            unique_samples: int = maxsize,
            shuffle: bool=False,
            tokenizer_path: Optional[Path]=None,
            prompt_len=0,
            answer_len = 1
        ):
        """
        Args:
            dataset_paths (Sequence[Path]): Paths to the dataset sets. Each dataset set must be a csv file with columns "prompt" and "answer" or a txt file with text.
            tokenizer (PreTrainedTokenizerFast): Tokenizer to use.
            cache_base_path (Path): Path to the cache directory.
            num_tokens (Optional[Sequence[int]]): Number of tokens to load from each dataset set. By default, the entire dataset set is loaded.
            context_length (int): The length of the context.
            unique_samples (int): The number of unique samples to use.
            shuffle (bool): Whether to shuffle the samples
            tokenizer_path (Path): Path to the tokenizer directory. If provided, the tokenizer is loaded from this path.
            prompt_len (int): The length of the prompt. If 0, the prompt length is set to context_length//4.
            answer_len (int): The length of the answer. If 0, the answer length is set to context_length//4.
        """
        self.prompt_len = prompt_len or context_length//4
        self.answer_len = answer_len
        super().__init__(dataset_paths, tokenizer, cache_base_path, cached_paths, num_tokens, context_length, unique_samples, shuffle, tokenizer_path=tokenizer_path)

    def __len__(self):
        return len(self.data) - self.prompt_len
    
    def __getitem__(self, idx) -> EfficientNumberPromptBatch:
        data = self.data[idx:idx+self.prompt_len].astype(np.int64)
        attention_mask: np.ndarray = (data != self.pad_token_id).astype(np.int64)
        position_ids = attention_mask.cumsum(-1)-1
        position_ids[attention_mask == 0] = 1

        labels = self.labels[idx+self.prompt_len:idx+self.prompt_len+self.answer_len].astype(np.int64)
        labels[labels == IGNORE_INDEX_UINT16] = IGNORE_INDEX
        return {
            "idx": idx,
            "input_ids": np.array(data),
            "attention_mask": attention_mask,
            "labels": np.array(labels),
            "position_ids": position_ids,
            "input_numbers": np.array(self.numbers[idx:idx+self.prompt_len]),
            "label_numbers": np.array(self.numbers[idx+self.prompt_len:idx+self.prompt_len+self.answer_len])
        }
    
if __name__=="__main__":
    from argparse import ArgumentParser
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = ArgumentParser()
    parser.add_argument("--dataset_paths", nargs="+", type=str)
    parser.add_argument("--tokenizer_dir", type=str)
    parser.add_argument("--cache_base_path", type=str)
    args = parser.parse_args()

    dataset_paths = [Path(p) for p in args.dataset_paths]
    tokenizer_dir = Path(args.tokenizer_dir)
    cache_base_path = Path(args.cache_base_path)

    tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    tokenizer.padding_side = "left"
    dataset = PretokenizedNumberTrainDataset(
        dataset_paths,
        tokenizer=tokenizer,
        cache_base_path=cache_base_path,
        context_length=256,
        tokenizer_path=tokenizer_dir
    )
    print("Done.")
