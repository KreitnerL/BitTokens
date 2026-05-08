from pathlib import Path

if __name__=="__main__":
    import sys
    # Add the root directory of your project to the PYTHONPATH
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import logging
import os
from abc import ABC
from hashlib import sha1
from random import randint
from sys import maxsize
from time import time
from typing import Optional, Sequence
from warnings import deprecated

import numpy as np
from pandas import DataFrame, concat, read_csv
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from dataloader.datasets.efficient_prompt_dataset import EfficientPromptBatch
from dataloader.datasets.memmap_concatenator import MemmapConcatenator

tqdm.pandas()
IGNORE_INDEX = -100
IGNORE_INDEX_UINT16 = 65535
TEXT_CHUNK_SIZE = 10_000_000
DF_CHUNK_SIZE = 100_000


class _PretokenizedDataset(Dataset, ABC):
    """
    Dataset for evaluating a model on a prompt completion task.
    The input_ids only contain the prompt. The answer is stored under the label key.
    No padding is applied.
    """

    def _get_cache_dir(self, *, tokenizer: PreTrainedTokenizerFast, cache_base_path: Path, df: Optional[DataFrame]=None, text_dataset: Optional[str]=None) -> Path:
        """
        Returns the cache directory for the dataset.
        If the dataset is a DataFrame, the cache directory is determined by the hash of the first 5 rows of the DataFrame.
        If the dataset is a text dataset, the cache directory is determined by the hash of the first 1000 characters of the text dataset.
        The base path of the cache directory is determined by the hash of the tokenizer.

        Args:
            tokenizer (PreTrainedTokenizerFast): Tokenizer used to preprocess the dataset.
            cache_base_path (Path): Base path of the cache directory.
            df (Optional[DataFrame]): DataFrame containing the dataset.
            text_dataset (Optional[str]): Text dataset.
        
        Returns:
            Path to the cache directory.
        """
        if df is not None:
            df_bytes = df.to_string().encode()
        elif text_dataset is not None:
            df_bytes = text_dataset.encode()
        else:
            raise ValueError("Either df or text_dataset must be provided")
        h = str(abs(int(sha1(df_bytes).hexdigest(),16))%10**8)
        h_tok = f"{tokenizer.name_or_path.split('/')[-1]}_{abs(int(sha1(str(sorted(tokenizer.vocab.items())).encode()).hexdigest(),16))%10**8}"
        if not getattr(self, "allow_negative", True):
            h_tok = h_tok+"_pos"
        return cache_base_path / h_tok / h
    
    @staticmethod
    def _save_cache(data: np.ndarray, labels: Optional[np.ndarray], cache_path: Path, dataset_path: Path, tokenizer_path: Optional[Path], length: int=0):
        """
        Saves the dataset to the cache directory.

        Args:
            data (np.ndarray): Data to save.
            labels (np.ndarray): Labels to save.
            cache_path (Path): Path to the cache directory.
            dataset_path (Path): Path to the dataset.
            tokenizer_path (Path): Path to the tokenizer.
            length (int): Length of the dataset.
        """
        os.makedirs(cache_path, exist_ok=True)
        if not (cache_path.parent/"info.txt").exists():
            with open(cache_path.parent/"info.txt", "w") as f:
                f.write(f"Tokenizer path: {tokenizer_path}")
        # Save self.data as np.memmap
        np.memmap(cache_path/"data.memmap", mode="w+" if length==0 else "r+", shape=(length + len(data),), dtype=np.uint16)[-len(data):] = data
        if labels is not None:
            np.memmap(cache_path/"labels.memmap", mode="w+" if length==0 else "r+", shape=(length + len(data),), dtype=np.uint16)[-len(labels):] = labels
        
        with open(cache_path/"info.txt", "w") as f:
            f.write(f"Dataset path: {dataset_path}")
            f.write(f"\nnum_tokens: {length+len(data)}\n")

    def _preprocess(self, *, tokenizer: PreTrainedTokenizerFast, cache_path: Path, dataset_path: Path, tokenizer_path: Optional[Path], df: Optional[DataFrame]=None, text_dataset: Optional[str]=None):
        """
        Preprocesses the dataset.
        If the dataset is a DataFrame, the prompt and answer columns are tokenized and concatenated using the folowing structure: [bos_token_id, prompt, eoq_token_id, answer, eos_token_id].
        The tokenized text is concatenated flattened. The result consists of two identical length 1D tensors: data and labels.
        After preprocessing, the dataset is saved to the cache directory.

        Args:
            tokenizer (PreTrainedTokenizerFast): Tokenizer used to preprocess the dataset.
            cache_path (Path): Path to the cache directory.
            dataset_path (Path): Path to the dataset.
            tokenizer_path (Path): Path to the tokenizer.
            df (Optional[DataFrame]): DataFrame containing the dataset.
            text_dataset (str): Text dataset.
        """
        logging.info(f"Preprocessing {dataset_path}...")
        s = time()
        bos_token: str = tokenizer.bos_token
        eoq_token = tokenizer.init_kwargs["eoq_token"]
        eos_token = tokenizer.eos_token
        length=0
        if df is not None:
            for i in tqdm(range(0, len(df), DF_CHUNK_SIZE), desc="Tokenizing dataframe"):
                chunk = df.iloc[i:i+DF_CHUNK_SIZE]
                text = "".join((bos_token + chunk["prompt"].astype(str) + eoq_token + chunk["answer"].astype(str) + eos_token).astype(str).values)
                text_tokens: dict[str, np.ndarray] = tokenizer.__call__(text, padding="do_not_pad", return_tensors="np", return_attention_mask=False)
                data = text_tokens["input_ids"].astype(np.uint16).reshape(-1)
                bos_mask = data==self.bos_token_id
                eoq_mask = data==self.eoq_token_id
                prompt_mask = (np.cumsum(bos_mask) > np.cumsum(eoq_mask)) | eoq_mask
                labels = np.copy(data)
                labels[prompt_mask] = IGNORE_INDEX_UINT16

                _PretokenizedDataset._save_cache(
                    data,
                    labels,
                    cache_path,
                    dataset_path,
                    tokenizer_path,
                    length
                )
                length += len(data.reshape(-1))
        elif text_dataset is not None:
            # Split the text_dataset into chunks of 100_000 chars
            for i in tqdm(range(0, len(text_dataset), TEXT_CHUNK_SIZE), desc="Tokenizing text"):
                text_dataset_i = text_dataset[i:i+TEXT_CHUNK_SIZE]
                data: np.ndarray = tokenizer.__call__(text_dataset_i, padding="do_not_pad", return_tensors="np", return_attention_mask=False)["input_ids"].astype(np.uint16).reshape(-1)

                _PretokenizedDataset._save_cache(
                    data,
                    None,
                    cache_path,
                    dataset_path,
                    tokenizer_path,
                    length=length
                )
                length += len(data)
        else:
            raise ValueError("Either df or text_dataset must be provided")
        with open(cache_path.parent/"info.txt", "a") as f:
            f.write(f"\n{cache_path.name}")
            f.write(f"\n\tDataset path: {dataset_path}")
            f.write(f"\n\tnum_tokens: {length+len(data)}\n")

        logging.info(f"Finished preprocessing data in {time()-s:.2f}s")

    def _load_cache(self, cache_path: Path, num_tok: int=-1) -> int:
        """
        Loads the dataset from the cache directory. The data and labels tensors are concatenated with the already loaded datasets.

        Args:
            cache_path (Path): Path to the cache directory.
            num_tok (int): Number of tokens to load from the dataset. If num_tok is negative, the entire dataset is loaded.
        """
        logging.info(f"Loading cached dataset from {cache_path}")
        
        try:
            if num_tok < 0:
                num_tok=maxsize
            num_tok = min(num_tok, len(np.memmap(cache_path/"data.memmap", mode="r", dtype=np.uint16)))
            # num_tok = num_tok - num_tok % context_length
            data =np.memmap(cache_path/"data.memmap", mode="r", dtype=np.uint16, shape=(num_tok,))
            if (cache_path/"labels.memmap").exists():
                labels = np.memmap(cache_path/"labels.memmap", mode="r", dtype=np.uint16, shape=(num_tok,))
            else:
                labels = data
        except ValueError as e:
            logging.error(f"Could not load numbers from {cache_path}. The cache is likely smaller than the requested number of tokens ({num_tok}). If this error persists, delete the cache directory and try again.")
            raise e
        self.data_l.append(data)
        self.labels_l.append(labels)
        return num_tok

    def _init_vars(self, context_length, tokenizer: PreTrainedTokenizerFast):
        self.context_length = context_length
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.eoq_token_id = tokenizer.init_kwargs["eoq_token_id"]

        self.data_l = list()
        self.labels_l = list()

    def __init__(
            self,
            dataset_paths: Sequence[Path],
            tokenizer: PreTrainedTokenizerFast,
            cache_base_path: Path,
            cached_paths: list[Path|None]=[],
            num_tokens: Optional[Sequence[int]]=None,
            context_length: int = 1024,
            unique_samples: int = maxsize,
            shuffle: bool=False,
            tokenizer_path: Optional[Path]=None,
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
            shuffle (bool): Whether to shuffle the samples.
            tokenizer_path (Optional[Path]): Path to the tokenizer directory. If provided, the tokenizer is loaded from this path.
        """
        super().__init__()
        assert cache_base_path is not None, "cache_base_path must be provided"
        assert len(dataset_paths)>0, "At least one dataset set path must be provided"
        self._init_vars(context_length, tokenizer)
        if cached_paths == []:
            cached_paths = [None]*len(dataset_paths)
        for i, path in enumerate(dataset_paths):
            if cached_paths[i] is None:
                text_dataset = df = None
                df_head = text_head_str = None
                if path.suffix == ".csv":
                    df_head = read_csv(path, dtype=str, nrows=5)
                elif path.suffixes == [".csv", ".gz"]:
                    df_head = read_csv(path, dtype=str, nrows=5, compression="gzip")
                elif path.suffix == ".txt":
                    with open(path, "r") as f:
                        text_head_b: bytes = os.pread(f.fileno(), 1000, 0)
                        text_head_str = text_head_b.decode("ISO-8859-1")
                else:
                    raise ValueError(f"Unsupported file type {path.suffix}")
                cache_path = self._get_cache_dir(tokenizer=tokenizer, cache_base_path=cache_base_path, df=df_head, text_dataset=text_head_str)
                if not cache_path.exists():
                    logging.info(f"No cached dataset found at {cache_path}")
                    logging.info(f"Loading full file for preprocessing: {path}")
                    logging.info(f"Loading columns {['prompt', 'answer'] + additional_columns} from csv file")
                    if df_head is not None:
                        df = concat([chunk for chunk in tqdm(read_csv(path, dtype=str, chunksize=DF_CHUNK_SIZE, usecols=lambda col: col in (["prompt", "answer"]+ additional_columns)), desc='Loading data', unit='100k rows')], ignore_index=True)
                        df = df.sample(n=min(unique_samples, len(df)), random_state=42) if shuffle else df[:unique_samples]
                    else:
                        text_dataset = ""
                        with tqdm(total=os.path.getsize(path), desc="Loading file...") as pbar:
                            with open(path, "rt") as f:
                                for line in f:
                                    pbar.update(len(line))
                                    text_dataset += line
                    self._preprocess(tokenizer=tokenizer, cache_path=cache_path, dataset_path=path, tokenizer_path=tokenizer_path, df=df, text_dataset=text_dataset)
                cached_paths[i] = cache_path
        if num_tokens is None:
            num_tokens = [-1]*len(cached_paths)
        for cache_path, num_tok in zip(cached_paths, num_tokens):
            assert cache_path is not None, "Cached path must be provided"
            self._load_cache(cache_path, num_tok)
        self.data = MemmapConcatenator(self.data_l, context_length)
        self.labels = MemmapConcatenator(self.labels_l, context_length)
        del self.data_l
        del self.labels_l
        logging.info(f"Finished loading dataset(s). Total number of tokens: {len(self.data)}")

class PretokenizedTrainDataset(_PretokenizedDataset):
    def __init__(
            self,
            dataset_paths: Sequence[Path],
            tokenizer: PreTrainedTokenizerFast,
            cache_base_path: Path,
            *,
            cached_paths: list[Path|None]=[],
            num_tokens: Optional[Sequence[int]]=None,
            context_length: int = 1024,
            unique_samples: int = maxsize,
            shuffle: bool=False,
            tokenizer_path: Optional[Path]=None,
            rand_offset=True
        ):
        super().__init__(dataset_paths, tokenizer, cache_base_path, cached_paths, num_tokens, context_length, unique_samples, shuffle, tokenizer_path)
        self.rand_offset = rand_offset

    def __len__(self):
        # Returns the number of context windows in the dataset
        return len(self.data) // self.context_length -  (1 if self.rand_offset else 0)
    
    def __getitem__(self, context_idx) -> EfficientPromptBatch:
        offset = randint(0, self.context_length) if self.rand_offset else 0
        idx = context_idx*self.context_length+offset
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
            "sample_idx": self.data.get_dataset_idx(idx)
        }

@deprecated("Use PretokenizedTrainDataset instead")
class PretokenizedEvalDataset(_PretokenizedDataset):
    def __init__(
            self,
            dataset_paths: Sequence[Path],
            tokenizer: PreTrainedTokenizerFast,
            cache_base_path: Path,
            *,
            cached_paths: list[Path|None]=[],
            num_tokens: Optional[Sequence[int]]=None,
            context_length: int = 1024,
            unique_samples: int = maxsize,
            shuffle: bool=False,
            tokenizer_path: Optional[Path]=None,
            prompt_len=0,
            answer_len=1
        ):
        """
        Args:
            dataset_paths (Sequence[Path]): Paths to the dataset sets. Each dataset set must be a csv file with columns "prompt" and "answer" or a txt file with text.
            tokenizer (PreTrainedTokenizerFast): Tokenizer to use.
            cache_base_path (Path): Path to the cache directory.
            cached_paths (Optional[Sequence[Path]]): Paths to the cached dataset directories. If provided, the dataset sets are loaded from these paths.
            num_tokens (Sequence[int]): Number of tokens to load from each dataset set. By default, the entire dataset set is loaded.
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

    def __getitem__(self, idx) -> EfficientPromptBatch:
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
            "position_ids": position_ids
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
    dataset = PretokenizedTrainDataset(
        dataset_paths,
        tokenizer=tokenizer,
        cache_base_path=cache_base_path,
        context_length=256,
        tokenizer_path=tokenizer_dir
    )
    print("Done.")
