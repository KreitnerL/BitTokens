import logging
from abc import ABC
from pathlib import Path
from random import randint
from sys import maxsize
from typing import Any, Optional, Sequence

import numpy as np
from pandas import DataFrame
from transformers import PreTrainedTokenizerFast

from dataloader.datasets.curriculum_abc import (
    CurriculumDatasetABC,
    CurriculumPretokenizedMixin,
)
from dataloader.datasets.efficient_prompt_dataset import (
    EfficientPromptBatch,
    EfficientPromptEvalDataset,
)
from dataloader.datasets.pretokenized_dataset import (
    IGNORE_INDEX,
    IGNORE_INDEX_UINT16,
    _PretokenizedDataset,
)
from utils.enums import DATASET_CURRICULUM_TYPE


class CurriculumPretokenizedDatasetABC(CurriculumPretokenizedMixin, CurriculumDatasetABC, _PretokenizedDataset, ABC):
    """
    Dataset for curriculum learning with difficulty levels.
    Extends _PretokenizedDataset to handle difficulty-based sampling.
    """
    
    def __init__(
        self,
        dataset_paths: Sequence[Path],
        dataset_curriculum_types: list[DATASET_CURRICULUM_TYPE],
        tokenizer: PreTrainedTokenizerFast,
        cache_base_path: Path,
        cached_paths: list[Path | None] = [],
        num_tokens: Optional[Sequence[int]] = None,
        context_length: int = 1024,
        unique_samples: int = maxsize,
        shuffle: bool = False,
        tokenizer_path: Optional[Path] = None,
        difficulty_column: str = "difficulty"
    ):
        """
        Args:
            dataset_paths (Sequence[Path]): Paths to the dataset sets. Each dataset set must be a csv file with columns "prompt", "answer", and optionally "difficulty".
            dataset_curriculum_types (list[DATASET_CURRICULUM_TYPE]): Curriculum type for each dataset set.
            tokenizer (PreTrainedTokenizerFast): Tokenizer to use.
            cache_base_path (Path): Path to the cache directory.
            cached_paths (Optional[Sequence[Path]]): Paths to the cached dataset directories. If provided, the dataset sets are loaded from these paths.
            num_tokens (Optional[Sequence[int]]): Number of tokens to load from each dataset set. By default, the entire dataset set is loaded.
            context_length (int): The length of the context.
            unique_samples (int): The number of unique samples to use.
            shuffle (bool): Whether to shuffle the samples
            tokenizer_path (Path): Path to the tokenizer directory. If provided, the tokenizer is loaded from this path.
            difficulty_column (str): Column name for difficulty levels (default: "difficulty").
        """
        self.dataset_paths = dataset_paths
        # Initialize curriculum attributes
        self._init_curriculum_attributes(dataset_paths, dataset_curriculum_types, difficulty_column)
        
        # Initialize parent class
        super().__init__(
            dataset_paths, tokenizer, cache_base_path, cached_paths=cached_paths,
            num_tokens=num_tokens, context_length=context_length, unique_samples=unique_samples,
            shuffle=shuffle, tokenizer_path=tokenizer_path, additional_columns=[difficulty_column]
        )
        
        # Build reverse mapping after all data is loaded
        self._build_reverse_difficulty_mapping()
    
    def _preprocess(self, *, tokenizer: PreTrainedTokenizerFast, cache_path: Path, dataset_path: Path, tokenizer_path: Optional[Path], df: Optional[DataFrame] = None, text_dataset: Optional[str] = None):
        """Override preprocessing to sort by difficulty and store difficulty indices."""
        if df is None:
            # For text datasets, fall back to standard behavior
            logging.info("Text dataset detected, falling back to standard preprocessing")
            return super()._preprocess(
                tokenizer=tokenizer, cache_path=cache_path, dataset_path=dataset_path, 
                tokenizer_path=tokenizer_path, df=df, text_dataset=text_dataset
            )
        
        # Check for difficulty column - if not present, fall back to standard behavior
        idx = self.dataset_paths.index(dataset_path)
        if self.difficulty_column not in df.columns or self.dataset_curriculum_types[idx] != DATASET_CURRICULUM_TYPE.CURRICULUM:
            if self.dataset_curriculum_types[idx] != DATASET_CURRICULUM_TYPE.CURRICULUM:
                logging.info(f"Dataset curriculum type is {self.dataset_curriculum_types[idx]}, falling back to standard preprocessing")
            else:
                logging.info(f"No '{self.difficulty_column}' column found, falling back to standard preprocessing")
            return super()._preprocess(
                tokenizer=tokenizer, cache_path=cache_path, dataset_path=dataset_path, 
                tokenizer_path=tokenizer_path, df=df, text_dataset=text_dataset
            )
        
        # Define the callback for processing chunks
        def save_cache_callback(chunk, tokenizer, cache_path, dataset_path, tokenizer_path, total_tokens):
            # Process chunk
            bos_token: str = tokenizer.bos_token
            eoq_token = tokenizer.init_kwargs["eoq_token"]
            eos_token = tokenizer.eos_token
            
            text = "".join((bos_token + chunk["prompt"].astype(str) + eoq_token + chunk["answer"].astype(str) + eos_token).astype(str).values)
            text_tokens: dict[str, np.ndarray] = tokenizer.__call__(text, padding="do_not_pad", return_tensors="np", return_attention_mask=False)
            data = text_tokens["input_ids"].astype(np.uint16).reshape(-1)
            
            # Create labels (mask prompt tokens)
            bos_mask = data == self.bos_token_id
            eoq_mask = data == self.eoq_token_id
            prompt_mask = (np.cumsum(bos_mask) > np.cumsum(eoq_mask)) | eoq_mask
            labels = np.copy(data)
            labels[prompt_mask] = IGNORE_INDEX_UINT16
            
            # Save cache for this chunk
            _PretokenizedDataset._save_cache(
                data, labels, cache_path, dataset_path, tokenizer_path, length=total_tokens
            )
            
            return data, labels, total_tokens + len(data)
        
        # Use the curriculum preprocessing
        self._preprocess_curriculum_dataframe(
            df, tokenizer, cache_path, dataset_path, tokenizer_path, save_cache_callback
        )
    
    def _load_cache(self, cache_path: Path, num_tok: int = -1) -> int:
        """Override cache loading to also load difficulty indices."""
        num_tok = super()._load_cache(cache_path, num_tok)
        
        # Load curriculum-specific cache
        self._load_curriculum_cache(cache_path)
        
        return num_tok


class CurriculumTrainDataset(CurriculumPretokenizedDatasetABC):
    """Training dataset with curriculum learning support."""
    
    def __init__(self, *args, rand_offset=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.rand_offset = rand_offset
    
    def __len__(self):
        return len(self.data) // self.context_length - (1 if self.rand_offset else 0)
    
    def __getitem__(self, context_idx) -> EfficientPromptBatch:
        """Get item by context index with difficulty information."""
        offset = randint(0, self.context_length) if self.rand_offset else 0
        idx = context_idx * self.context_length + offset
        data = self.data[idx:idx + self.context_length].astype(np.int64)
        pad_mask: np.ndarray = (data != self.pad_token_id)
        eos_indices = (data == self.eos_token_id).nonzero()[0]
        reps = np.concatenate([eos_indices[:1] + 1, eos_indices[1:] - eos_indices[:-1], self.context_length - eos_indices[-1:] - 1])
        position_ids = np.arange(self.context_length)
        if reps.size > 0:
            position_ids -= np.repeat(np.concatenate([np.array([0]), eos_indices + 1], axis=0), reps)

        labels = self.labels[idx:idx + self.context_length].astype(np.int64)
        labels[labels == IGNORE_INDEX_UINT16] = IGNORE_INDEX
        
        result = {
            "idx": idx,
            "input_ids": np.array(data),
            "attention_mask": pad_mask.astype(np.int64),
            "labels": np.array(labels),
            "position_ids": position_ids,
            "sample_idx": self.data.get_dataset_idx(idx),
            "difficulty": self.get_difficulty_for_idx(idx)
        }
        
        return result
    
class EfficientCurriculumPromptBatch(EfficientPromptBatch):
    difficulty: int

class CurriculumEvalDataset(CurriculumDatasetABC, EfficientPromptEvalDataset):
    """Evaluation dataset with curriculum learning support."""
    
    def __init__(self, tokenizer: PreTrainedTokenizerFast, dataset_path: Optional[Path]=None, unique_samples: int = maxsize, shuffle: bool = False, allow_negative: bool = True, difficulty_column: str = "difficulty", dataset_curriculum_type: DATASET_CURRICULUM_TYPE = DATASET_CURRICULUM_TYPE.CURRICULUM):
        """
        Args:
            dataset_path (Path): Path to datset with columns "prompt", "answer", and optionally "difficulty"
            tokenizer: Tokenizer to use
            unique_samples: Number of unique samples to use
            shuffle: Whether to shuffle samples
            allow_negative: Whether to allow negative numbers
        """
        super().__init__(dataset_path, tokenizer, maxsize, shuffle, additional_columns=[difficulty_column])
        self._init_curriculum_attributes([dataset_path] if dataset_path is not None else ["placeholder"], [dataset_curriculum_type], difficulty_column)
        
        # Check if difficulty column exists
        self.has_difficulty = self.difficulty_column in self.df.columns
        if self.has_difficulty:
            self.df[difficulty_column] = self.df[difficulty_column].astype(int)
            self.df = self.df.sort_values(difficulty_column).reset_index(drop=True)
            self.difficulty_column = difficulty_column
            self.difficulties: list[int] = sorted(self.df[self.difficulty_column].unique().tolist())
            self.difficulty_indices: dict[int, tuple[int, int]] = {
                difficulty: (self.df[self.df[self.difficulty_column] == difficulty].index[0].item(), 
                                self.df[self.df[self.difficulty_column] == difficulty].index[-1].item() + 1)
                for difficulty in self.difficulties
            }

            logging.info(f"Found difficulties: {self.difficulties}")
        else:
            self.difficulties = [0]  # Default single difficulty
            self.difficulty_indices = {0: (0, len(self.df))}
        self._set_remaining_curriculum_attributes(max_difficulty=[max(self.difficulties)], has_difficulty_column=[self.has_difficulty])

    def switch_to_standard_sampling(self):
        """Switch to standard sampling mode (no difficulty sampling)."""
        self.difficulties = [0]  # Default single difficulty
        self.difficulty_indices = {0: (0, len(self.df))}

    def get_available_difficulties(self, _: Any) -> list[int]:
        """Get list of available difficulty levels."""
        return self.difficulties

    def __getitem__(self, idx) -> EfficientCurriculumPromptBatch:
        ret: EfficientCurriculumPromptBatch = super().__getitem__(idx)
        ret["difficulty"] = self.df.iloc[idx][self.difficulty_column] if self.has_difficulty else 0
        return ret

    def get_difficulty_range(self, task_id: int, difficulty: int) -> tuple[int, int]:
        if difficulty not in self.difficulties:
            raise ValueError(f"Difficulty {difficulty} not found in dataset")
        return self.difficulty_indices[difficulty]
