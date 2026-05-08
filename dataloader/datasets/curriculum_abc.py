import bisect
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from time import time
from typing import Callable, Optional

import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from dataloader.datasets.pretokenized_dataset import DF_CHUNK_SIZE
from utils.enums import DATASET_CURRICULUM_TYPE

tqdm.pandas()


class CurriculumDatasetABC(ABC):
    """Abstract base class for datasets that support curriculum learning."""
    number_of_tasks: int
    max_difficulty: list[int]
    has_difficulty_column: list[bool]
    difficulty_column: str
    difficulty_indices: list[dict[int, tuple[int, int]]]
    dataset_curriculum_types: list[DATASET_CURRICULUM_TYPE]

    def _init_curriculum_attributes(self, dataset_paths, dataset_curriculum_types: list[DATASET_CURRICULUM_TYPE], difficulty_column="difficulty"):
        """Initialize curriculum-related attributes."""
        self.difficulty_indices: list[dict[int, tuple[int, int]]] = []
        self.max_difficulty: list[int] = []
        self.difficulty_column = difficulty_column
        self.has_difficulty_column: list[bool] = []
        self.number_of_tasks = len(dataset_paths)
        self.dataset_curriculum_types = dataset_curriculum_types

    def _set_remaining_curriculum_attributes(self, max_difficulty: list[int], has_difficulty_column: list[bool]):
        """Set any remaining curriculum attributes after loading all datasets."""
        self.max_difficulty: list[int] = max_difficulty
        self.has_difficulty_column: list[bool] = has_difficulty_column
    
    @abstractmethod
    def get_available_difficulties(self, task_id: int) -> list[int]:
        """Get list of available difficulty levels in the dataset.
        
        Args:
            task_id (int): ID of the task (dataset).
            
        Returns:
            list[int]: List of available difficulty levels.
        """
        pass

    def supports_curriculum_learning(self, task_id: int) -> bool:
        """Check if this dataset supports curriculum learning (has difficulty column).
        
        Args:
            task_id (int): ID of the task (dataset).
            
        Returns:
            bool: True if the dataset has a difficulty column, False otherwise.
        """
        return True
    
    @abstractmethod
    def get_difficulty_range(self, task_id: int, difficulty: int) -> tuple[int, int]:
        """Get the token index range for a specific difficulty level.
        
        Args:
            task_id (int): ID of the task (dataset).
            difficulty (int): Difficulty level to get the range for.
            
        Returns:
            tuple[int, int]: Start and end token indices for the specified difficulty.
            
        Raises:
            ValueError: If the difficulty is not found or if the dataset does not support curriculum learning
        """
        pass


class CurriculumPretokenizedMixin:
    """Mixin class that provides shared curriculum learning functionality."""
    number_of_tasks: int
    max_difficulty: list[int]
    has_difficulty_column: list[bool]
    difficulty_column: str
    difficulty_indices: list[dict[int, tuple[int, int]]]
    
    def _preprocess_curriculum_dataframe(
        self, 
        df: DataFrame, 
        tokenizer: PreTrainedTokenizerFast, 
        cache_path: Path, 
        dataset_path: Path, 
        tokenizer_path: Optional[Path],
        save_cache_callback: Callable
    ):
        """
        Preprocess curriculum dataframe with difficulty tracking.
        
        Args:
            df (DataFrame): DataFrame with difficulty column
            tokenizer (PreTrainedTokenizerFast): Tokenizer to use
            cache_path (Path): Path to cache directory
            dataset_path (Path): Path to original dataset
            tokenizer_path (Path): Path to tokenizer
            save_cache_callback (Callable): Function to call for saving cache chunks
            
        Returns:
            None - saves difficulty indices to cache
        """
        logging.info("Preprocessing curriculum data...")
        s = time()
        
        # Cast difficulty column to int
        df[self.difficulty_column] = df[self.difficulty_column].astype(int)

        # Sort DataFrame by difficulty
        df = df.sort_values(self.difficulty_column).reset_index(drop=True)
        max_difficulty = int(df[self.difficulty_column].max())
        min_difficulty = int(df[self.difficulty_column].min())
        
        logging.info(f"Found {len(df)} samples with difficulties {min_difficulty}-{max_difficulty}")
        
        # Track difficulty boundaries in token space
        difficulty_token_indices = {}
        current_difficulty = min_difficulty
        difficulty_start_token = np.int64(0)
        total_tokens = 0
        
        for i in tqdm(range(0, len(df), DF_CHUNK_SIZE), desc="Tokenizing curriculum dataframe"):
            chunk = df.iloc[i:i+DF_CHUNK_SIZE]
            
            # Process chunk - this will be customized by the callback
            data, labels, total_tokens = save_cache_callback(
                chunk, tokenizer, cache_path, dataset_path, tokenizer_path, total_tokens
            )
            
            # Track difficulty boundaries within this chunk
            chunk_difficulties = chunk[self.difficulty_column].values

            # Vectorized approach: find all BOS and EOS positions at once
            bos_positions = np.where(data == getattr(self, 'bos_token_id', tokenizer.bos_token_id))[0]
            eos_positions = np.where(data == getattr(self, 'eos_token_id', tokenizer.eos_token_id))[0]

            # Ensure we have matching BOS/EOS pairs
            min_pairs = min(len(bos_positions), len(eos_positions))
            if min_pairs != len(chunk_difficulties):
                logging.warning(f"Token pair mismatch: expected {len(chunk_difficulties)}, found {min_pairs}")
                min_pairs = min(min_pairs, len(chunk_difficulties))

            # Process difficulty changes by comparing consecutive difficulties
            difficulty_changes = np.where(chunk_difficulties[1:] != chunk_difficulties[:-1])[0] + 1
            difficulty_change_positions = np.concatenate([[0], difficulty_changes, [len(chunk_difficulties)]])

            for j in range(len(difficulty_change_positions) - 1):
                start_row = difficulty_change_positions[j]
                row_difficulty = chunk_difficulties[start_row]
                
                if start_row < min_pairs:  # Ensure we don't go out of bounds
                    row_start = bos_positions[start_row]
                    
                    # Check if we've moved to a new difficulty
                    if row_difficulty != current_difficulty:
                        # Save the previous difficulty range
                        if current_difficulty is not None:
                            difficulty_token_indices[current_difficulty] = (
                                difficulty_start_token.item(), 
                                (total_tokens - len(data) + row_start).item()
                            )
                        
                        # Start new difficulty
                        current_difficulty = row_difficulty.item()
                        difficulty_start_token = total_tokens - len(data) + row_start
        
        # Save the last difficulty range
        difficulty_token_indices[current_difficulty] = (difficulty_start_token.item(), total_tokens)
        
        # Save difficulty indices to JSON file
        self._save_difficulty_indices(cache_path, difficulty_token_indices, max_difficulty, dataset_path, total_tokens)
        
        logging.info(f"Finished preprocessing curriculum data in {time()-s:.2f}s")
        logging.info(f"Difficulty ranges: {difficulty_token_indices}")
    
    def _save_difficulty_indices(self, cache_path: Path, difficulty_token_indices: dict, max_difficulty: int, dataset_path: Path, total_tokens: int) -> None:
        """
        Save difficulty indices to JSON file and update cache info.
        Args:
            cache_path (Path): Path to cache directory
            difficulty_token_indices (dict): Mapping of difficulty levels to token index ranges
            max_difficulty (int): Maximum difficulty level found
            dataset_path (Path): Path to original dataset
            total_tokens (int): Total number of tokens processed
        """
        difficulty_indices_path = cache_path / "difficulty_indices.json"
        with open(difficulty_indices_path, "w") as f:
            json.dump({
                "difficulty_indices": {str(k): v for k, v in difficulty_token_indices.items()},
                "max_difficulty": max_difficulty
            }, f, indent=2)
        
        # Update cache info
        with open(cache_path.parent / "info.txt", "a") as f:
            f.write(f"\n{cache_path.name}")
            f.write(f"\n\tDataset path: {dataset_path}")
            f.write(f"\n\tnum_tokens: {total_tokens}")
            f.write(f"\n\tmax_difficulty: {max_difficulty}")
            f.write(f"\n\tdifficulty_levels: {len(difficulty_token_indices)}\n")
    
    def _load_curriculum_cache(self, cache_path: Path) -> None:
        """
        Load difficulty indices from cache.
        Args:
            cache_path (Path): Path to cache directory
        """
        difficulty_indices_path = cache_path / "difficulty_indices.json"
        try:
            with open(difficulty_indices_path, "r") as f:
                data = json.load(f)
                self.difficulty_indices.append({int(k): v for k, v in data["difficulty_indices"].items()})
                self.max_difficulty.append(data["max_difficulty"])
                self.has_difficulty_column.append(True)
            logging.info(f"Loaded difficulty indices for {len(self.difficulty_indices[-1])} difficulty levels")
        except FileNotFoundError:
            logging.info("No difficulty indices file found, dataset does not support curriculum learning")
            self.difficulty_indices.append({})
            self.max_difficulty.append(0)
            self.has_difficulty_column.append(False)
    
    def _build_reverse_difficulty_mapping(self) -> None:
        """Build reverse mapping for difficulty indices for fast lookup."""
        self.reverse_difficulty_indices = []
        data_lengths = getattr(self, 'data').lengths
        # Step 1. Flatten intervals
        for task_idx, d in enumerate(self.difficulty_indices):
            dataset_start = data_lengths[task_idx-1] if task_idx > 0 else 0
            for src, (lo, hi) in d.items():
                self.reverse_difficulty_indices.append((dataset_start+lo, dataset_start+hi, src))
        # Step 2. Sort by start
        self.reverse_difficulty_indices.sort(key=lambda t: t[0])   # sort by lo
        self.difficulty_starts = [lo for lo, hi, src in self.reverse_difficulty_indices]
    
    def get_available_difficulties(self, task_id: int) -> list[int]:
        """
        Get list of available difficulty levels in the dataset.
        Args:
            task_id (int): ID of the task (dataset).
        Returns:
            list[int]: List of available difficulty levels.
        """
        if task_id >= len(self.has_difficulty_column) or not self.has_difficulty_column[task_id]:
            # Fallback: single difficulty level 0
            return [0]
        return sorted(self.difficulty_indices[task_id].keys())
    
    def get_difficulty_range(self, task_id: int, difficulty: int) -> tuple[int, int]:
        """
        Get the token index range for a specific difficulty level.
        Args:
            task_id (int): ID of the task (dataset).
            difficulty (int): Difficulty level to get the range for.
        Returns:
            tuple[int, int]: Start and end token indices for the specified difficulty.
        Raises:
            ValueError: If the difficulty is not found or if the dataset does not support curriculum learning
        """
        data_lengths = getattr(self, 'data').lengths
        dataset_start = data_lengths[task_id-1] if task_id > 0 else 0
        dataset_end = data_lengths[task_id]
        if not self.supports_curriculum_learning(task_id):
            # Fallback: treat entire dataset as single difficulty level
            if difficulty == 0:
                # Need to access data.lengths from the concrete class
                return (dataset_start, dataset_end)
            else:
                raise ValueError("Dataset does not support curriculum learning, only difficulty 0 is available")
        
        if difficulty not in self.difficulty_indices[task_id]:
            raise ValueError(f"Difficulty {difficulty} not found in dataset")
        difficulty_start, difficulty_end =  self.difficulty_indices[task_id][difficulty]
        return dataset_start + difficulty_start, dataset_start + difficulty_end
    
    def supports_curriculum_learning(self, task_id: int) -> bool:
        """
        Check if this dataset supports curriculum learning.
        Args:
            task_id (int): ID of the task (dataset).
        Returns:
            bool: True if the dataset has a difficulty column, False otherwise.
        """
        if task_id >= len(self.has_difficulty_column):
            return False
        if self.dataset_curriculum_types[task_id] in [DATASET_CURRICULUM_TYPE.STANDARD, DATASET_CURRICULUM_TYPE.ENDGAME]:
            return False
        return self.has_difficulty_column[task_id]
    
    def get_difficulty_for_idx(self, x: int) -> int:
        """
        Get difficulty level for a specific index in the dataset.
        Args:
            x (int): Index in the dataset.
        Returns:
            int: Difficulty level for the given index, or -1 if not found.
        """
        if not hasattr(self, 'reverse_difficulty_indices'):
            self._build_reverse_difficulty_mapping()
        
        if not self.difficulty_starts:  # No curriculum data available
            return 0
            
        idx = bisect.bisect_right(self.difficulty_starts, x) - 1
        if idx >= 0:
            lo, hi, src = self.reverse_difficulty_indices[idx]
            if lo <= x <= hi:
                return src
        return -1
