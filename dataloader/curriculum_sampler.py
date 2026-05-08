import logging
from math import ceil
from typing import Dict, Generator, Optional, override

import numpy as np
import torch

from dataloader.curriculum_manager import CurriculumManager
from dataloader.dataset_utils import RatioSampler
from dataloader.datasets.curriculum_abc import CurriculumDatasetABC


def _create_difficulty_sampler(start, end) -> Generator[int, None, None]:
    """Create a generator that yields shuffled indices from start to end."""
    assert start < end, f"Start {start} must be less than end {end}"
    indices = np.arange(start, end)
    while True:
        np.random.shuffle(indices)
        for idx in indices:
            yield idx


class CurriculumFixedRatioSampler(RatioSampler):
    """
    Sampler that handles both task-level and difficulty-level ratios for curriculum learning.
    Can be seen as a FixedRatioSampler with additional difficulty management.
    """
    
    def __init__(
        self,
        dataset: CurriculumDatasetABC,
        task_ratios: torch.FloatTensor,
        difficulty_ratios: Dict[int,torch.FloatTensor],
        sampling_difficulties: Dict[int, torch.LongTensor],
        context_length: int,
        batch_size: int,
        curriculum_manager: Optional[CurriculumManager] = None,
    ):
        """
        Args:
            datasets: List of curriculum datasets
            task_ratios: Array of ratios for each task (shape: num_tasks)
            difficulty_ratios: Dict mapping task_idx -> array of ratios for sampling difficulties
            sampling_difficulties: Dict mapping task_idx -> list of all sampling difficulty levels
            context_length: Context length
            batch_size: Batch size
        """
        self.dataset = dataset
        self.num_tasks = dataset.number_of_tasks
        self.context_length = context_length
        self.batch_size = batch_size
        self.curriculum_manager = curriculum_manager
        
        # Validate inputs
        assert len(task_ratios) == self.num_tasks, f"task_ratios length {len(task_ratios)} != num_tasks {self.num_tasks}"
        assert len(difficulty_ratios) == self.num_tasks, f"difficulty_ratios length {len(difficulty_ratios)} != num_tasks {self.num_tasks}"
        assert len(sampling_difficulties) == self.num_tasks, f"sampling_difficulties length {len(sampling_difficulties)} != num_tasks {self.num_tasks}"
        
        self.difficulty_proportions: Dict[int, torch.FloatTensor] = {k: torch.full_like(v, -1) for k, v in difficulty_ratios.items()}
        self.sampling_difficulties: dict[int, torch.LongTensor] = {k: torch.full_like(v, -1) for k,v in sampling_difficulties.items()}
        self.difficulty_samplers: Dict[tuple[int, int], Generator] = {}
        self.global_sampler: Dict[int, Generator] = dict()
        self.update_ratios(
            task_ratios=task_ratios,
            difficulty_ratios=difficulty_ratios,
            sampling_difficulties=sampling_difficulties
        )
        if curriculum_manager is not None:
            self.task_proportions: torch.FloatTensor = torch.tensor(
                [1. if curriculum_manager.isTaskActive(i) else 0. for i in range(curriculum_manager.num_tasks)],
                dtype=torch.float32
            )
            self.difficulty_sampling: dict[int, bool] = curriculum_manager.difficulty_sampling
        else:
            self.task_proportions: torch.FloatTensor = torch.ones((self.num_tasks,), dtype=torch.float32)
            self.difficulty_sampling: dict[int, bool] = {k: True for k in sampling_difficulties.keys()}
        
        logging.info(f"Initialized CurriculumFixedRatioSampler with {self.num_tasks} tasks")
        for task_idx in range(self.num_tasks):
            sampling_diffs = self.sampling_difficulties[task_idx]
            logging.info(f"Task {task_idx}: sampling difficulties {sampling_diffs.tolist()}")
    
    def _validate_difficulties(self, sampling_difficulties: Dict[int, torch.LongTensor]):
        """Validate that all sampling difficulties exist in the datasets."""
        
        for task_idx in range(self.num_tasks):
            # Check if dataset supports curriculum learning
            if not self.dataset.supports_curriculum_learning(task_idx):
                # For non-curriculum datasets, only difficulty 0 should be active
                sampling_diffs = set(sampling_difficulties[task_idx].tolist())
                if sampling_diffs != {0}:
                    raise ValueError(
                        f"Task {task_idx}: Dataset does not support curriculum learning. "
                        f"Only difficulty 0 is allowed, but sampling difficulties are: {sorted(sampling_diffs)}"
                    )
                logging.debug(f"Task {task_idx}: validated non-curriculum dataset (difficulty 0 only)")
                continue
            
            available_diffs = set(self.dataset.get_available_difficulties(task_idx))
            sampling_diffs = set(sampling_difficulties[task_idx].tolist())
            
            missing_diffs = sampling_diffs - available_diffs
            if missing_diffs:
                raise ValueError(
                    f"Task {task_idx}: Sampling difficulties {sorted(missing_diffs)} not found in dataset. "
                    f"Available difficulties: {sorted(available_diffs)}"
                )
            
    def update_ratios(
        self,
        task_ratios: Optional[torch.FloatTensor]=None,
        difficulty_ratios: Optional[Dict[int, torch.FloatTensor]]=None,
        sampling_difficulties: Optional[Dict[int, torch.LongTensor]]=None,
    ):
        """
        Update ratios and sampling difficulties.
        Args:
            task_ratios: New task ratios (optional)
            difficulty_ratios: New difficulty ratios for each task (optional)
            sampling_difficulties: New sampling difficulties for each task (optional)
        """
        if task_ratios is not None:
            self.task_ratios: torch.FloatTensor = task_ratios / task_ratios.sum()
            logging.debug(f"Updated task ratios: {self.task_ratios.tolist()}")
        
        if difficulty_ratios is not None:
            self.difficulty_ratios = {k: v.float() for k, v in difficulty_ratios.items()}
            logging.debug("Updated difficulty ratios")
        
        if sampling_difficulties is not None:
            # Validate new difficulties before updating
            self._validate_difficulties(sampling_difficulties)
            
            old_sampling = self.sampling_difficulties.copy()
            self.sampling_difficulties = sampling_difficulties.copy()
            
            # Check if we need to create new samplers
            for task_idx in range(self.num_tasks):
                old_diffs = set(old_sampling[task_idx].tolist())
                new_diffs = set(self.sampling_difficulties[task_idx].tolist())
                
                if new_diffs != old_diffs:
                    logging.info(f"Task {task_idx}: difficulty change {old_diffs} -> {new_diffs}")
                    
                    # Remove old samplers
                    for difficulty in old_diffs:
                        key = (task_idx, difficulty)
                        if key in self.difficulty_samplers:
                            del self.difficulty_samplers[key]
                    
                    # Create new samplers
                    self._create_difficulty_samplers(task_idx, new_diffs)
        
        # Update difficulty proportions
        self._update_difficulty_proportions()


    def _update_difficulty_proportions(self):
        """Update difficulty proportions based on current ratios."""
        for task_idx in range(self.num_tasks):
            sampling_diffs = self.sampling_difficulties[task_idx]
            ratios = self.difficulty_ratios[task_idx][:len(sampling_diffs)]
            
            # Normalize difficulty ratios to sum to task ratio
            normalized_ratios: torch.FloatTensor = ratios * self.task_ratios[task_idx] / (ratios.sum() if ratios.sum() > 0 else 1)
            self.difficulty_proportions[task_idx] = CurriculumFixedRatioSampler._compute_proportions(normalized_ratios, self.batch_size, self.difficulty_proportions[task_idx])
    
    def _create_difficulty_samplers(self, task_idx: int, difficulties: set[int]):
        """Create samplers for each (task, difficulty) combination."""
        difficulty_list: list[int] = sorted(difficulties)
        for difficulty in difficulties:
            try:
                start_token, end_token = self.dataset.get_difficulty_range(task_idx, difficulty)
                start_context = ceil((start_token) / self.context_length)
                end_context = max(start_context, end_token // self.context_length - 1)
                if end_context == start_context:
                    # Check if can move context range
                    if self.curriculum_manager is not None:
                       max_difficulty = self.curriculum_manager.max_difficulties[task_idx]
                    else:
                       max_difficulty = self.sampling_difficulties[task_idx].max().item()
                    if difficulty < max_difficulty:
                       end_context += 1
                       logging.warning(f"Task {task_idx}, difficulty {difficulty}: Adjusted end_context to {end_context} to ensure valid range")
                    else:
                       start_context = max(0, start_context - 1)
                       logging.warning(f"Task {task_idx}, difficulty {difficulty}: Adjusted start_context to {start_context} to ensure valid range")
                
                assert start_context < end_context, f"Start {start_context} must be less than end {end_context}"
                sampler = _create_difficulty_sampler(start_context, end_context)
                self.difficulty_samplers[(task_idx, difficulty)] = sampler
                logging.debug(f"Created sampler for task {task_idx}, difficulty {difficulty}: contexts {start_context}-{end_context}")
            except ValueError as e:
                logging.error(f"Error creating sampler for task {task_idx}, difficulty {difficulty}: {e}")
        start_token = self.dataset.get_difficulty_range(task_idx, difficulty_list[0])[0]
        end_token = self.dataset.get_difficulty_range(task_idx, difficulty_list[-1])[1]
        start_context = ceil((start_token) / self.context_length)
        end_context = (end_token // self.context_length - 1)
        self.global_sampler[task_idx] = _create_difficulty_sampler(start_context,end_context)
    
    def _select_task(self) -> int:
        """Select a task based on current task proportions."""
        if (self.task_proportions < 1).all():
            self.task_proportions = CurriculumFixedRatioSampler._compute_proportions(
                self.task_ratios, self.batch_size, self.task_proportions
            )
        return self.task_proportions.argmax().item()
    
    def _select_difficulty(self, task_idx: int) -> int:
        """Select a difficulty for the given task based on difficulty proportions."""
        if task_idx not in self.difficulty_proportions:
            # Fallback to first sampling difficulty
            return self.sampling_difficulties[task_idx][0].item()
        
        difficulty_props = self.difficulty_proportions[task_idx]
        
        if (difficulty_props < 1).all():
            sampling_diffs = self.sampling_difficulties[task_idx]
            ratios = self.difficulty_ratios[task_idx][:len(sampling_diffs)]
            # Normalize to sum to 1 for proportional sampling
            normalized_ratios: torch.FloatTensor = ratios / ratios.sum() if ratios.sum() > 0 else torch.ones_like(ratios) / len(ratios)
            difficulty_props = CurriculumFixedRatioSampler._compute_proportions(
                normalized_ratios, self.batch_size, difficulty_props
            )
            self.difficulty_proportions[task_idx] = difficulty_props
        
        selected_difficulty_idx: int = difficulty_props.argmax().item()
        selected_difficulty = self.sampling_difficulties[task_idx][selected_difficulty_idx]
        
        # Decrease proportion for selected difficulty
        self.difficulty_proportions[task_idx][selected_difficulty_idx] -= 1
        
        return selected_difficulty.item()
    
    def _sample_from_task_difficulty(self, task_idx: int, difficulty: int) -> int:
        """Sample an index from the specified task and difficulty combination."""
        sampler_key = (task_idx, difficulty)
        
        if sampler_key not in self.difficulty_samplers:
            raise ValueError(f"No sampler available for task {task_idx}, difficulty {difficulty}")
        
        return next(self.difficulty_samplers[sampler_key])

    
    @override
    def __iter__(self):
        while True:
            # 1. Select task based on task ratios
            task_idx = self._select_task()
            self.task_proportions[task_idx] -= 1
            
            if self.difficulty_sampling[task_idx]:
                # 2. Select difficulty within task based on difficulty ratios
                difficulty = self._select_difficulty(task_idx)
                
                # 3. Sample index from the selected (task, difficulty) combination
                try:
                    idx = self._sample_from_task_difficulty(task_idx, difficulty)
                    yield idx
                except ValueError as e:
                    logging.error(f"Error sampling from task {task_idx}, difficulty {difficulty}: {e}")
                    # Fallback: try first available difficulty for this task
                    fallback_difficulty: int = self.sampling_difficulties[task_idx][0]
                    try:
                        idx = self._sample_from_task_difficulty(task_idx, fallback_difficulty)
                        yield idx
                    except ValueError:
                        logging.error(f"Fallback failed for task {task_idx}")
                        continue
            else:
                # If not using difficulty sampling, sample from global sampler
                idx = next(self.global_sampler[task_idx])
                yield idx

    def __len__(self):
        """Return total number of available context windows across all sampling difficulties."""
        total_length = 0
        for task_idx in range(self.num_tasks):
            for difficulty in self.sampling_difficulties[task_idx].tolist():
                try:
                    start_token, end_token = self.dataset.get_difficulty_range(task_idx,difficulty)
                    length = (end_token - start_token) // self.context_length
                    total_length += max(0, length - 1)  # -1 for rand_offset
                except ValueError:
                    continue
        return total_length

    @override
    def update_task_ratio(self, task_selector: slice | list[int], ratios: torch.FloatTensor):
        assert self.curriculum_manager is not None, "Curriculum manager must be set to use this method"
        self.proxy_task_ratios[task_selector] = ratios

        task_ratios: torch.FloatTensor = self.task_ratios.clone()
        task_ratios[task_selector] = ratios
        if self.curriculum_manager is not None:
            self.curriculum_manager.update_task_ratios(task_ratios)
            # Get updated sampling difficulties from curriculum manager
            sampling_difficulties = {}
            for task_idx in range(self.num_tasks):
                sampling_difficulties[task_idx] = self.curriculum_manager.get_all_sampling_difficulties(task_idx)
            
            self.update_ratios(
                task_ratios=task_ratios,
                difficulty_ratios=self.curriculum_manager.difficulty_ratios,
                sampling_difficulties=sampling_difficulties
            )
        else:
            self.update_ratios(task_ratios=task_ratios)
    
    # Helper method for when no parameters are passed
    def update_ratios_from_curriculum_manager(self):
        """Update ratios by fetching current state from curriculum manager."""
        if self.curriculum_manager is not None:
            # Get updated sampling difficulties from curriculum manager
            sampling_difficulties = {}
            for task_idx in range(self.num_tasks):
                sampling_difficulties[task_idx] = self.curriculum_manager.get_all_sampling_difficulties(task_idx)
            self.proxy_task_ratios = self.curriculum_manager.task_ratios.clone()
            self.update_ratios(
                task_ratios=self.curriculum_manager.task_ratios,
                difficulty_ratios=self.curriculum_manager.difficulty_ratios,
                sampling_difficulties=sampling_difficulties
            )

    @override
    def get_task_ratios(self) -> torch.FloatTensor:
        return self.task_ratios.clone()
