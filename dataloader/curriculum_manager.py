import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
from numpy import clip

from utils.enums import DATASET_CURRICULUM_TYPE
from utils.metrics import inverse_generalized_mean

if TYPE_CHECKING:
    from dataloader.datasets.curriculum_abc import (
        CurriculumDatasetABC,
    )


class CurriculumManager:
    """
    Manages curriculum learning by tracking performance and updating difficulty ratios.
    """
    
    def __init__(
        self,
        num_tasks: int,
        effective_batch_size: int = 384,
        initial_max_difficulty_fraction: float = .1,
        advancement_threshold: float = 0.9,
        new_difficulty_ratio: float = 0.3,
        generalized_mean_power: float = -1.0,
        performance_history_size: int = 3,
        advancement_difficulty_window_size: int = 3,
        advancement_performance_window_size: int = 1,
        advancement_step_size_fraction: float = .02,
        preview_difficulty_ratio: float = 0.0,
        preview_exponential_decay: float = 0.5,
        endgame_switch_step_fraction: float = 1.,
        endgame_switch_lr_fraction: float = 0.,
        standby_task_fraction: float = 0.1,
        total_training_steps: int = 0,
        max_lr: float = 0,
        min_difficulty_ratio: float = 0.01
    ):
        """
        Args:
            num_tasks (int): Number of tasks
            effective_batch_size (int): Effective batch size for scaling advancement step size (default: 384)
            initial_max_difficulty_fraction (float): Initial maximum difficulty level (default: 10% of max difficulty)
            advancement_threshold (float): Performance threshold for advancing to next difficulty. (default: 0.9)
            new_difficulty_ratio (float): Fraction of task ratio to assign to new difficulties (default: 0.3)
            generalized_mean_power (float): Power for generalized mean calculation (default: -1.0 for harmonic mean)
            performance_history_size (int): Number of evaluations to track for computing sampling ratios (default: 3)
            advancement_difficulty_window_size (int): Number of evaluations to consider for advancement (default: 3)
            advancement_performance_window_size (int): Number of evaluations to track for advancement (default: 1)
            advancement_step_size_fraction (float): Number of difficulty levels to advance at once as a fraction of the number of available difficulties (default: 2%)
            preview_difficulty_ratio (float): Fraction of task ratio for preview difficulties (default: 0.0)
            preview_exponential_decay (float): Exponential decay rate for preview difficulties (default: 0.5)
            endgame_switch_step_fraction (float): Step to switch to endgame dataset if curriculum is not mastered by then (default: 1, i.e. never)
            endgame_switch_lr_fraction (float): Learning rate fraction to switch to endgame dataset if
            standby_task_fraction (float): Fraction of ratio to keep for standby tasks (endame already active)
            total_training_steps (int): Total number of training steps planned (for endgame switching)
            max_lr (float): Maximum learning rate (for endgame switching)
            min_difficulty_ratio (float): Minimum difficulty ratio to avoid zero probabilities (default: 1% of effective batch size)
        """
        self.num_tasks = num_tasks
        self.effective_batch_size = effective_batch_size
        self.initial_max_difficulty_fraction = initial_max_difficulty_fraction
        self.advancement_threshold = advancement_threshold
        self.new_difficulty_ratio = new_difficulty_ratio
        self.generalized_mean_power = generalized_mean_power
        self.performance_history_size = performance_history_size
        self.advancement_difficulty_window_size = advancement_difficulty_window_size
        self.advancement_performance_window_size = advancement_performance_window_size
        self.advancement_step_size_fraction = advancement_step_size_fraction
        self.preview_difficulty_ratio = preview_difficulty_ratio
        self.preview_exponential_decay = preview_exponential_decay
        self.endgame_switch_step_fraction = endgame_switch_step_fraction
        self.endgame_switch_lr_fraction = endgame_switch_lr_fraction
        self.standby_task_fraction = standby_task_fraction
        self.total_training_steps = total_training_steps
        self.max_lr = max_lr
        self.min_difficulty_ratio = min_difficulty_ratio

        self.threshold_borders: dict[int, dict[int, float]] = dict()
        
        self.advancement_step_size: list[int] = []
        # Initialize frontier difficulties (will be set properly during synchronization)
        self.frontier_difficulties: Dict[int, int] = {}
        for task_idx in range(num_tasks):
            self.frontier_difficulties[task_idx] = -1
        self.supports_curriculum: dict[int, bool] = dict()
        # Initialize ratios
        self.task_ratios: torch.FloatTensor = torch.ones(num_tasks, dtype=torch.float32) / num_tasks
        self.difficulty_ratios: Dict[int, torch.FloatTensor] = {}
        
        # Performance tracking
        self.performance_history: Dict[Tuple[int, int], List[float]] = {}  # (task, difficulty) -> scores

        self.dataset_curriculum_types: list[DATASET_CURRICULUM_TYPE] = list()
        
        # Available and maximum difficulties per task (auto-detected from data)
        self.available_difficulties: Dict[int, List[int]] = {}  # All difficulties available in datasets
        self.max_difficulties: Dict[int, int] = {}
        self.min_difficulties: Dict[int, int] = {}
        self.difficulty_sampling: Dict[int, bool] = {i: True for i in range(num_tasks)}
        
        logging.info(f"Initialized CurriculumManager with {num_tasks} tasks")
        if preview_difficulty_ratio > 0:
            logging.info(f"Preview sampling enabled: ratio={preview_difficulty_ratio}, decay={preview_exponential_decay}")

    def get_all_sampling_difficulties(self, task_idx: int) -> torch.LongTensor:
        """
        Get all difficulties that can be sampled (frontier + preview).
        Args:
            task_idx (int): Task index
        Returns:
            sampling_difficulties (torch.LongTensor): All difficulties available for sampling.
        """
        frontier_and_below = self._get_frontier_and_below_difficulties(task_idx)
        
        if self.preview_difficulty_ratio == 0:
            return torch.LongTensor(frontier_and_below)
        
        preview_diffs = self._get_preview_difficulties(task_idx)
        all_sampling = frontier_and_below + preview_diffs
        return torch.LongTensor(all_sampling)
    
    def _update_threshold_borders(self, lr: float, step: int):
        """
        Update threshold borders for all tasks based on current learning rate and step.
        
        Cascading schedule per difficulty level (controlled by endgame_switch_*_fraction parameters):
        - Easiest difficulty (0): threshold reduces from 100% to 0% over progress 0% to 50%
        - Hardest difficulty (max): threshold reduces from 100% to 0% over progress 50% to 75%
        - Middle difficulties: linear interpolation of start/end points
        
        Example with endgame_switch_lr_fraction=0.5:
        - Difficulty 0: starts reducing immediately, reaches 0% at 50% LR decay
        - Difficulty max: starts reducing at 50% LR decay, reaches 0% at 75% LR decay
        
        Args:
            lr (float): Current learning rate
            step (int): Current training step
        Modifies:
        ---------
            self.threshold_borders (dict[int, Dict[int, float]]): Per-difficulty threshold multipliers for each task.
        """
        lr_percentage = (self.max_lr - lr) / (self.endgame_switch_lr_fraction * self.max_lr + 1e-100)
        step_percentage = step / (self.endgame_switch_step_fraction * self.total_training_steps + 1e-100)
        progress = max(lr_percentage, step_percentage)
        
        # Calculate per-difficulty threshold multipliers for each task
        self.threshold_borders = {}
        for task_idx in range(self.num_tasks):
            max_diff = self.max_difficulties.get(task_idx, 0)
            if max_diff == 0:
                self.threshold_borders[task_idx] = {0: 1.0}  # No curriculum
                continue
            
            difficulty_multipliers = {}
            for diff in range(max_diff + 1):
                difficulty_ratio = diff / max_diff
                
                # Cascading schedule:
                # - Difficulty 0: starts at 0% progress, reaches 0 at 50% progress (1.0)
                # - Difficulty max_diff: starts at 50% progress (1.0), reaches 0 at 75% progress (1.5)
                start_progress = difficulty_ratio * 1.0
                end_progress = start_progress + 0.5
                
                if progress <= start_progress:
                    multiplier = 1.0  # Full threshold
                elif progress <= end_progress:
                    # Linear decrease from 1.0 to 0.0
                    multiplier = 1.0 - (progress - start_progress) / (end_progress - start_progress)
                else:
                    multiplier = 0.0  # No threshold
                
                difficulty_multipliers[diff] = multiplier
            
            self.threshold_borders[task_idx] = difficulty_multipliers
        
        logging.debug(f"Updated threshold borders: {self.threshold_borders}")
        
    
    def _get_frontier_and_below_difficulties(self, task_idx: int) -> List[int]:
        """
        Get all difficulties <= frontier difficulty.
        Args:
            task_idx (int): Task index
        Returns:
            frontier_and_below (List[int]): All difficulties available for sampling that are <= frontier.
        """
        if not self.supports_curriculum.get(task_idx, True):
            return [0]
        
        frontier = self.frontier_difficulties[task_idx]
        available_diffs = self.available_difficulties.get(task_idx, [])
        return [d for d in available_diffs if d <= frontier]
    
    def _get_preview_difficulties(self, task_idx: int) -> List[int]:
        """
        Get preview difficulties (> frontier).
        Args:
            task_idx (int): Task index
        Returns:
            preview_candidates (List[int]): Preview difficulties available for sampling.
        """
        if not self.supports_curriculum.get(task_idx, True):
            return []
        
        frontier = self.frontier_difficulties[task_idx]
        available_diffs = self.available_difficulties.get(task_idx, [])
        preview_candidates = [d for d in available_diffs if d > frontier]
        return preview_candidates
    
    def _compute_preview_ratios(self, task_idx: int, preview_diffs: List[int]) -> torch.FloatTensor:
        """
        Compute exponential decay ratios for preview difficulties.
        Args:
            task_idx (int): Task index
            preview_diffs (List[int]): Preview difficulties available for sampling.
        Returns:
            ratios (torch.FloatTensor): Ratios for preview difficulties, normalized to sum to 1
        """
        if not preview_diffs:
            return torch.FloatTensor([])
        
        frontier = self.frontier_difficulties[task_idx]
        
        # Exponential decay: ratio ∝ decay^(difficulty - frontier)
        distances = torch.FloatTensor([d - frontier for d in preview_diffs])
        weights = self.preview_exponential_decay ** distances
        
        # Normalize to sum to 1
        ratios = weights / weights.sum()
        return ratios
    
    def _check_frontier_advancement(self, task_idx: int) -> bool:
        r"""
        Check if frontier should advance using window-based performance.
        The current advancement threshold is calculated by:
        .. math::
            \tau_t = min(\tau, \tau * d / b_t)
        
        where :math:`\tau` is the advancement_threshold, :math:`d` is the current frontier difficulty,
        and :math:`\b_t` is the threshold border for the task at the current training step.
        Args:
            task_idx (int): Task index
        Returns:
            can_advance (bool): Whether the frontier can advance.
        """
        if not self.supports_curriculum.get(task_idx, True):
            return False
        
        frontier = self.frontier_difficulties[task_idx]
        
        # Check last performance_history_size difficulties ≤ frontier
        available_diffs = self.available_difficulties.get(task_idx, [])
        min_diff = self.min_difficulties.get(task_idx, 0)
        
        window_start = max(min_diff, frontier - self.advancement_difficulty_window_size + 1)
        window_diffs = [d for d in range(window_start, frontier + 1) if d in available_diffs]
        
        if not window_diffs or len(list(self.performance_history.values())[0])<self.advancement_performance_window_size:
            return False
        
        # All difficulties in window must meet threshold
        difficulty_multipliers = self.threshold_borders.get(task_idx, {})
        
        for diff in window_diffs:
            performance = self.get_recent_performance(task_idx, diff, self.advancement_performance_window_size)
            
            # Get the threshold multiplier for this specific difficulty
            difficulty_multiplier = difficulty_multipliers.get(diff, 1.0)
            
            advancement_threshold = self.advancement_threshold * difficulty_multiplier
            if performance < advancement_threshold:
                logging.debug(f"Task {task_idx}: no advancement (window performance {performance} <= {advancement_threshold}) for difficulty {diff}")
                return False
        
        return True
    
    def get_difficulty_thresholds(self, task_idx: int) -> Dict[int, float]:
        """
        Maps difficulty levels to performance thresholds for the given task.
        Args:
            task_idx (int): Task index
        Returns:
            thresholds (Dict[int, float]): Mapping from difficulty level to performance threshold.
        """
        if not self.supports_curriculum.get(task_idx, False):
            return {0: 0.0}
        
        available_diffs = self.available_difficulties.get(task_idx, [])
        difficulty_multipliers = self.threshold_borders.get(task_idx, {})
        
        thresholds = {}
        for diff in available_diffs:
            # Get the threshold multiplier for this specific difficulty
            difficulty_multiplier = difficulty_multipliers.get(diff, 1.0)
            thresholds[diff] = self.advancement_threshold * difficulty_multiplier
        
        return thresholds
    
    def get_summary(self) -> str:
        """Get a summary of current curriculum state."""
        curriculum_tasks = [i for i in range(self.num_tasks) if self.isTaskActive(i)]
        summary = [f"Curriculum Summary (Tasks: {self.num_tasks}, Curriculum-enabled: {len(curriculum_tasks)})"]
        summary.append(f"Task ratios: {self.task_ratios}")
        summary.append(f"Curriculum-enabled tasks: {curriculum_tasks}")
        summary.append(f"Preview sampling: ratio={self.preview_difficulty_ratio}, decay={self.preview_exponential_decay}")
        
        for task_idx in range(self.num_tasks):
            available_diffs = self.available_difficulties.get(task_idx, [])
            sampling_diffs = self.get_all_sampling_difficulties(task_idx)
            diff_ratios = self.difficulty_ratios.get(task_idx, torch.FloatTensor([]))
            
            frontier = self.frontier_difficulties.get(task_idx, 0)
            
            summary.append(f"Task {task_idx} ({self.dataset_curriculum_types[task_idx]}):")
            summary.append(f"  Available difficulties: {available_diffs}")
            summary.append(f"  Frontier difficulty: {frontier}")
            summary.append(f"  Sampling difficulties: {sampling_diffs.tolist()}")
            summary.append(f"  Difficulty ratios: {diff_ratios.tolist()}")
        
        return "\n".join(summary)

    def synchronize_with_datasets(self, dataset: "CurriculumDatasetABC", train_set_ratios: Optional[torch.FloatTensor]=None) -> None:
        """
        Synchronize curriculum manager with actual difficulties available in datasets.
        
        Args:
            dataset (CurriculumDatasetABC): curriculum dataset instance to synchronize with.
            train_set_ratios (Optional[torch.FloatTensor]): Optional initial task ratios to set. Values must sum to 1.
        """
        self.dataset_curriculum_types = dataset.dataset_curriculum_types
        for task_idx in range(dataset.number_of_tasks):
            if dataset.dataset_curriculum_types[task_idx] == DATASET_CURRICULUM_TYPE.ENDGAME:
                assert task_idx>0 and dataset.dataset_curriculum_types[task_idx-1] == DATASET_CURRICULUM_TYPE.CURRICULUM, (
                    f"Endgame curriculum type requires preceding curriculum type but found {dataset.dataset_curriculum_types[task_idx-1] if task_idx>0 else "N/A"} at task index {task_idx-1}"
                )
            # Check if dataset supports curriculum learning
            if not dataset.supports_curriculum_learning(task_idx) or dataset.dataset_curriculum_types[task_idx] in [DATASET_CURRICULUM_TYPE.STANDARD, DATASET_CURRICULUM_TYPE.ENDGAME]:
                logging.info(f"Dataset {task_idx} does not support curriculum learning, using single difficulty level")
                # For non-curriculum datasets, use difficulty 0 only
                assert dataset.dataset_curriculum_types[task_idx] in [DATASET_CURRICULUM_TYPE.STANDARD, DATASET_CURRICULUM_TYPE.ENDGAME], (
                    f"Non-curriculum dataset must be STANDARD or ENDGAME type, but found {dataset.dataset_curriculum_types[task_idx]} at task index {task_idx}"
                )
                available_diffs = [0]
                self.available_difficulties[task_idx] = available_diffs
                self.min_difficulties[task_idx] = 0
                self.max_difficulties[task_idx] = 0
                self.frontier_difficulties[task_idx] = 0
                self.supports_curriculum[task_idx] = False
                self.advancement_step_size.append(0)
                logging.info(f"Task {task_idx}: set to single difficulty level 0")
                continue
            self.supports_curriculum[task_idx] = True
            # Get available difficulties from dataset
            available_diffs = dataset.get_available_difficulties(task_idx)
            self.available_difficulties[task_idx] = available_diffs
            self.advancement_step_size.append(max(1, round(len(available_diffs)*self.advancement_step_size_fraction)))
            
            if available_diffs:
                self.min_difficulties[task_idx] = min(available_diffs)
                self.max_difficulties[task_idx] = max(available_diffs)
                initial_frontier_idx = clip(int(self.initial_max_difficulty_fraction * len(available_diffs)), 0, len(available_diffs)-1)
                initial_frontier = available_diffs[initial_frontier_idx]
                self.frontier_difficulties[task_idx] = initial_frontier
                
                # Ensure frontier is actually available in the dataset
                available_up_to_initial = [d for d in available_diffs if d <= initial_frontier]
                if available_up_to_initial:
                    self.frontier_difficulties[task_idx] = max(available_up_to_initial)
                else:
                    # Fallback to smallest available
                    self.frontier_difficulties[task_idx] = min(available_diffs)
                
                logging.info(f"Task {task_idx}: available difficulties {available_diffs}")
                logging.info(f"Task {task_idx}: initial frontier difficulty {self.frontier_difficulties[task_idx]}")
            else:
                logging.warning(f"Task {task_idx}: no difficulties found in dataset")
                self.available_difficulties[task_idx] = []
                self.frontier_difficulties[task_idx] = 0
        
        # Initialize difficulty ratios with correct dimensions
        if train_set_ratios is None:
            self.task_ratios = torch.tensor([1.  if self.isTaskActive(i) else 0 for i in range(self.num_tasks)], dtype=torch.float32)
            self.task_ratios = self.task_ratios / self.task_ratios.sum()
        else:
            assert abs(1-train_set_ratios.sum()) < 1e-3, f"Provided train_set_ratios must sum to 1, but sums to {train_set_ratios.sum()}. Values: {train_set_ratios}"
            self.task_ratios = train_set_ratios
        self._initialize_difficulty_ratios()
        
        logging.info("Synchronized curriculum manager with dataset difficulties")
    
    def _initialize_difficulty_ratios(self):
        """Initialize difficulty ratios with proper frontier/preview weighting."""
        for task_idx in range(self.num_tasks):
            # Use the same logic as compute_difficulty_ratios for initialization
            self.difficulty_ratios[task_idx] = self.compute_difficulty_ratios(task_idx)
        
        logging.info("Initialized difficulty ratios with proper frontier/preview weighting")
    
    def update_performance(self, task_idx: int, difficulty_performances: Dict[int, float]):
        """
        Update performance for a task across multiple difficulties.
        Only keeps the most recent performance scores (up to 10 evaluations).
        
        Args:
            task_idx: Task index
            difficulty_performances: Dict mapping difficulty -> performance score
        """
        for difficulty, performance in difficulty_performances.items():
            key = (task_idx, difficulty)
            
            if key not in self.performance_history:
                self.performance_history[key] = []
            
            self.performance_history[key].append(performance)
            
            # Keep only recent history (last 10 evaluations)
            if len(self.performance_history[key]) > 10:
                self.performance_history[key].pop(0)
        
        logging.debug(f"Updated performance for task {task_idx}: {difficulty_performances}")
    
    def get_recent_performance(self, task_idx: int, difficulty: int, performance_history_size: int = 5) -> float:
        """Get recent average performance for a specific (task, difficulty) combination.
        Args:
            task_idx (int): Task index
            difficulty (int): Difficulty level
            window_size (int): Number of recent evaluations to consider (default: 3)
        Returns:
            performance (float): Average performance over the specified window size. If no performance data is available, returns 0.0.
        """
        key = (task_idx, difficulty)
        
        if key not in self.performance_history or len(self.performance_history[key]) == 0:
            return 0.0  # No performance data available
        
        recent_scores = self.performance_history[key][-performance_history_size:]
        return torch.tensor(recent_scores).mean().item()
    
    def compute_difficulty_ratios(self, task_idx: int) -> torch.FloatTensor:
        """
        Compute ratios for all sampling difficulties (frontier + preview) of a task.
        Args:
            task_idx (int): Task index
        Returns:
            difficulty_ratios (torch.FloatTensor): Ratios for all sampling difficulties, normalized to sum to 1.
        """
        frontier_and_below_diffs = self._get_frontier_and_below_difficulties(task_idx)
        
        # Get recent performance for each difficulty
        performances: torch.FloatTensor = torch.tensor([
            self.get_recent_performance(task_idx, difficulty, self.performance_history_size) for difficulty in frontier_and_below_diffs
        ], dtype=torch.float32)
        # Use inverse generalized mean to compute ratios
        frontier_ratios = inverse_generalized_mean(task_accs=performances, p=self.generalized_mean_power)
        # Normalize to sum to 1
        frontier_ratios = frontier_ratios / frontier_ratios.sum()
        frontier_ratios = frontier_ratios.clamp_min(self.min_difficulty_ratio / len(frontier_and_below_diffs))
        frontier_ratios = frontier_ratios / frontier_ratios.sum()
        frontier_budget = (1 - self.preview_difficulty_ratio)
        
        # Compute preview ratios (exponential decay)
        preview_diffs = self._get_preview_difficulties(task_idx)
        preview_ratios = self._compute_preview_ratios(task_idx, preview_diffs)
        preview_budget = self.preview_difficulty_ratio
        
        # Combine ratios
        if len(preview_ratios) == 0:
            # No preview difficulties available
            combined_ratios = frontier_ratios
        else:
            combined_ratios = torch.cat([
                frontier_ratios * frontier_budget,
                preview_ratios * preview_budget
            ])
        
        # Scale to task ratio
        combined_ratios = combined_ratios * self.task_ratios[task_idx]
        
        return combined_ratios
    
    def isTaskActive(self, task_idx: int) -> bool:
        """Check if a task is active (curriculum or standard)."""
        return self.dataset_curriculum_types[task_idx] in [DATASET_CURRICULUM_TYPE.CURRICULUM, DATASET_CURRICULUM_TYPE.STANDARD]
    
    def advance_if_possible(self, step: int, current_lr: float) -> bool:
        """Check if any task should advance to the next difficulty level using frontier logic."""
        advancement_made = False
        
        for task_idx in range(self.num_tasks):
            if not self.supports_curriculum.get(task_idx, True):
                continue  # Skip if curriculum not supported
                
            current_frontier = self.frontier_difficulties[task_idx]
            task_max_difficulty = self.max_difficulties.get(task_idx, 0)
            
            self._update_threshold_borders(current_lr, step)
            
            # Check if we can advance (not at maximum difficulty)
            if self.dataset_curriculum_types[task_idx] == DATASET_CURRICULUM_TYPE.CURRICULUM:
                if current_frontier >= task_max_difficulty:
                    has_endgame = len(self.dataset_curriculum_types)>task_idx+1 and self.dataset_curriculum_types[task_idx+1] == DATASET_CURRICULUM_TYPE.ENDGAME
                    if self._check_frontier_advancement(task_idx):
                        if has_endgame:
                            logging.info(f"Task {task_idx}: mastered maximum difficulty {task_max_difficulty}. Transitioning to endgame dataset.")
                            self.dataset_curriculum_types[task_idx] = DATASET_CURRICULUM_TYPE.STANDBY
                            self.dataset_curriculum_types[task_idx+1] = DATASET_CURRICULUM_TYPE.STANDARD
                            self.task_ratios[task_idx+1] = (1-self.standby_task_fraction) * self.task_ratios[task_idx]
                            self.task_ratios[task_idx] = self.standby_task_fraction * self.task_ratios[task_idx]
                            advancement_made = True
                        elif self.difficulty_sampling.get(task_idx, True):
                            logging.info(f"Task {task_idx}: mastered maximum difficulty {task_max_difficulty}. Use random sampling from all difficulties.")
                            self.difficulty_sampling[task_idx] = False
                            advancement_made = True
            
                # Use window-based advancement check
                elif self._check_frontier_advancement(task_idx):
                    # Advance frontier by curriculum_step_size
                    new_frontier = min(current_frontier + self.advancement_step_size[task_idx], task_max_difficulty)
                    
                    # Ensure new frontier is actually available in the dataset
                    available_diffs = self.available_difficulties.get(task_idx, [])
                    available_up_to_new = [d for d in available_diffs if current_frontier < d <= new_frontier]
                    
                    if available_up_to_new:
                        self.frontier_difficulties[task_idx] = max(available_up_to_new)
                        advancement_made = True
                        
                        logging.info(f"Task {task_idx}: advanced frontier {current_frontier} -> {self.frontier_difficulties[task_idx]}")
                    else:
                        logging.info(f"Task {task_idx}: no higher difficulty available beyond {current_frontier}")
        
        for task_idx in range(self.num_tasks):
            self.difficulty_ratios[task_idx] = self.compute_difficulty_ratios(task_idx)
        
        return advancement_made

    def get_state(self) -> dict[str, Any]:
        """Get current state for serialization.
        Returns:
            state (dict): Dictionary containing current curriculum state.
        """
        return {
            # Core configuration parameters
            "num_tasks": self.num_tasks,
            "initial_max_difficulty": self.initial_max_difficulty_fraction,
            "advancement_threshold": self.advancement_threshold,
            "new_difficulty_ratio": self.new_difficulty_ratio,
            "generalized_mean_power": self.generalized_mean_power,
            "performance_history_size": self.performance_history_size,
            "advancement_step_size_fraction": self.advancement_step_size_fraction,
            "preview_difficulty_ratio": self.preview_difficulty_ratio,
            "preview_exponential_decay": self.preview_exponential_decay,
            
            # Curriculum state
            "frontier_difficulties": self.frontier_difficulties,
            "available_difficulties": self.available_difficulties,
            "task_ratios": self.task_ratios.tolist(),
            "difficulty_ratios": {k: v.tolist() for k, v in self.difficulty_ratios.items()},
            "performance_history": {str(k): v for k, v in self.performance_history.items()},
            "max_difficulties": self.max_difficulties,
            "min_difficulties": self.min_difficulties,
            "supports_curriculum": self.supports_curriculum,
            
            # Task and dataset management
            "dataset_curriculum_types": [dt.value for dt in self.dataset_curriculum_types],
        }
    
    def load_state(self, state: dict):
        """Load state from serialization."""
        # Load core configuration parameters (with defaults for backward compatibility)
        self.num_tasks = state.get("num_tasks", self.num_tasks)
        self.initial_max_difficulty_fraction = state.get("initial_max_difficulty", self.initial_max_difficulty_fraction)
        self.advancement_threshold = state.get("advancement_threshold", self.advancement_threshold)
        self.new_difficulty_ratio = state.get("new_difficulty_ratio", self.new_difficulty_ratio)
        self.generalized_mean_power = state.get("generalized_mean_power", self.generalized_mean_power)
        self.performance_history_size = state.get("performance_history_size", self.performance_history_size)
        self.advancement_step_size_fraction = state.get("advancement_step_size_fraction", self.advancement_step_size_fraction)
        self.preview_difficulty_ratio = state.get("preview_difficulty_ratio", self.preview_difficulty_ratio)
        self.preview_exponential_decay = state.get("preview_exponential_decay", self.preview_exponential_decay)
        
        # Handle backward compatibility: convert active_difficulties to frontier_difficulties
        if "frontier_difficulties" in state:
            self.frontier_difficulties = state["frontier_difficulties"]
        elif "active_difficulties" in state:
            # Migrate from old format: frontier = max(active_difficulties)
            old_active = state["active_difficulties"]
            self.frontier_difficulties = {}
            for task_idx, active_diffs in old_active.items():
                if isinstance(active_diffs, list) and active_diffs:
                    self.frontier_difficulties[int(task_idx)] = max(active_diffs)
                else:
                    self.frontier_difficulties[int(task_idx)] = 0
            logging.info("Migrated active_difficulties to frontier_difficulties")
        
        self.available_difficulties = state.get("available_difficulties", {})
        self.task_ratios = torch.tensor(state["task_ratios"])
        self.difficulty_ratios = {k: torch.tensor(v) for k, v in state["difficulty_ratios"].items()}
        
        # Convert string keys back to tuples for performance history
        self.performance_history = {}
        for k, v in state["performance_history"].items():
            task_idx, difficulty = eval(k)  # Safe since we control the format
            self.performance_history[(task_idx, difficulty)] = v
        
        self.max_difficulties = state["max_difficulties"]
        self.min_difficulties = state.get("min_difficulties", {})
        self.supports_curriculum = state.get("supports_curriculum", {})
        
        # Load dataset curriculum types (with conversion from string values back to enum)
        if "dataset_curriculum_types" in state:
            from utils.enums import DATASET_CURRICULUM_TYPE
            self.dataset_curriculum_types = [DATASET_CURRICULUM_TYPE(dt_value) for dt_value in state["dataset_curriculum_types"]]
        else:
            self.dataset_curriculum_types = []
        
        logging.info("Loaded curriculum state")

    def update_task_ratios(self, task_ratios: torch.FloatTensor):
        """
        Update task ratios and difficulty ratios.
        Args:
            task_ratios (torch.FloatTensor): New task ratios to set.
        """
        if task_ratios is not None:
            self.task_ratios = task_ratios
        for task_idx in range(self.num_tasks):
            self.difficulty_ratios[task_idx] = self.compute_difficulty_ratios(task_idx)
        logging.debug("Updated task and difficulty ratios")

    def get_task_selector(self, optimize_last: bool) -> tuple[list[int], float]:
        """
        Get task selector slice and ratio magnitude for loss/task weighting.
        Args:
            optimize_last (bool): Whether to optimize the last task (usually validation).
        Returns:
            task_selector (slice): Slice selecting tasks to optimize.
            ratio_magnitude (float): Magnitude of selected task ratios.
        """
        task_selector = [i for i in range(self.num_tasks) if self.isTaskActive(i)]
        if not optimize_last and self.num_tasks - 1 in task_selector:
                task_selector.remove(self.num_tasks - 1) # Exclude last task
        true_task_selector, _ = self.compute_true_task_ratios(task_selector, self.task_ratios)
        ratio_magnitude = self.task_ratios[true_task_selector].sum().item()
        
        return task_selector, ratio_magnitude
    
    def compute_true_task_ratios(self, task_selector: list[int], ratios: torch.FloatTensor) -> tuple[list[int], torch.FloatTensor]:
        """
        Compute true task ratios considering by also considering standby tasks.
        Args:
            task_selector (list[int]): List of task indices to consider.
            ratios (torch.FloatTensor): Ratios corresponding to the selected tasks.
        Returns:
            true_task_selector (list[int]): List of task indices including standby tasks.
            true_ratios (torch.FloatTensor): Ratios for curriculum-enabled tasks.
        """
        true_task_selector = list()
        true_ratios = list()
        # true_ratios = ratios.clone()
        
        # Include standby tasks (preceding endgame tasks)
        for task_idx, ratio in zip(task_selector, ratios):
            if task_idx > 0 and self.dataset_curriculum_types[task_idx - 1] == DATASET_CURRICULUM_TYPE.STANDBY:
                true_task_selector.append(task_idx - 1)
                standby_ratio = self.standby_task_fraction * ratio
                true_ratios.append(standby_ratio)
                curriculum_ratio = (1 - self.standby_task_fraction) * ratio
                true_ratios.append(curriculum_ratio)
            else:
                true_ratios.append(ratio)
            true_task_selector.append(task_idx)
        
        return true_task_selector, torch.tensor(true_ratios, dtype=torch.float32)
