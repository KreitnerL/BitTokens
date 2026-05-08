import logging
from abc import ABC, abstractmethod
from functools import partial
from math import ceil
from multiprocessing import Pool
from sys import maxsize
from typing import Generator, Literal, Optional, override

import numpy as np
import torch
from torch.utils.data import Sampler

from dataloader.datasets.efficient_prompt_dataset import EfficientPromptTrainDataset


def _get_length(dataset, i):
    return i, len(dataset[i]['input_ids'])

def create_length_based_sampler(dataset: EfficientPromptTrainDataset, batch_size: int, num_processes=16, shuffle=False, unique_samples=maxsize, mode: Literal["fast", "exact"]="fast") -> Sampler:
    """
    Create a sampler that samples indices based on the length of the input_ids.
    The indices are sorted based on the length of the input_ids and then sorted in similar length batches of batch_size.
    
    Args:
        dataset (EfficientPromptAnswerDataset): The dataset to sample from
        batch_size (int): The batch size
        num_processes (int, optional): The number of processes to use. Defaults to 48.
        shuffle (bool, optional): Whether to shuffle the batches. Defaults to False.
        unique_samples (int, optional): The number of unique samples to use. Defaults to maxsize.
    """
    if unique_samples > len(dataset):
        logging.warning(f"unique_samples ({unique_samples}) is greater than the length of the dataset ({len(dataset)}). Setting unique_samples to the length of the dataset.")
    if shuffle:
        dataset.df = dataset.df.sample(n = min(unique_samples, len(dataset)), random_state=42).reset_index()
    else:
        dataset.df = dataset.df[:unique_samples].reset_index()

    if mode=="exact":
        num_processes = max(num_processes, 1)
        logging.info(f"Using {num_processes} processes to compute lengths...")
        with Pool(processes=num_processes) as pool:
            results = pool.map(partial(_get_length, dataset), range(len(dataset)), chunksize=len(dataset) // num_processes)
        idx_len_map = dict(results)
        logging.info("Done.")
        sorted_idx_len_map = dict(sorted(idx_len_map.items(), key=lambda item: item[1]))
        index_list = np.array(list(sorted_idx_len_map.keys()))
    elif mode=="fast":
        # Sort rows by sum of num_digits column and num_decimals column
        dataset.df["length"] = dataset.df["prompt"].apply(lambda x: len(x))
        # Sort the DataFrame by the 'sum_columns' column
        df_sorted = dataset.df.sort_values(by='length')
        dataset.df = dataset.df.drop(columns=['length'])
        # Retrieve the sorted indices
        index_list = np.array(df_sorted.index)
    else:
        raise ValueError(f"Invalid mode {mode}")

    # Select all indices except the last batch
    num_batches = len(dataset) // batch_size
    index_list, remainder = index_list[:len(dataset)-(len(dataset) % batch_size)], index_list[len(dataset)-(len(dataset) % batch_size):]
    index_list = index_list.reshape(num_batches, batch_size)
    # Shuffle the index_list along the first axis
    if shuffle:
        np.random.shuffle(index_list)
    # Flatten the index_list
    index_list = index_list.flatten()
    # Append the remainder to the end
    index_list = np.append(index_list, remainder)
    return index_list

def _create_sampler(start, end) -> Generator[int, None, None]:
    indices = np.arange(start, end)
    while True:
        np.random.shuffle(indices)
        for idx in indices:
            yield idx

class RatioSampler(Sampler, ABC):
    @staticmethod
    def _compute_proportions(ratios: torch.FloatTensor, batch_size: int, remaining_samples: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        proportions = ratios * batch_size
        if remaining_samples is None:
            return proportions
        return torch.nn.functional.pad(remaining_samples, (0, len(proportions)-len(remaining_samples))) + proportions
    @abstractmethod
    def get_task_ratios(self) -> torch.FloatTensor:
        """
        Returns the percentage of samples to use for each task (sums to 1)
        """
        pass

    @abstractmethod
    def update_task_ratio(self, task_selector: slice | list[int], ratios: torch.FloatTensor):
        """
        Update the ratios for the specified tasks.
        Args:
            task_selector (slice): The task index or slice to update.
            ratios (torch.FloatTensor): The new ratios for the task.
        """
        pass

class FixedRatioSampler(RatioSampler):
    def __init__(self, cum_sum_lengths: np.ndarray, ratios: list[float], context_length: int, batch_size: int):
        """
        Args:
            cum_sum_lengths (np.ndarray): Cumulative sum of lengths of the datasets in tokens without the last context length.
            ratios (np.ndarray): Ratios of the datasets.
            context_length (int): Context length.
            batch_size (int): Batch size.
        """
        self.cum_sum_lengths = cum_sum_lengths
        cum_sum_lengths_0 = [0, *cum_sum_lengths]
        # self.dataset_lengths = np.array([len(d)//context_length-1 for d in datasets])
        self.ratios: torch.FloatTensor = torch.tensor(ratios) / sum(ratios)
        self.batch_size = batch_size
        self.proportions: torch.FloatTensor = torch.ones_like(self.ratios)
        self.context_length = context_length
        self.dataset_samplers: list[Generator] = [_create_sampler(ceil(cum_sum_lengths_0[i]/context_length),cum_sum_lengths_0[i+1]//context_length-1) for i in range(len(cum_sum_lengths))]

    @override
    def __iter__(self):
        while True:
            if (self.proportions < 1).all():
                self.proportions = FixedRatioSampler._compute_proportions(self.ratios, self.batch_size, self.proportions)
            dataset_idx = self.proportions.argmax()
            idx = next(self.dataset_samplers[dataset_idx])
            self.proportions[dataset_idx] -= 1
            yield idx

    def __len__(self):
        # Return the total number of context windows in all datasets without the last context_length
        return self.cum_sum_lengths[-1]//self.context_length-1
    
    @override
    def get_task_ratios(self) -> torch.FloatTensor:
        return self.ratios

    @override
    def update_task_ratio(self, task_selector: slice | list[int], ratios: torch.FloatTensor):
        self.ratios[task_selector] = ratios