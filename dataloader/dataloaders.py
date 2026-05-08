# Create Dataloader from the dataframe
import logging
import random
from functools import partial
from pathlib import Path
from sys import maxsize
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from transformers import PreTrainedTokenizerFast

from dataloader.curriculum_manager import CurriculumManager
from dataloader.curriculum_sampler import CurriculumFixedRatioSampler
from dataloader.dataset_utils import FixedRatioSampler
from dataloader.datasets.curriculum_dataset import (
    CurriculumEvalDataset,
    CurriculumTrainDataset,
)
from dataloader.datasets.curriculum_number_dataset import (
    CurriculumNumberEvalDataset,
    CurriculumNumberTrainDataset,
    EfficientCurriculumNumberPromptBatch,
)
from dataloader.datasets.efficient_number_prompt_dataset import (
    EfficientNumberPromptBatch,
    EfficientNumberPromptEvalDataset,
)
from dataloader.datasets.efficient_prompt_dataset import (
    EfficientPromptBatch,
    EfficientPromptEvalDataset,
)
from dataloader.datasets.pretokenized_dataset import PretokenizedTrainDataset
from dataloader.datasets.pretokenized_number_dataset import (
    PretokenizedNumberTrainDataset,
)
from utils.enums import DATASET_CURRICULUM_TYPE, DATASET_TYPE
from utils.util_funcs import get_num_token_mask


def _pad_left(sequence: List[torch.Tensor], batch_first=True, padding_value: int=0):
    """
    Pad a list of sequences to the maximum length in the batch. The padding is applied to the left side of the sequence.
    Args:
        sequence (List[torch.Tensor]): The list of sequences to pad
        batch_first (bool, optional): Whether the sequences are batch first. Defaults to True.
        padding_value (int, optional): The value to pad with. Defaults to 0.
    Returns:
        torch.Tensor: The padded sequences
    """
    max_len = max(len(seq) for seq in sequence)
    return torch.stack([
        pad(seq,(max_len - len(seq), 0), value=padding_value) if batch_first
        else pad(seq, (0, max_len - len(seq)), value=padding_value)
        for seq in sequence
    ]) 
def generate_collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int, num_token_ids: Optional[torch.LongTensor]) -> EfficientPromptBatch | EfficientNumberPromptBatch | EfficientCurriculumNumberPromptBatch:
    """
    Collate function for the EfficientPromptDataset. Pads the input_ids and attention_masks to the maximum length in the batch.
    Absolute position ids are calculated based on the attention_masks.

    Args:
        batch (List[Dict[str, torch.Tensor]]): The batch to collate
        pad_token_id (int): The id of the padding token
        num_token_ids (Optional[torch.LongTensor]): The id of the number token
    Returns:
        Dict[str, torch.Tensor]: The collated batch
    """
    # Separate input_ids and attention_masks
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    answers = [item['labels'] for item in batch]
    answer_lengths = torch.tensor([item['answer_length'] for item in batch])
    orig_prompts = [item['orig_prompt'] for item in batch]
    orig_answers = [item['orig_answer'] for item in batch]
    difficulties: torch.LongTensor | None = torch.tensor([item['difficulty'] for item in batch]) if 'difficulty' in batch[0] else None
    
    # Pad sequences
    input_ids_padded: torch.LongTensor = _pad_left(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks_padded = _pad_left(attention_masks, batch_first=True, padding_value=0)
    labels_padded: torch.LongTensor = pad_sequence(answers, batch_first=True, padding_value=-100) # right padding for generate

    position_ids = attention_masks_padded.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_masks_padded == 0, 1)
    
    if "input_numbers" in batch[0] and num_token_ids is not None:
        input_numbers = torch.zeros_like(input_ids_padded, dtype=torch.float64)
        input_numbers[get_num_token_mask(input_ids_padded, num_token_ids)] = torch.stack([i for item in batch for i in item["input_numbers"]])
        label_numbers = torch.zeros_like(labels_padded, dtype=torch.float64)
        if "label_numbers" in batch[0]:
            if batch[0]["label_numbers"].numel():
                label_numbers[get_num_token_mask(labels_padded, num_token_ids)] = torch.stack([i for item in batch for i in item["label_numbers"]])
            return {
                'input_ids': input_ids_padded,
                'attention_mask': attention_masks_padded,
                'labels': labels_padded,
                'position_ids': position_ids,
                'answer_lengths': answer_lengths,
                "input_numbers": input_numbers,
                "label_numbers": label_numbers,
                "orig_prompts": orig_prompts,
                "orig_answers": orig_answers,
                'max_seq_length': position_ids.max().item() + 1,
                'difficulties': difficulties
            }
        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_masks_padded,
            'labels': labels_padded,
            'position_ids': position_ids,
            'answer_lengths': answer_lengths,
            "input_numbers": input_numbers,
            "orig_prompts": orig_prompts,
            "orig_answers": orig_answers,
            'max_seq_length': position_ids.max().item() + 1,
            'difficulties': difficulties
        }
    else:
        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_masks_padded,
            'labels': labels_padded,
            'position_ids': position_ids,
            'answer_lengths': answer_lengths,
            "orig_prompts": orig_prompts,
            "orig_answers": orig_answers,
            'max_seq_length': position_ids.max().item() + 1,
            'difficulties': difficulties
        }

def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int, num_token_ids: Optional[torch.LongTensor]=None) -> EfficientPromptBatch | EfficientNumberPromptBatch | EfficientCurriculumNumberPromptBatch:
    """
    Collate function for the EfficientPromptAnswerDataset. Pads the input_ids and attention_masks to the maximum length in the batch.
    Absolute position ids are calculated based on the attention_masks.

    Args:
        batch (List[Dict[str, torch.Tensor]]): The batch to collate
        pad_token_id (int): The id of the padding token
        num_token_ids (Optional[torch.LongTensor]): The id of the number token
    Returns:
        Dict[str, torch.Tensor]: The collated batch
    """
    # Separate input_ids and attention_masks
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    answer_lengths = [item['answer_length'] for item in batch]
    difficulties: torch.LongTensor | None = torch.tensor([item['difficulty'] for item in batch]) if 'difficulty' in batch[0] else None
    
    # Pad sequences
    input_ids_padded: torch.LongTensor = _pad_left(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks_padded = _pad_left(attention_masks, batch_first=True, padding_value=0)
    labels_padded = _pad_left(labels, batch_first=True, padding_value=-100)

    # create position_ids on the fly for batch generation
    # Similar to transformers.models.gpt2.GPT2PreTrainedModel.prepare_inputs_for_generation
    position_ids = attention_masks_padded.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_masks_padded == 0, 1)
    
    if "input_numbers" in batch[0] and num_token_ids is not None:
        input_numbers = torch.zeros_like(input_ids_padded, dtype=torch.float64)
        input_numbers[get_num_token_mask(input_ids_padded, num_token_ids)] = torch.stack([i for item in batch for i in item["input_numbers"]])
        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_masks_padded,
            'labels': labels_padded,
            'position_ids': position_ids,
            'answer_lengths': answer_lengths,
            'input_numbers': input_numbers,
            'max_seq_length': position_ids.max().item() + 1,
            'difficulties': difficulties
        }
    else:
        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_masks_padded,
            'labels': labels_padded,
            'position_ids': position_ids,
            'answer_lengths': answer_lengths,
            'max_seq_length': position_ids.max().item() + 1,
            'difficulties': difficulties
        }
    
def seq_pack_collate_fn(batch, collate_fun):
    ret = collate_fun(batch)
    position_ids = ret["position_ids"].flatten()
    indices_q = torch.arange(position_ids.size(0), dtype=torch.int32)

    cu_seq_lens = torch.cat(
        (
            indices_q[position_ids == 0],
            torch.tensor(position_ids.size(), dtype=torch.int32),
        )
    )

    ret["cu_seq_lens"] = cu_seq_lens
    ret["max_seq_length"] = position_ids.max().item() + 1
    return ret

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_train_loader(
        dataset_type: DATASET_TYPE,
        dataset_curriculum_types: list[DATASET_CURRICULUM_TYPE],
        train_set_paths: Sequence[Path],
        tokenizer: PreTrainedTokenizerFast,
        batch_size: int,
        effective_batch_size: int,
        context_length: int,
        num_workers: int=0,
        train_cached_paths: list[Path|None]=[],
        num_tokens: Optional[Sequence[int]]=None,
        unique_samples: int=maxsize,
        seed: int=42,
        shuffle=True,
        cache_base_path: Optional[Path]=None,
        tokenizer_path: Optional[Path]=None,
        train_set_ratios: Optional[list[float]]=None,
        curriculum_manager: Optional[CurriculumManager]=None,
        difficulty_column="difficulty") -> DataLoader:
    kwargs = {}
    collate_fun: Callable
    match dataset_type:
        case "efficient_prompt":
            raise NotImplementedError("efficient_prompt dataset type is no longer supported, please use pretokenized instead")
        case "efficient_number_prompt" | "efficient_number_prompt_pos":
            raise NotImplementedError("efficient_number_prompt dataset type is no longer supported, please use pretokenized_number instead")
        case "pretokenized":
            assert cache_base_path is not None, "cache_base_path must be provided for pretokenized datasets"
            if curriculum_manager is not None:
                train_dataset = CurriculumTrainDataset(
                    dataset_paths=train_set_paths,
                    dataset_curriculum_types=dataset_curriculum_types,
                    tokenizer=tokenizer,
                    cache_base_path=cache_base_path,
                    cached_paths=train_cached_paths,
                    num_tokens=num_tokens,
                    context_length=context_length,
                    unique_samples=unique_samples,
                    shuffle=shuffle,
                    tokenizer_path=tokenizer_path,
                    difficulty_column=difficulty_column
                )
            else:
                assert train_set_ratios is not None, "train_set_ratios must be provided for pretokenized datasets"
                train_dataset = PretokenizedTrainDataset(train_set_paths, tokenizer, cache_base_path=cache_base_path, cached_paths=train_cached_paths, num_tokens=num_tokens, context_length=context_length, unique_samples=unique_samples, shuffle=shuffle, tokenizer_path=tokenizer_path)
                kwargs["sampler"] = FixedRatioSampler(train_dataset.data.lengths, train_set_ratios, context_length, effective_batch_size)
            collate_fun = default_collate
        case "pretokenized_number" | "pretokenized_number_pos":
            assert cache_base_path is not None, "cache_base_path must be provided for pretokenized datasets"
            if curriculum_manager is not None:
                # Use curriculum learning - CurriculumNumberTrainDataset handles both curriculum and non-curriculum datasets
                train_dataset = CurriculumNumberTrainDataset(
                    dataset_paths=train_set_paths,
                    dataset_curriculum_types=dataset_curriculum_types,
                    tokenizer=tokenizer,
                    cache_base_path=cache_base_path,
                    cached_paths=train_cached_paths,
                    num_tokens=num_tokens,
                    context_length=context_length,
                    unique_samples=unique_samples,
                    shuffle=shuffle,
                    tokenizer_path=tokenizer_path,
                    allow_negative=dataset_type == "pretokenized_number",
                    difficulty_column=difficulty_column
                )
            else:
                assert train_set_ratios is not None, "train_set_ratios must be provided for pretokenized datasets"
                train_dataset = PretokenizedNumberTrainDataset(train_set_paths, tokenizer, cache_base_path=cache_base_path, cached_paths=train_cached_paths, num_tokens=num_tokens, context_length=context_length, unique_samples=unique_samples, shuffle=shuffle, tokenizer_path=tokenizer_path, allow_negative=dataset_type == "pretokenized_number")
                kwargs["sampler"] = FixedRatioSampler(train_dataset.data.lengths, train_set_ratios, context_length, effective_batch_size)
            collate_fun = default_collate
        case _:
            raise NotImplementedError(f"Invalid dataset type {dataset_type}")
    if isinstance(train_dataset, (CurriculumTrainDataset, CurriculumNumberTrainDataset)) and curriculum_manager is not None:
        # Synchronize with datasets (handles both curriculum and non-curriculum datasets)
        train_set_ratios_tensor: torch.FloatTensor = torch.tensor(train_set_ratios)
        curriculum_manager.synchronize_with_datasets(train_dataset, train_set_ratios_tensor)
        logging.info(curriculum_manager.get_summary())
        
        # Create curriculum sampler
        # Get all sampling difficulties from curriculum manager
        sampling_difficulties = {}
        for task_idx in range(curriculum_manager.num_tasks):
            sampling_difficulties[task_idx] = curriculum_manager.get_all_sampling_difficulties(task_idx)
        
        kwargs["sampler"] = CurriculumFixedRatioSampler(
            dataset=train_dataset,
            task_ratios=curriculum_manager.task_ratios,
            difficulty_ratios=curriculum_manager.difficulty_ratios,
            sampling_difficulties=sampling_difficulties,
            context_length=context_length,
            batch_size=effective_batch_size,
            curriculum_manager=curriculum_manager,
        )
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        generator=g,
        worker_init_fn=seed_worker,
        collate_fn=partial(seq_pack_collate_fn, collate_fun=collate_fun),
        **kwargs
    )
    return train_loader

def get_eval_loader(
        dataset_type: DATASET_TYPE,
        val_set_path: Path,
        tokenizer: PreTrainedTokenizerFast,
        batch_size: int,
        context_length: int,
        seed: int,
        num_workers: int,
        val_cached_path: Optional[Path]=None,
        num_tokens: int=-1,
        unique_samples: int=maxsize,
        shuffle=False,
        cache_base_path=None,
        difficulty_column="difficulty",
        additional_columns: List[str]=[],
        load_all_columns = False,
        dataset_curriculum_type: Optional[DATASET_CURRICULUM_TYPE] = DATASET_CURRICULUM_TYPE.CURRICULUM) -> DataLoader:
    assert isinstance(tokenizer.pad_token_id, int), "Tokenizer must have an integer pad token id"
    number_tokens = tokenizer.init_kwargs["model_specific_special_tokens"].get("num_token", None)
    if number_tokens is None:
        num_token_ids = None
    else:
        if not isinstance(number_tokens, list):
            number_tokens = [number_tokens]
        num_token_ids: torch.LongTensor = torch.tensor([tokenizer.convert_tokens_to_ids(token) for token in number_tokens])
    collate_fun: Callable = partial(generate_collate_fn, pad_token_id=tokenizer.pad_token_id, num_token_ids=num_token_ids)
    kwargs = {}
    match dataset_type:
        case "efficient_prompt":
            test_dataset = EfficientPromptEvalDataset(val_set_path, tokenizer, unique_samples=unique_samples, shuffle=shuffle,additional_columns=additional_columns, load_all_columns=load_all_columns)
            collate_fun = partial(generate_collate_fn, pad_token_id=tokenizer.pad_token_id, num_token_ids=None)
        case "efficient_number_prompt" | "efficient_number_prompt_pos":
            test_dataset = EfficientNumberPromptEvalDataset(val_set_path, tokenizer, unique_samples=unique_samples, shuffle=shuffle, allow_negative=dataset_type == "efficient_number_prompt",additional_columns=additional_columns, load_all_columns=load_all_columns)
            collate_fun = partial(generate_collate_fn, pad_token_id=tokenizer.pad_token_id, num_token_ids=num_token_ids)
        case "pretokenized":
            assert cache_base_path is not None, "cache_base_path must be provided for pretokenized datasets"
            test_dataset = PretokenizedTrainDataset([val_set_path], tokenizer, cache_base_path=cache_base_path, cached_paths=[val_cached_path], num_tokens=[num_tokens], context_length=context_length, unique_samples=unique_samples, shuffle=shuffle, rand_offset=False)
            collate_fun = partial(seq_pack_collate_fn, collate_fun=default_collate)
        case "pretokenized_number" | "pretokenized_number_pos":
            assert cache_base_path is not None, "cache_base_path must be provided for pretokenized datasets"
            test_dataset = PretokenizedNumberTrainDataset([val_set_path], tokenizer, cache_base_path=cache_base_path, cached_paths=[val_cached_path], num_tokens=[num_tokens], context_length=context_length, unique_samples=unique_samples, shuffle=shuffle, rand_offset=False, allow_negative=dataset_type == "pretokenized_number")
            collate_fun = partial(seq_pack_collate_fn, collate_fun=default_collate)
        case "curriculum_number" | "curriculum_number_pos":
            test_dataset = CurriculumNumberEvalDataset(
                tokenizer=tokenizer,
                dataset_path=val_set_path,
                unique_samples=unique_samples,
                shuffle=shuffle,
                allow_negative=dataset_type == "curriculum_number",
                difficulty_column=difficulty_column,
                dataset_curriculum_type=dataset_curriculum_type or DATASET_CURRICULUM_TYPE.CURRICULUM
            )
        case "curriculum":
            test_dataset = CurriculumEvalDataset(
                tokenizer=tokenizer,
                dataset_path=val_set_path,
                unique_samples=unique_samples,
                shuffle=shuffle,
                difficulty_column=difficulty_column,
                dataset_curriculum_type=dataset_curriculum_type or DATASET_CURRICULUM_TYPE.CURRICULUM
            )
        case _:
            raise NotImplementedError(f"Invalid dataset type {dataset_type}")
    if isinstance(test_dataset, (CurriculumNumberEvalDataset, CurriculumEvalDataset)):
        # Create curriculum sampler
        task_ratios: torch.FloatTensor = torch.tensor([1.])
        available_difficulties = test_dataset.get_available_difficulties(None)
        available_difficulties_tensor: torch.LongTensor = torch.tensor(available_difficulties, dtype=torch.long)
        difficulty_ratios: torch.FloatTensor = torch.ones(len(available_difficulties), dtype=torch.float32) / len(available_difficulties)
        kwargs["sampler"] = CurriculumFixedRatioSampler(
            dataset=test_dataset,
            task_ratios=task_ratios,
            difficulty_ratios={0: difficulty_ratios},
            sampling_difficulties={0: available_difficulties_tensor},
            context_length=1,
            batch_size=batch_size,
            curriculum_manager=None,  # No curriculum manager needed for evaluation
        )
    g = torch.Generator()
    g.manual_seed(seed)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fun,
        num_workers=num_workers,
        generator=g,
        worker_init_fn=seed_worker,
        **kwargs
    )
    return test_loader