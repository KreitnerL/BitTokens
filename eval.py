import logging
import os
from pathlib import Path
from sys import maxsize
from typing import Iterator, Optional, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from data_generation.data_gen_utils import get_strat_params, plot_accuracy_distribution
from dataloader.dataloaders import get_eval_loader
from dataloader.datasets.pretokenized_dataset import _PretokenizedDataset
from networks.models import get_model
from networks.number_embedding_modules.abc_embedding import ABCEmbedding
from utils.enums import (
    DATASET_CURRICULUM_TYPE,
    CausalLMOutputWithCrossAttentionsAndNumbers,
    EvalBatch,
    EvalOutput,
    GenerateOutputWithNumbers,
    SamplesDict,
)
from utils.eval_argument_parser import EvalArgumentParser
from utils.metrics import MetricFunction
from utils.util_funcs import (
    check_intervals_all_true,
    get_num_token_mask,
    parse_numbers_from_text,
    print_and_save_arguments,
    replace_in_order,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def evaluate(
        model: GPT2LMHeadModel,
        test_loader: DataLoader,
        metric_func: MetricFunction,
        tokenizer: PreTrainedTokenizerFast,
        max_eval_steps=maxsize,
        use_tqdm=True,
        leave_tqdm=False,
        return_predictions=False,
        return_samples=False,
        post_fix: str = "",
        dtype: torch.dtype | str =torch.float32,
        additional_metric_funcs: Optional[list[MetricFunction]] = None
    ) -> EvalOutput:
    """
    Evaluate the model on the test set
    Args:
        model (GPT2LMHeadModel): The model to evaluate
        tokenizer (PreTrainedTokenizerFast): The tokenizer
        test_loader (DataLoader): The test loader
        max_eval_steps (int): Maximum number of evaluation steps. Defaults to maxsize.
        use_tqdm (bool): Whether to use tqdm. Defaults to True.
        return_predictions (bool): Whether to return predictions. Defaults to False.
    Returns:
        Tuple[np.ndarray, list[str]]: The correct predictions and the generated predictions    
    """
    model.eval()
    gen_predictions = list()
    acc_list: list[float] = list()
    per_sample_acc_list: list[float] = list()
    per_sample_correct_list: list[bool] = list()
    gen_losses: list[float] = list()
    text_gen_losses: list[float] = list()
    numeric_text_gen_losses: list[float] = list()

    # Create a list of all numeric tokens. These can be tokenizer.num_token_id or any number-like token in the tokenizer
    numeric_token_ids: list[int] = []
    if hasattr(model, "num_token_ids"):
        numeric_token_ids.extend(model.num_token_ids.cpu().tolist()) # pyright: ignore[reportCallIssue]
    else:
        # Try to find number-like tokens in the tokenizer
        for token_id in range(len(tokenizer.vocab)):
            token: str = tokenizer.decode([token_id])
            try:
                if float(token)+1 > float(token):
                    numeric_token_ids.append(token_id)
            except ValueError:
                continue

    num_gen_losses: list[float] = list()
    additional_metrics: dict[str, list[float]] = dict()
    per_sample_additional_metrics: dict[str, list] = dict()
    num_preds: list[str] = list()
    num_trues: list[str] = list()
    additional_train_signals: dict[str, list[float]] = dict()
    per_difficulty_acc: dict[int, list] = dict()
    iterator = tqdm(test_loader, desc="Evaluating"+post_fix, leave=leave_tqdm, total=min(len(test_loader), max_eval_steps)) if use_tqdm else test_loader
    for step, batch in enumerate(iterator, start=1):
        batch: EvalBatch
        per_sample_num_loss = list()
        prompt_length = batch["input_ids"].shape[1]
        answer_length = batch["labels"].shape[1]
        labels: torch.LongTensor = batch["labels"]
        autocast_dtype = dtype if isinstance(dtype, torch.dtype) else torch.__dict__.get(dtype, torch.float32)
        with torch.amp.autocast_mode.autocast(device_type=str(model.device), dtype=autocast_dtype, enabled=autocast_dtype != torch.float32):
            with torch.no_grad():
                kwargs = dict()
                if "input_numbers" in batch and batch["input_numbers"] is not None:
                    kwargs["numbers"] = batch["input_numbers"].to(model.device)
                if "cu_seq_lens" in batch and batch["cu_seq_lens"] is not None:
                    kwargs["cu_seq_lens"] = batch["cu_seq_lens"].to(device=model.device)
                kwargs["max_seq_length"] = batch.get("max_seq_length", batch["position_ids"].max().item())
                label_mask = labels != -100
                if hasattr(model, "num_token_ids"):
                    num_token_ids: torch.LongTensor = model.num_token_ids.cpu()
                    label_num_mask: torch.BoolTensor = get_num_token_mask(labels, num_token_ids)
                else:
                    label_num_mask = torch.ones_like(labels, dtype=torch.bool)

                if "attention_mask" in batch and batch["attention_mask"] is None:
                    pred_selector = slice(None,-1)
                    labels = labels[:,1:]
                    if batch["input_numbers"]:
                        label_numbers: torch.DoubleTensor = batch["input_numbers"][:,1:]
                        batch["label_numbers"] = label_numbers
                    label_num_mask = label_num_mask[:,1:]

                num_true_padded: torch.DoubleTensor = None
                if "label_numbers" in batch and batch["label_numbers"] is not None:
                    num_true_padded = batch["label_numbers"].to(device=model.device)
                    num_true = batch["label_numbers"][label_num_mask]
                elif hasattr(model, "num_token_ids") and (orig_answers:=batch.get("orig_answers")) is not None:
                    num_true = [torch.tensor(parse_numbers_from_text(label_text, split_large_numbers=False), dtype=torch.float64) for label_text in orig_answers]
                else:
                    num_true = [
                        torch.tensor(parse_numbers_from_text(cast(str,tokenizer.decode(label)), split_large_numbers=False), dtype=torch.float64)
                        for label in torch.where(labels == -100, torch.tensor(tokenizer.pad_token_id), labels)
                    ]
                label_num_mask = label_num_mask.to(device=model.device)
                if batch.get("orig_answers") is not None:
                    outputs: GenerateOutputWithNumbers = model.generate( # pyright: ignore[reportRedeclaration]
                        input_ids=batch["input_ids"].to(device=model.device),
                        max_new_tokens=answer_length,
                        do_sample=False,
                        num_beams=1,
                        output_scores=True,
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                        attention_mask=batch["attention_mask"].to(model.device),
                        position_ids=batch["position_ids"].to(model.device),
                        **kwargs
                    )
                    assert outputs.hidden_states is not None, "Model output must contain hidden states for number loss computation."
                    if outputs.sequences.shape[1] == (prompt_length+answer_length):
                        preds = outputs.sequences.cpu()[:, prompt_length:]
                    else:
                        assert isinstance(tokenizer.pad_token_id, int), "Tokenizer must have a pad token id"
                        preds: torch.LongTensor = torch.nn.functional.pad(
                            outputs.sequences,
                            pad=(0,prompt_length+answer_length-outputs.sequences.shape[1]),
                            value=tokenizer.pad_token_id
                        )[:,prompt_length:].cpu()
                    # Compute validation loss
                    logits: torch.FloatTensor = torch.stack([
                        *outputs.scores,
                        *torch.zeros((answer_length-len(outputs.scores), *outputs.scores[0].shape),device=outputs.scores[0].device)
                    ], dim=1).detach().cpu()
                    loss = torch.nn.functional.cross_entropy(logits[label_mask], labels[label_mask]).item()
                    pred_selector = slice(prompt_length, None)
                    if "label_numbers" in batch and batch["label_numbers"]  is not None and label_num_mask.any() and hasattr(model, "num_embedding_module"):
                        label_ids =  torch.where(labels < 0, torch.tensor(tokenizer.pad_token_id), labels)
                        padded_num_encoding = model.get_padded_num_encoding(label_ids.to(model.device), num_true_padded, label_num_mask) # pyright: ignore[reportCallIssue]
                        num_embedding_module: ABCEmbedding = model.num_embedding_module
                        # `generate(..., output_hidden_states=True)` returns:
                        #   outputs.hidden_states: tuple[generation_step] of tuple[layer] of Tensor[B, step_len, E]
                        # where prefill has step_len=prompt_len and each decode step has step_len=1.
                        # For number-loss, we need a single Tensor[B, answer_len, E] aligned with `label_num_mask`.
                        per_step_last_layer = [hs_step[-1] for hs_step in outputs.hidden_states[1:]]  # skip prefill
                        if len(per_step_last_layer) == 0:
                            gen_last_layer = torch.zeros(
                                (labels.shape[0], 0, padded_num_encoding.shape[-1]),
                                device=model.device,
                                dtype=padded_num_encoding.dtype,
                            )
                        else:
                            gen_last_layer = torch.cat(per_step_last_layer, dim=1)

                        if gen_last_layer.shape[1] < answer_length:
                            gen_last_layer = torch.nn.functional.pad(
                                gen_last_layer, (0, 0, 0, answer_length - gen_last_layer.shape[1]), value=0
                            )
                        else:
                            gen_last_layer = gen_last_layer[:, -answer_length:, :]

                        # Make `compute_num_loss` see a forward-like hidden_states tuple.
                        outputs.hidden_states = (gen_last_layer,)

                        eval_signals: dict | torch.FloatTensor = num_embedding_module.compute_num_loss(
                            outputs,
                            padded_num_encoding,
                            label_num_mask,
                            num_true_padded,
                            hidden_states_slice=slice(None),
                        )  # pyright: ignore[reportArgumentType]
                        if isinstance(eval_signals, dict):
                            num_loss = eval_signals.pop("num_loss")
                            additional_train_signals.setdefault("num_loss_per_frequency", []).append(eval_signals.pop("num_loss_per_frequency").cpu().numpy())
                            if "prob_use_linear" in eval_signals:
                                tokens: torch.LongTensor = preds[:,pred_selector]
                                pred_num_mask = get_num_token_mask(tokens, num_token_ids)
                                additional_train_signals.setdefault("prob_use_linear", []).append(eval_signals.pop("prob_use_linear")[pred_num_mask].mean().item())
                            for k, v in eval_signals.items():
                                additional_train_signals.setdefault(k, []).append(v)
                        else: # legacy support
                            num_loss = eval_signals
                        per_sample_num_loss.extend(num_loss.sum(dim=1).cpu().numpy().tolist())
                        num_loss = num_loss.sum().item() / labels.numel()
                        num_gen_losses.append(num_loss)
                    else:
                        num_gen_losses.append(0)
                else:
                    outputs: CausalLMOutputWithCrossAttentionsAndNumbers = model.forward( 
                        input_ids=batch["input_ids"].to(model.device), # pyright: ignore[reportArgumentType]
                        attention_mask=batch["attention_mask"].to(model.device), # pyright: ignore[reportArgumentType]
                        labels=batch["labels"].to(model.device), # pyright: ignore[reportArgumentType]
                        position_ids=batch["position_ids"].to(model.device), # pyright: ignore[reportArgumentType]
                        **kwargs
                    )
                    assert outputs.loss is not None, "Model output must contain loss for evaluation."
                    answer_length-=1
                    pred_selector = slice(None,-1)
                    preds = cast(torch.LongTensor, outputs.logits[:,:-1].argmax(dim=-1).cpu())
                    logits = cast(torch.FloatTensor, outputs.logits[:,:-1].cpu())
                    labels = labels[:,1:]
                    loss = outputs.loss.mean().item()
                    num_gen_losses.append(getattr(outputs, "num_loss", torch.zeros(1)).mean().item())
        preds_text: list[str] = None
        if "label_numbers" in batch and batch["label_numbers"] is not None and outputs.numbers is not None and outputs.numbers.numel() > 0:
            num_pred = outputs.numbers[:,pred_selector].cpu()
            label_num_mask = label_num_mask.cpu()
            num_pred = torch.nn.functional.pad(num_pred,pad=(0,answer_length-num_pred.shape[1]),value=np.nan)[label_num_mask]
            # num_pred and num_true are flattened tensors of shape (num_numbers,)

            # num_pred_per_sample is a list of length (B,) of DoubleTensors, where each inner tesnor contains the predicted numbers for a sample
            num_pred_padded = torch.zeros_like(label_num_mask, dtype=torch.double)
            num_pred_padded[label_num_mask] = num_pred
            num_pred_per_sample: list[torch.DoubleTensor] = [num_pred_padded[i][label_num_mask[i]].cpu() for i in range(num_pred_padded.shape[0])]

            num_true_padded = torch.zeros_like(label_num_mask, dtype=torch.double)
            num_true_padded[label_num_mask] = num_true
            num_true_per_sample: list[torch.DoubleTensor] = [num_true_padded[i][label_num_mask[i]].cpu() for i in range(num_true_padded.shape[0])]
        else:
            preds_list = [preds[i].tolist() for i in range(len(preds))]
            if hasattr(outputs, "numbers") and outputs.numbers is not None:
                # Special case where numbers are parsed but not completely. We first add the numbers to the text and then parse them back
                preds_text = tokenizer.batch_decode(preds_list)
                num_pred = outputs.numbers[:,pred_selector].cpu()
                num_pred = torch.nn.functional.pad(num_pred,pad=(0,answer_length-num_pred.shape[1]),value=0)[get_num_token_mask(preds, num_token_ids)]
                num_token = tokenizer.init_kwargs["model_specific_special_tokens"]["num_token"]
                num_pred_iter = iter(num_pred)
                # For each pred_text, count the number of numbers in the text
                num_pred_per_sample = [[f"{next(num_pred_iter).item():.14f}" for _ in range(pt.count(num_token))] for pt in preds_text]
                preds_text = [replace_in_order(t, [num_token], [ff.rstrip("0") for ff in f if "." in ff]) for t, f in zip(preds_text, num_pred_per_sample)]
            else:
                preds_text = tokenizer.batch_decode(preds_list)
            num_pred_list = list()
            num_pred_per_sample = []
            for pred_text, true_nums in zip(preds_text, num_true):
                pred_nums: torch.DoubleTensor = torch.tensor(parse_numbers_from_text(pred_text, split_large_numbers=False), dtype=torch.float64)
                num_pred_per_sample.append(pred_nums)
                pred_nums = torch.nn.functional.pad(pred_nums[:true_nums.numel()], (0,max(0,true_nums.numel()-pred_nums.numel())), value=0)
                num_pred_list.append(pred_nums)
            num_pred: torch.DoubleTensor = torch.concatenate(num_pred_list)
            if isinstance(num_true, list):
                num_true_per_sample = num_true.copy()
                num_true: torch.DoubleTensor = torch.concatenate(num_true)
            else:
                raise ValueError("This should no happen! num_true must be a list if label_numbers is not None")

        num_preds.extend([str(s.tolist()) for s in num_pred_per_sample])
        num_trues.extend([str(s.tolist()) for s in num_true_per_sample])

        acc, gen_correct, sample_acc = metric_func.__call__(
            y_pred=preds,
            logits=logits,
            y_true=labels,
            num_pred=num_pred,
            num_true=num_true
        )
        label_num_mask = label_num_mask.cpu()
        if gen_correct.numel() > label_num_mask.shape[0]:
            # Multiple numbers per sample, e.g. for arithmetic tasks
            # We need to assign the numbers to the samples and then compute the sample-wise accuracy
            gen_correct_iter = iter(gen_correct.tolist())
            sample_acc_iter = iter(sample_acc.tolist())
            gen_correct_per_sample: torch.BoolTensor = torch.zeros(label_num_mask.shape[0], dtype=torch.bool)
            sample_acc_per_sample: torch.FloatTensor = torch.zeros(label_num_mask.shape[0], dtype=torch.float32)
            for i in range(label_num_mask.shape[0]):
                gen_correct_per_sample[i] = all(next(gen_correct_iter) for _ in range(num_true_per_sample[i].numel()))
                sample_acc_per_sample[i] = sum(next(sample_acc_iter) for _ in range(num_true_per_sample[i].numel())) / max(1, num_true_per_sample[i].numel())
            gen_correct = gen_correct_per_sample
            sample_acc = sample_acc_per_sample

        acc_list.append(acc)
        per_sample_correct_list.extend(gen_correct.tolist())
        per_sample_acc_list.extend(sample_acc.tolist())

        if "difficulties" in batch and batch["difficulties"] is not None:
            for acc, difficulty in zip(sample_acc.tolist(), batch["difficulties"].tolist()):
                per_difficulty_acc.setdefault(difficulty, []).append(acc)

        if additional_metric_funcs is not None:
            for additional_metric_func in additional_metric_funcs:
                if str(additional_metric_func) == str(metric_func):
                    continue
                additional_metric,_, sample_additional_acc = additional_metric_func.__call__(
                    y_pred=preds,
                    logits=logits,
                    y_true=labels,
                    num_pred=num_pred,
                    num_true=num_true
                )
                if sample_additional_acc.numel() > label_num_mask.shape[0]:
                    # Multiple numbers per sample, e.g. for arithmetic tasks
                    # We need to assign the numbers to the samples and then compute the sample-wise accuracy
                    sample_additional_acc_iter = iter(sample_additional_acc.tolist())
                    sample_additional_acc_per_sample = torch.zeros(label_num_mask.shape[0], dtype=torch.float32)
                    for i in range(label_num_mask.shape[0]):
                        sample_additional_acc_per_sample[i] = sum(next(sample_additional_acc_iter) for _ in range(num_true_per_sample[i].numel())) / max(1, num_true_per_sample[i].numel())
                    sample_additional_acc = sample_additional_acc_per_sample

                additional_metrics.setdefault(str(additional_metric_func), []).append(additional_metric)
                per_sample_additional_metrics.setdefault(str(additional_metric_func), []).extend(sample_additional_acc.tolist())
        # if per_sample_num_loss != []:
            # per_sample_additional_metrics.setdefault("num_loss", []).extend(per_sample_num_loss)
        
        if return_predictions:
            preds_list = [preds[i][label_mask[i,-preds.shape[1]:]].tolist() for i in range(len(preds))]
            gen_predictions.extend(tokenizer.batch_decode(preds_list))
        gen_losses.append(loss)

        if isinstance(test_loader.dataset, _PretokenizedDataset):
            # Use numeric_token_ids to create a mask of numeric tokens in labels
            numeric_label_mask = get_num_token_mask(labels, torch.LongTensor(numeric_token_ids))
            # Check the token right before each true in numeric_label_mask, if it is a sign token (+,-), include it in the mask
            sign_token_ids = [tokenizer.convert_tokens_to_ids("-")]
            for i in range(labels.shape[0]):
                for j in range(1, labels.shape[1]):
                    if numeric_label_mask[i,j] and labels[i,j-1].item() in sign_token_ids:
                        numeric_label_mask[i,j-1] = True
            non_numeric_label_mask = ~numeric_label_mask & label_mask[:,1:]

            # Compute text-only loss
            text_loss = torch.nn.functional.cross_entropy(
                logits[non_numeric_label_mask],
                labels[non_numeric_label_mask],
                reduction="none"
            ).flatten().tolist()
            text_gen_losses.extend(text_loss)

            # Create mask for non-numeric tokens within K=15 tokens of any numeric token using convolution
            K = 20
            # Use 1D convolution to efficiently find tokens within K positions of numeric tokens
            # Create a kernel of size 2*K+1 filled with ones
            kernel = torch.ones(1, 1, 2*K+1, dtype=torch.float32)
            # Apply convolution with padding to maintain sequence length
            near_numeric_mask = torch.nn.functional.conv1d(numeric_label_mask.float().unsqueeze(1), kernel, padding=K).squeeze(1) > 0  # [B, seq_len]
            
            # Select only non-numeric tokens that are near numeric tokens
            non_numeric_label_mask_near_numeric = ~numeric_label_mask & near_numeric_mask & label_mask[:,1:]

            # Compute numeric text-only loss
            numeric_text_loss = torch.nn.functional.cross_entropy(
                logits[non_numeric_label_mask_near_numeric],
                labels[non_numeric_label_mask_near_numeric],
                reduction="none"
            ).flatten().tolist()
            numeric_text_gen_losses.extend(numeric_text_loss)

            if len(text_gen_losses)>=9_724_429:
                text_gen_losses = text_gen_losses[:9_724_429]
                break

        if use_tqdm:
            iterator.set_description(f"Evaluating{post_fix}, Validation loss: {loss:.3f}")
        if step >= max_eval_steps:
            break

    #############################
    # End of For Loop
    #############################
    assert batch, "Batch must not be empty for sample generation."
    if return_samples and (orig_answers:=batch.get("orig_answers")) is not None:
        if preds_text is None:
            preds_list = [preds[i][label_mask[i,-preds.shape[1]:]].tolist() for i in range(min(-10, len(preds)), 0)]
            preds_text = tokenizer.batch_decode(preds_list)
            num_pred_per_sample=[]
            if "label_numbers" in batch and batch["label_numbers"] is not None and num_pred.numel() > 0:
                num_token = tokenizer.init_kwargs["model_specific_special_tokens"]["num_token"]
                total_num_token_occurrences = sum(pt.count(num_token) for pt in preds_text)
                num_pred_iter: Iterator[float] = iter(num_pred[-total_num_token_occurrences:].tolist())
                # For each pred_text, count the number of numbers in the text
                num_pred_per_sample = [[f"{next(num_pred_iter):.17f}".rstrip("0").rstrip(".") for _ in range(pt.count(num_token))] for pt in preds_text]
                preds_text = [replace_in_order(t, [num_token], f if isinstance(f, list) else f.tolist()) for t, f in zip(preds_text, num_pred_per_sample)]
        # Count the cumulative number of num_pred_per_sample
        number_of_nums_in_sample = torch.tensor([len(pt) for pt in num_pred_per_sample]) if num_pred_per_sample else torch.ones(len(preds_text))
        cum_sum: torch.LongTensor = number_of_nums_in_sample.cumsum(0).int()
        gen_correct = check_intervals_all_true(gen_correct, cum_sum)

        assert batch["orig_prompts"] is not None, "Batch must contain original prompts for samples."
        if (difficulties := batch.get("difficulties")) is not None:
            difficulty_d = {"difficulty": difficulties[-10:].tolist()}
        else:
            difficulty_d = {"difficulty": [0.]*10}
        samples_dict = SamplesDict(
            input=batch["orig_prompts"][-10:],
            label=orig_answers[-10:],
            pred=preds_text[-10:],
            correct=per_sample_correct_list[-10:],
            additional_metrics = {
                str(metric_func): sample_acc[-10:].tolist(),
                **{
                    k: v[-10:] for k, v in per_sample_additional_metrics.items()
                },
                **difficulty_d
            }
        )
        assert len(gen_correct)>=10
    else:
        samples_dict = None
    out: EvalOutput = EvalOutput(
        acc = np.mean(acc_list).item(),
        correct_acc = np.array(per_sample_correct_list).astype(float).mean(),
        loss = np.mean(gen_losses).item(),
        num_loss = np.mean(num_gen_losses).item(),
        perplexity=np.mean(np.exp(gen_losses)).item(),
        text_perplexity=np.exp(np.mean(text_gen_losses)).item() if text_gen_losses else np.nan,
        numeric_text_perplexity=np.exp(np.mean(numeric_text_gen_losses)).item() if numeric_text_gen_losses else np.nan,
        additional_metrics={k: np.mean(v).item() for k, v in per_sample_additional_metrics.items()},
        per_sample_acc = per_sample_acc_list,
        per_sample_correct = per_sample_correct_list,
        per_sample_additional_metrics = per_sample_additional_metrics,
        samples_dict=samples_dict,
        gen_predictions = gen_predictions,
        num_preds = num_preds,
        num_trues = num_trues,
        additional_signals={k: np.stack(v).mean(0) for k, v in additional_train_signals.items()},
        per_difficulty_acc={k: np.mean(v).item() for k,v in per_difficulty_acc.items()}
    )
    return out

if __name__ == "__main__":
    import logging
    from collections import OrderedDict
    from typing import cast

    from dataloader.datasets.efficient_prompt_dataset import EfficientPromptEvalDataset
    from utils.visualizer import save_or_add_to_csv
    args = EvalArgumentParser().parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device(args.device)

    tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    tokenizer.padding_side = "left"

    # Load the model
    model: GPT2LMHeadModel = get_model(
        tokenizer,
        args,
        pretrained_model_dir=args.model_dir,
        device=device)

    test_loaders: OrderedDict[str, DataLoader] = OrderedDict()
    for test_set_path, test_cache_paths, test_dataset_type in zip(args.test_set_paths, args.test_cache_paths, args.test_dataset_types):
        logging.info(f"Loading validation dataset from {test_set_path}")
        test_loader = get_eval_loader(
            test_dataset_type,
            test_set_path,
            tokenizer,
            args.device_batch_size,
            context_length=args.context_length,
            seed=42,
            num_workers=args.num_workers,
            val_cached_path=test_cache_paths,
            shuffle=False,
            cache_base_path=args.cache_base_path,
            difficulty_column=args.difficulty_column,
            load_all_columns=True,
            dataset_curriculum_type=DATASET_CURRICULUM_TYPE.STANDARD
        )
        test_loaders[test_set_path.stem] = test_loader
    time_stamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.save_dir is None:
        save_dir = Path(args.model_dir) / "eval" / time_stamp
    else:
        save_dir = f"{str(args.save_dir)}_{time_stamp}"
    args.save_dir = Path(save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    print_and_save_arguments(args, args.save_dir)

    all_task_metrics = list(dict())
    
    for i, (dataset_name, test_loader) in enumerate(test_loaders.items()):
        eval_out: EvalOutput = evaluate(
            model,
            test_loader,
            args.test_set_metrics[i],
            tokenizer,
            max_eval_steps=(384//args.device_batch_size)*26 if "pretokenized" in args.test_dataset_types[i] else maxsize, # Equal to 10_485_760 val tokens used by modded-nanogpt
            use_tqdm=True,
            leave_tqdm=True,
            return_predictions=True,
            return_samples=False,
            post_fix=f" {i+1}/{len(test_loaders)}: {dataset_name}",
            dtype=args.data_type,
            additional_metric_funcs=args.additional_metrics
        )
        test_df = cast(EfficientPromptEvalDataset, test_loader.dataset).df if hasattr(test_loader.dataset, "df") else None
        if args.save_testset_predictions[i] and test_df is not None:
            test_df["token prediction"] = eval_out.gen_predictions
            test_df["token correct"] = np.array(eval_out.per_sample_correct, dtype=bool)
            if eval_out.num_preds != []:
                test_df["number prediction"] = eval_out.num_preds
                test_df["number correct"] = np.array(eval_out.num_trues, dtype=bool)
        if test_df is not None:
            test_df[str(args.test_set_metrics[i])] = np.array(eval_out.per_sample_acc)
            for k, v in eval_out.per_sample_additional_metrics.items():
                test_df[k] =  np.array(v)
            test_df[str(args.test_set_metrics[i])] = np.array(eval_out.per_sample_acc)
            params = get_strat_params(dataset_name)
            plot_accuracy_distribution(test_df, x_col=params[1], y_col=params[0], acc_col=str(args.test_set_metrics[i]), save_path=args.save_dir / f"{dataset_name}_distribution.png", title=f"{dataset_name} {str(args.test_set_metrics[i])}")
            if args.save_testset_predictions[i]:
                test_df.to_csv(args.save_dir / f"test_predictions_{dataset_name}.csv", index=False)
        metrics: dict[str, float] = {
            "loss": eval_out.loss,
            "num_loss": eval_out.num_loss,
            "perplexity": eval_out.perplexity,
            "text_perplexity": eval_out.text_perplexity,
            "numeric_text_perplexity": eval_out.numeric_text_perplexity,
            "sample_acc": eval_out.correct_acc,
            str(args.test_set_metrics[i]): eval_out.acc,
            **eval_out.additional_metrics,
        }
        logging.info(f"Metrics for dataset '{dataset_name}':")
        for metric_name, metric_value in metrics.items():
            logging.info(f"  {metric_name}: {metric_value:.6f}")
        save_or_add_to_csv(metrics, args.save_dir / f"test_metrics_{dataset_name}.csv")
        all_task_metrics.append({"dataset_name": dataset_name, **metrics})

    if len(test_loaders) > 1:
        pd.DataFrame.from_records(all_task_metrics).to_csv(args.save_dir / "all_test_metrics.csv", index=False)
    print("Evaluation complete.")
    print(f"Results saved to {args.save_dir}")
