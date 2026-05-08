if True:
    print("Loading libraries")
from collections import Counter
from time import time
from typing import Optional, cast

import numpy as np
import pandas as pd
import torch
import wandb
import wandb.wandb_run

from dataloader.curriculum_manager import CurriculumManager
from dataloader.curriculum_sampler import CurriculumFixedRatioSampler
from dataloader.dataloaders import CurriculumEvalDataset
from dataloader.datasets.pretokenized_dataset import _PretokenizedDataset
from eval import evaluate
from networks.number_embedding_modules.abc_embedding import ABCEmbedding
from training.loss_and_task_weighting import perform_loss_and_task_weighting
from utils.enums import (
    DATASET_CURRICULUM_TYPE,
    EvalOutput,
    TrackedMetrics,
    Trainer,
    TrainMetrics,
)
from utils.metrics import generalized_mean
from utils.train_argument_parser import TrainArgumentParser
from utils.util_funcs import (
    save_checkpoint,
    to_flat_dict,
)
from utils.visualizer import (
    plot_accuracy,
    plot_dict,
    plot_difficulty_acc,
    plot_freq_loss_and_weights,
    plot_interval_duration,
    plot_losses,
    plot_lr_update,
    save_or_add_to_csv,
)


def perform_evaluation(
        trainer: Trainer,
        start_time: float,
        train_metrics: TrainMetrics,
        args: TrainArgumentParser,
        num_embedding_module: Optional[ABCEmbedding],
        wandb_run: Optional[wandb.wandb_run.Run],
        curriculum_manager: Optional[CurriculumManager] = None
    ):
    """
    Perform evaluation of the model and update metrics.
    This function evaluates the model on validation datasets, updates training metrics,
    adjusts loss weights if necessary, and saves the results to CSV files.
    It also generates plots for losses, accuracies, and learning rates.
    If curriculum learning is enabled, performs curriculum-specific evaluation.
    
    Args:
        trainer (Trainer): The trainer instance containing model, optimizer, and other training parameters.
        start_time (float): The start time of the evaluation.
        train_metrics (TrainMetrics): The training metrics to be updated.
        args (TrainArgumentParser): The parsed training arguments.
        num_embedding_module (ABCEmbedding): The number embedding module used in the model.
        wandb_run (Optional[wandb.wandb_run.Run]): The Weights & Biases run instance for logging metrics.
        curriculum_manager (Optional[CurriculumManager]): The curriculum manager for curriculum learning.
    """
    interval_duration = time() - start_time
    trainer.eval_steps.append(trainer.step)
    num_steps_in_interval = trainer.step - (trainer.eval_steps[-2] if len(trainer.eval_steps) > 1 else 0)
    trainer.total_duration += interval_duration
    token_since_last_eval = train_metrics.num_tokens_per_step[trainer.step]
    if len(trainer.eval_steps) > 1:
        token_since_last_eval-=train_metrics.num_tokens_per_step[trainer.eval_steps[-2]]
    trainer.interval_durations.append(interval_duration / token_since_last_eval * 1e6)

    train_metrics.val_gen_accs.append({})
    train_metrics.val_gen_losses.append({})
    train_metrics.val_gen_perplexities.append({})
    train_metrics.additional_metrics.append({})
    accs = []
    samples_dict = dict()
    for i, (dataset_name, val_loader) in enumerate(trainer.val_loaders.items()):
        eval_output: EvalOutput = evaluate(
            trainer.model,
            val_loader,
            args.val_set_metrics[i],
            trainer.tokenizer,
            max_eval_steps=(args.effective_batch_size//args.device_batch_size)*(1 if isinstance(val_loader.dataset, _PretokenizedDataset) else args.max_eval_steps),
            use_tqdm=args.tqdm,
            return_predictions=False,
            return_samples=True,
            post_fix=f" {i+1}/{len(trainer.val_loaders)}: {dataset_name}",
            dtype=args.data_type,
            additional_metric_funcs=args.val_additional_metrics
        )
        train_metrics.val_gen_accs[-1][dataset_name+" "+str(args.val_set_metrics[i])] = eval_output.acc
        if len(args.train_set_loss_weights) > i:
            loss_weight = args.train_set_loss_weights[i].item()
        else:
            loss_weight = 1
        train_metrics.val_gen_losses[-1][dataset_name] = loss_weight * eval_output.loss + (args.num_loss_weight * eval_output.num_loss if eval_output.num_loss is not np.nan else 0)
        train_metrics.val_gen_perplexities[-1][dataset_name] = eval_output.perplexity
        train_metrics.additional_metrics[-1][dataset_name] = eval_output.additional_metrics
        accs.append(eval_output.acc)
        if eval_output.samples_dict:
            items_in_dict = len(samples_dict.get("dataset_name", []))
            items_in_sample = len(eval_output.samples_dict.pred)
            samples_dict.setdefault("dataset_name", []).extend([dataset_name] * items_in_sample)
            key_set = {"dataset_name"}
            for k,v in vars(eval_output.samples_dict).items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        key_set.add(k2)
                        assert len(v2) == items_in_sample, f"Expected {items_in_sample} items for key {k2}, got {len(v2)} for dataset {dataset_name}"
                        samples_dict.setdefault(k2, [np.nan]*items_in_dict).extend(v2)
                else:
                    key_set.add(k)
                    samples_dict.setdefault(k, [np.nan]*items_in_dict).extend(v)
                    assert len(v) == items_in_sample, f"Expected {items_in_sample} items for key {k}, got {len(v)} for dataset {dataset_name}"
            # For all existing keys in samples_dict that are not in eval_output.samples_dict, extend with NaNs
            for k in samples_dict.keys():
                if k not in key_set:
                    samples_dict[k].extend([np.nan]*items_in_sample)
        assert eval_output.additional_signals is not None, "Eval output must contain additional signals"
        if "num_loss_per_frequency" in eval_output.additional_signals:
            num_loss_per_frequency: np.ndarray = eval_output.additional_signals.pop("num_loss_per_frequency")
            trainer.eval_loss_per_freq.append(num_loss_per_frequency)
        if "prob_use_linear" in eval_output.additional_signals:
            prob_use_linear: float = eval_output.additional_signals.pop("prob_use_linear")
            trainer.eval_use_linear_prob[dataset_name].append(prob_use_linear)
        for k,v in eval_output.additional_signals.items():
            assert isinstance(v, float), f"Expected additional signal {k} to be a float, got {type(v)}"
            trainer.additional_eval_losses.setdefault(k, []).append(v)

        if curriculum_manager is not None and eval_output.per_difficulty_acc and i<curriculum_manager.num_tasks and curriculum_manager.supports_curriculum[i]:
            curriculum_manager.update_performance(i, eval_output.per_difficulty_acc)
            advancement_made = curriculum_manager.advance_if_possible(trainer.step, trainer.lr_schedulers[0].get_last_lr()[0] if trainer.step>args.num_warmup_steps else args.lr)
            if advancement_made and not curriculum_manager.difficulty_sampling[i]:
                cast(CurriculumEvalDataset, val_loader.dataset).switch_to_standard_sampling()
            cast(CurriculumFixedRatioSampler, trainer.train_loader.sampler).update_ratios_from_curriculum_manager()
            trainer.eval_difficulty_acc[dataset_name] = {
                k: trainer.eval_difficulty_acc.get(dataset_name, dict()).get(k, []) + [eval_output.per_difficulty_acc[k]]
                for k in eval_output.per_difficulty_acc
            }
    try:
        pd.DataFrame(samples_dict).to_csv(trainer.save_dir / "eval_samples.csv", index=False)
    except ValueError as e:
        # This can happen if the arrays in samples_dict have different lengths
        print("Warning: Could not save eval samples to CSV!")
        # Print the lengths of the arrays in samples_dict for debugging
        for k,v in samples_dict.items():
            print(f"{k}: {len(v)}")
        raise e


    train_losses = {
        k: 0. if np.all(np.isnan(v[-num_steps_in_interval:])) else np.nanmean(v[-num_steps_in_interval:]).item()
        for k,v in train_metrics.train_losses.items()
    }
    # Adjust the loss weights and task ratios if necessary
    perform_loss_and_task_weighting(args, trainer, curriculum_manager, train_metrics, accs, train_losses)

    # Add val total metrics
    task_accs: torch.FloatTensor = torch.tensor((list(train_metrics.val_gen_accs[-1].values())))
    if curriculum_manager is not None:
        task_accs_eval = list()
        for i in range(len(trainer.val_loaders)):
            if i < curriculum_manager.num_tasks-1:
                if curriculum_manager.dataset_curriculum_types[i+1] == DATASET_CURRICULUM_TYPE.ENDGAME:
                    continue
            task_accs_eval.append(task_accs[i].item())
        task_accs = torch.tensor(task_accs_eval)

    train_metrics.val_gen_accs[-1]["total"] = generalized_mean(task_accs=task_accs, p=args.task_loss_exponent) # Generalized mean as global comparison metric
    train_metrics.val_gen_perplexities[-1]["total"] = np.nanmean(list(train_metrics.val_gen_perplexities[-1].values())).item()
    if len(trainer.train_loss_per_freq) > 0:
        trainer.train_loss_per_freq_agg.append(np.stack(trainer.train_loss_per_freq).mean(axis=0))
        trainer.train_loss_per_freq = []
    if len(trainer.additional_train_losses) > 0:
        for k,v in trainer.additional_train_losses.items():
            trainer.additional_train_losses_agg.setdefault(k, []).append(np.mean(v).item())
            trainer.additional_train_losses[k] = []
    if curriculum_manager is not None:
        for i,(dataset_name, loader) in enumerate(trainer.val_loaders.items()):
            if hasattr(loader.dataset, "supports_curriculum_learning") and loader.dataset.supports_curriculum_learning(None) and i<curriculum_manager.num_tasks:
                trainer.maximum_difficulty.setdefault(dataset_name, []).append(curriculum_manager.frontier_difficulties[i] - curriculum_manager.min_difficulties[i])
        if len(trainer.eval_difficulty_acc)>0:
            for dataset_name,v in trainer.eval_difficulty_acc.items():
                trainer.eval_difficulty_acc_agg[dataset_name] = {
                    k: trainer.eval_difficulty_acc_agg.get(dataset_name, dict()).get(k, []) + [np.mean(trainer.eval_difficulty_acc[dataset_name][k]).item()]
                    for k in trainer.eval_difficulty_acc[dataset_name]
                }
                trainer.eval_difficulty_acc_agg[dataset_name] = dict(sorted(trainer.eval_difficulty_acc_agg[dataset_name].items(), key=lambda x: x[0]))
                trainer.eval_difficulty_acc[dataset_name] = dict()
            for i, (dataset_name, val_loader) in enumerate(trainer.val_loaders.items()):
                if i<curriculum_manager.num_tasks and curriculum_manager.supports_curriculum[i]:
                    task_difficulty_ratios = curriculum_manager.difficulty_ratios[i]
                    with open(trainer.save_dir / f"eval_per_difficulty_acc_{dataset_name.removesuffix(".gz").removesuffix(".csv")}.csv", "w") as f:
                        f.write("difficulty,acc,ratio\n")
                        for j, (difficulty, accs) in enumerate(trainer.eval_difficulty_acc_agg[dataset_name].items()):
                            trainer.eval_difficulty_ratios.setdefault(dataset_name, dict()).setdefault(difficulty, []).append(task_difficulty_ratios[j].item() if j < len(task_difficulty_ratios) else 0)
                            f.write(f"{difficulty},{accs[-1]},{task_difficulty_ratios[j].item() if j<len(task_difficulty_ratios) else 0}\n")
                    trainer.eval_difficulty_ratios[dataset_name] = dict(sorted(trainer.eval_difficulty_ratios[dataset_name].items(), key=lambda x: x[0]))
                    advancement_thresholds: dict[int, float] = curriculum_manager.get_difficulty_thresholds(i)
                    for k,v in advancement_thresholds.items():
                        trainer.advancement_thresholds.setdefault(dataset_name, dict()).setdefault(k, list()).append(v)

                        difficulties_counter = dict(Counter(trainer.difficulties.get(i, [])))
                        step_difficulties = dict()
                        for diff in cast(CurriculumManager, trainer.curriculum_manager).available_difficulties[i]:
                            step_difficulties[diff] = difficulties_counter.get(diff, 0) / len(trainer.difficulties.get(i, [0]))


                    trainer.difficulties_agg.setdefault(dataset_name, list()).append(step_difficulties)
                    save_or_add_to_csv(
                        {k: v * args.effective_batch_size for k,v in step_difficulties.items()},
                        trainer.save_dir / f"difficulties_{dataset_name.removesuffix('.gz').removesuffix('.csv')}.csv"
                    )
            trainer.difficulties.clear()

            data_dict: dict[int, dict[str, float]] = dict()
            for dn, dl in trainer.difficulties_agg.items():
                for step_idx,difficulty_count in zip(trainer.eval_steps, dl):
                    data_dict.setdefault(train_metrics.num_tokens_per_step[step_idx], dict())[dn.removesuffix(".gz").removesuffix(".csv")] =sum([diff*count for diff,count in difficulty_count.items() if count>0])
            plot_dict(
                data_dict,
                trainer.save_dir / "difficulties_line.png",
                title="Mean training sample difficulty over time",
                ylabel="Mean difficulty",
                xlabel="Tokens"
            )

            if trainer.eval_difficulty_ratios != {}:
                plot_difficulty_acc(
                    eval_difficulty_acc_agg={k: np.stack(list(v.values())) for k,v in trainer.eval_difficulty_acc_agg.items()},
                    maximum_difficulty={k: v for k,v in trainer.maximum_difficulty.items() if k in trainer.eval_difficulty_acc_agg},
                    tokens=train_metrics.num_tokens_per_step[trainer.step],
                    save_path=trainer.save_dir / "eval_difficulty_acc.png",
                    min_difficulty={k: curriculum_manager.available_difficulties[i][0] for i,k in enumerate(trainer.val_loaders.keys()) if i<curriculum_manager.num_tasks and k in trainer.eval_difficulty_acc_agg},
                    eval_difficulty_ratios={k: np.stack([list(d.values()) for d in v]).T for k,v in trainer.difficulties_agg.items()},
                    advancement_thresholds={k: np.stack(list(v.values())) for k,v in trainer.advancement_thresholds.items()}
                )
    tracked_metrics_dict: dict = to_flat_dict(TrackedMetrics(
        step = trainer.step,
        num_tokens=train_metrics.num_tokens_per_step[trainer.step],
        train_loss = train_losses,
        train_perplexity = np.mean(train_metrics.train_perplexities[-num_steps_in_interval:]).item(),
        train_token_accs = np.mean(train_metrics.train_token_accs[-num_steps_in_interval:]).item(),
        val_gen_loss = train_metrics.val_gen_losses[-1],
        val_gen_perplexity = train_metrics.val_gen_perplexities[-1],
        val_gen_acc = train_metrics.val_gen_accs[-1],
        additional_metrics = train_metrics.additional_metrics[-1],
        lr = train_metrics.lr_updates[-1],
        duration = trainer.interval_durations[-1],
        max_difficulty={k: v[-1] for k,v in trainer.maximum_difficulty.items()}
    ))
    save_or_add_to_csv(tracked_metrics_dict, trainer.save_dir / "metrics.csv")


    if num_embedding_module is not None and hasattr(num_embedding_module, "freq_loss_weights"):
        trainer.freq_loss_weights_agg.append(num_embedding_module.freq_loss_weights.float().cpu().numpy()[0, :num_embedding_module.freq_size])
        
    # Repeat for eval
    if len(trainer.eval_loss_per_freq) > 0:
        trainer.eval_loss_per_freq_agg.append(np.stack(trainer.eval_loss_per_freq).mean(axis=0))
        trainer.eval_loss_per_freq = []
    if len(trainer.additional_eval_losses) > 0:
        for k,v in trainer.additional_eval_losses.items():
            trainer.additional_eval_losses_agg.setdefault(k, []).append(np.mean(v).item())
            trainer.additional_eval_losses[k] = []

    if wandb_run is not None:
        wandb_run.log(tracked_metrics_dict)
    if trainer.step > args.eval_every_k_steps and train_metrics.num_tokens_per_step[trainer.step] > args.online_weighting_warmup_tokens:
        plot_losses(train_metrics, trainer.save_dir / "loss_plot.png", skip_k_steps=args.online_weighting_warmup_tokens // args.eval_every_k_tokens)
        plot_accuracy(train_metrics, trainer.save_dir / "accuracy_plot.png", args.val_set_metrics)
        plot_lr_update(trainer.save_dir, train_metrics.lr_updates, list(train_metrics.num_tokens_per_step.values())[1:])
        plot_interval_duration(trainer.interval_durations, [train_metrics.num_tokens_per_step[s] for s in trainer.eval_steps], trainer.total_duration, trainer.save_dir, batch_size=args.device_batch_size)
        if num_embedding_module is not None:
            if len(trainer.train_loss_per_freq_agg) > 1:
                train_signals = {
                    "num_loss_per_freq": np.stack(trainer.train_loss_per_freq_agg),
                    "freq_loss_weights": np.stack(trainer.freq_loss_weights_agg),
                    **{k: np.stack(v) for k,v in trainer.additional_train_losses_agg.items()}
                }
                plot_freq_loss_and_weights(
                    train_signals, 
                    tokens=train_metrics.num_tokens_per_step[trainer.step],
                    save_path=trainer.save_dir / "train_loss_per_frequency.png"
                )
            
            if len(trainer.eval_loss_per_freq_agg) > 1:
                for k,v in trainer.eval_use_linear_prob.items():
                    if len(v) > 0:
                        trainer.eval_use_linear_prob_agg[k].append(np.mean(v).item())
                        trainer.eval_use_linear_prob[k] = []
                eval_signals = {
                    "num_loss_per_freq": np.stack(trainer.eval_loss_per_freq_agg),
                    "freq_loss_weights": np.stack(trainer.freq_loss_weights_agg),
                    **{k: np.stack(v) for k,v in trainer.additional_eval_losses_agg.items()}
                }
                plot_freq_loss_and_weights(
                    eval_signals, 
                    tokens=train_metrics.num_tokens_per_step[trainer.step],
                    save_path=trainer.save_dir / "eval_loss_per_frequency.png"
                )
    if args.loss_weight_momentum != 1:
        plot_dict(trainer.loss_weights, trainer.save_dir / "loss_weights.png", title="Loss weights", ylabel="Loss weight", xlabel="Tokens")
        plot_dict(trainer.train_ratios, trainer.save_dir / "train_ratios.png", title="Train ratios", ylabel="ratio [%]", xlabel="Tokens")

    # Save model checkpoints. Skip in warmup phase
    if trainer.step > args.num_warmup_steps:
        if max([list(d.values())[trainer.COMPARE_METRIC_INDEX] for d in train_metrics.val_gen_accs]) == list(train_metrics.val_gen_accs[-1].values())[trainer.COMPARE_METRIC_INDEX]:
            save_checkpoint(trainer.model, trainer.optimizer, trainer.lr_schedulers, trainer.step, prefix="best", save_dir=trainer.save_dir, curriculum_manager=curriculum_manager)
        if not args.no_save_latest:
            save_checkpoint(trainer.model, trainer.optimizer, trainer.lr_schedulers, trainer.step, prefix="latest", save_dir=trainer.save_dir, curriculum_manager=curriculum_manager)
    
    # Run milestone evaluations
    if trainer.step % args.save_checkpoint_steps == 0:
        save_checkpoint(trainer.model, trainer.optimizer, trainer.lr_schedulers, trainer.step, save_dir=trainer.save_dir, curriculum_manager=curriculum_manager)
