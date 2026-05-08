import logging

if True:
    print("Loading libraries")
import os
from time import time
from typing import Optional, cast

import numpy as np
import torch
import wandb
import wandb.wandb_run
from torch.amp.autocast_mode import autocast

from networks.number_embedding_modules.abc_embedding import ABCEmbedding
from training.evaluation import perform_evaluation
from training.setup import setup
from utils.enums import (
    CausalLMOutputWithCrossAttentionsAndNumbers,
    TrainMetrics,
)
from utils.train_argument_parser import TrainArgumentParser
from utils.util_funcs import (
    save_checkpoint,
)
from utils.warm_start_lr_scheduler import (
    WarmStartReduceLROnPlateau,
)

print("Libraries loaded")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def training(args: TrainArgumentParser, wandb_run: Optional[wandb.wandb_run.Run] = None):
    trainer, loaded_train_metrics = setup(args, wandb_run)
    model = trainer.model
    optimizer = trainer.optimizer
    train_metrics = loaded_train_metrics if loaded_train_metrics is not None else TrainMetrics()
    num_embedding_module: Optional[ABCEmbedding] = None
    if hasattr(model, "num_embedding_module"):
        num_embedding_module = cast(ABCEmbedding, model.num_embedding_module)

    minibatch_losses: dict[str, list[float]] = dict()

    if not isinstance(args.task_ratio_cap, torch.Tensor):
        args.task_ratio_cap = torch.tensor(args.task_ratio_cap, dtype=torch.float32)
    # Train model
    model.train()
    model.zero_grad()
    start_time = time()
    while trainer.step<args.max_train_steps:
        for s, batch in enumerate(trainer.train_loader):
            current_lr = trainer.lr_schedulers[0].get_last_lr()[0]
            if args.lr_scheduler_type == "plateau" and current_lr <= args.lr * 0.1 and trainer.step > args.num_warmup_steps:
                logging.info(f"Stopping training because the learning rate is too low: {current_lr} <= {args.lr * 0.1}")
                trainer.step = args.max_train_steps
                break
            trainer.step+=1
            train_metrics.num_tokens_per_step[trainer.step] = train_metrics.num_tokens_per_step[trainer.step-1]+batch["input_ids"].numel()
            
            labels: torch.LongTensor = batch["labels"]
            with autocast(device_type=str(args.device), dtype=cast(torch.dtype, args.data_type), enabled=args.data_type != torch.float32):
                kwargs = {}
                if "input_numbers" in batch:
                    kwargs["numbers"] = batch["input_numbers"].to(device=args.device)
                kwargs["cu_seq_lens"] = batch["cu_seq_lens"].to(device=args.device)
                kwargs["max_seq_length"] = batch["max_seq_length"]
                out: CausalLMOutputWithCrossAttentionsAndNumbers = model.forward(
                    input_ids=batch["input_ids"].to(args.device),
                    attention_mask=batch["attention_mask"].to(args.device),
                    labels=labels.to(args.device), # pyright: ignore[reportArgumentType]
                    position_ids=batch["position_ids"].to(args.device),
                    **kwargs
                )
                additional_train_losses: dict = getattr(out, "additional_train_losses", dict())
                out.additional_train_losses = additional_train_losses
                if "num_loss_per_frequency" in out.additional_train_losses:
                    trainer.train_loss_per_freq.append(out.additional_train_losses.pop("num_loss_per_frequency").cpu().numpy())
                for k,v in out.additional_train_losses.items():
                    assert isinstance(v, float), f"Additional train losses must be float but got {type(v)} for key {k}"
                    trainer.additional_train_losses.setdefault(k, []).append(v)
                assert out.loss is not None, "Model must return loss"
                sample_losses: torch.Tensor = out.loss.view(batch["input_ids"].shape[0],-1)
                mask = labels[..., 1:] != -100
                if "sample_idx" in batch:
                    sample_token_weights: torch.FloatTensor = torch.gather(cast(torch.Tensor,args.train_set_loss_weights), 0, batch["sample_idx"]).unsqueeze(1).to(device=args.device)
                    num_sample_losses = getattr(out, "num_loss", torch.zeros_like(sample_losses)) 
                    sample_losses = sample_losses + args.num_loss_weight * num_sample_losses
                    nan_mask = sample_losses.isnan()
                    if nan_mask.any() and not nan_mask.all():
                        nan_mask = nan_mask.cpu()
                        logging.warning("NaN in sample losses. Skipping sample.")
                        mask = mask & ~nan_mask
            train_metrics.train_perplexities.append(sample_losses[mask].detach().cpu().mean().exp().item())
            loss = (sample_losses* sample_token_weights)[mask].mean()
            assert not loss.isnan().any(), "Loss is NaN. Please check your model and data."
            loss.backward()
            with torch.no_grad():
                sample_losses = sample_losses.detach().cpu() / mask.sum()
                dataset_losses: dict[str, float] = {
                    train_set_paths.name.removesuffix(".csv.gz"): args.train_set_loss_weights[i].item() * sample_losses[mask & (batch["sample_idx"]==i).unsqueeze(1)].sum().item()
                    for i, train_set_paths in enumerate(args.train_set_paths)
                }

                minibatch_losses.setdefault("train_total", []).append(loss.item())
                for key, value in dataset_losses.items():
                    minibatch_losses.setdefault(key, []).append(value)

                equality = cast(torch.LongTensor, out.logits[..., :-1, :].detach().argmax(-1).cpu()[mask]) == labels[..., 1:][mask]
                train_metrics.train_token_accs.append(equality.float().mean().item())
                del out

                for task_id, diff in zip(batch.get("sample_idx", torch.empty(0)).tolist(), batch.get("difficulty", torch.empty(0)).tolist()):
                    trainer.difficulties.setdefault(task_id, list()).append(diff)

            train_metrics.lr_updates.append(trainer.lr_schedulers[0].get_last_lr()[0])
            if trainer.step % (args.effective_batch_size//args.device_batch_size) == 0:
                for p in model.parameters(): 
                    if p.grad is not None:
                        p.grad /= (args.effective_batch_size//args.device_batch_size) # pyright: ignore[reportOperatorIssue]
                if args.from_pretrained is None:
                    frac = (trainer.step / (args.effective_batch_size//args.device_batch_size))/ args.muon_momentum_warmup # momentum warmup for muon
                    if frac <= 1:
                        for group in optimizer.param_groups:
                            if group.get('use_muon', False):
                                group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
                if args.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                for lr_scheduler in trainer.lr_schedulers:
                    if isinstance(lr_scheduler, WarmStartReduceLROnPlateau):
                        lr_scheduler.step(list(train_metrics.val_gen_accs[-1].values())[0] if train_metrics.val_gen_accs else None) # pyright: ignore[reportArgumentType]
                    else:
                        lr_scheduler.step()
                model.zero_grad()

                for key, value in minibatch_losses.items():
                    train_metrics.train_losses.setdefault(key, []).append((np.mean(value)).item())
                minibatch_losses.clear()

                if trainer.tt is not None:
                    trainer.tt.set_description(f"Epoch {trainer.step//len(trainer.train_loader)+1}, loss: {train_metrics.train_losses["train_total"][-1]:.3f}")

            if trainer.step % args.eval_every_k_steps == 0:
                perform_evaluation(trainer, start_time, train_metrics, args, num_embedding_module, wandb_run, trainer.curriculum_manager)
                model.train()
                start_time = time()
            if trainer.tt is not None:
                trainer.tt.update(1)
            if trainer.step >= args.max_train_steps or trainer.step>= args.stop_after:
                break
            # If the last 5 total accs are 1.0, stop training
            if len(train_metrics.val_gen_accs) > 5 and all([v >= 1.0 for v in [list(d.values())[trainer.COMPARE_METRIC_INDEX] for d in train_metrics.val_gen_accs[-5:]]]):
                logging.info(f"Stopping training because the last 5 total accs are 1.0: {[list(d.values())[trainer.COMPARE_METRIC_INDEX] for d in train_metrics.val_gen_accs[-5:]]}")
                trainer.step = args.max_train_steps
                break

        if trainer.step >= args.max_train_steps:
            break

    # Save the model
    if args.no_save_latest:
        save_checkpoint(model, optimizer, trainer.lr_schedulers, trainer.step, save_dir=trainer.save_dir, curriculum_manager=trainer.curriculum_manager)

if __name__ == "__main__":
    args= TrainArgumentParser().parse_args()
    wandb_run = None
    if args.wandb_sweep_id:
        wandb.login()
        def run_training():
            arg_dict = args._log_all()
            arg_str_dict = {k:  str(v) for k,v in arg_dict.items()}
            sorted_arg_dict = {k: arg_str_dict[k] for k in sorted(arg_str_dict) if not callable(arg_str_dict[k])}
            with wandb.init(config=sorted_arg_dict) as wandb_run:
                final_dict = {k: (v if arg_str_dict[k] != v else arg_dict[k]) for k, v in wandb.config.as_dict().items()}
                training(TrainArgumentParser().from_dict(final_dict), wandb_run)
        wandb.agent(args.wandb_sweep_id, function=run_training, project=args.wandb_project, count=1)
        exit(0)
    elif args.wandb and args.wandb_project and args.wandb_project != "debug":
        wandb.login()
        arg_dict = args._log_all()
        sorted_arg_dict = {k: arg_dict[k] for k in sorted(arg_dict) if not callable(arg_dict[k])}
        wandb_run = wandb.init(project=args.wandb_project, group=args.wandb_group, config=sorted_arg_dict)
    training(args, wandb_run)
    if wandb_run:
        wandb_run.finish()
    logging.info("Training finished.")
