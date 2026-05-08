
from typing import override

import torch


class WarmStartLrScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, total_iters: int, init_lr: float = 1e-7):
        self.target_lrs: list[float] = [group["lr"] for group in optimizer.param_groups]
        self.total_iters = total_iters
        self.init_lr = init_lr
        self.ds = [(target_lr-init_lr) / total_iters for target_lr in self.target_lrs]
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > 0:
            return [group['lr'] + self.ds[i] for i, group in enumerate(self.optimizer.param_groups)]
        else:
            return [group['lr']*0+self.init_lr for group in self.optimizer.param_groups]
        

class WarmStartReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    Combines the functionality of `torch.optim.lr_scheduler.ReduceLROnPlateau` and `WarmStartLrScheduler`.
    This scheduler will first warm up the learning rate to the target learning rate over a specified number of iterations,
    and then it will reduce the learning rate when a metric has stopped improving.
    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        warmup_iters (int): Total number of iterations for the warm-up phase.
        init_lr (float, optional): Initial learning rate for the warm-up phase. Default: 1e-7
        **kwargs: Additional arguments for `torch.optim.lr_scheduler.ReduceLROnPlateau`.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_iters: int, init_lr: float = 1e-7, **kwargs):
        """
        Initialize the scheduler.
        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            warmup_iters (int): Total number of iterations for the warm-up phase.
            init_lr (float, optional): Initial learning rate for the warm-up phase. Default: 1e-7
            **kwargs: Additional arguments for `torch.optim.lr_scheduler.ReduceLROnPlateau`.
        """
        self.target_lrs: list[float] = [group["lr"] for group in optimizer.param_groups]
        self.warmup_iters = warmup_iters
        self.init_lr = init_lr
        self.ds = [(target_lr-init_lr) / warmup_iters for target_lr in self.target_lrs]
        super().__init__(optimizer, **kwargs)
        super(torch.optim.lr_scheduler.ReduceLROnPlateau, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > 0:
            return [group['lr'] + self.ds[i] for i, group in enumerate(self.optimizer.param_groups)]
        else:
            return [group['lr']*0+self.init_lr for group in self.optimizer.param_groups]
    
    @override
    def step(self, metrics=None, epoch=None):
        """Perform a step."""
        if self._step_count <= self.warmup_iters:
            super(torch.optim.lr_scheduler.ReduceLROnPlateau, self).step()
        else:
            super().step(metrics, epoch)
