from math import exp
from typing import Callable, Optional

import torch


def sMAPE(
        *,
        num_pred: torch.DoubleTensor,
        num_true: torch.DoubleTensor,
        eps: float=1e-100,
        correct_threshold: float=0.95,
        **kwargs
    ) -> tuple[float, torch.BoolTensor, torch.FloatTensor]:
    r"""Calculate Symmetric Mean Absolute Percentage Error (sMAPE).
    .. math::
        \text{sMAPE} = \frac{1}{n}\sum_{j=1}^n\frac{\left|A_j-P_j\right|}{\left|A_j\right|+\left|P_j\right|}

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.
    Args:
        num_pred (DoubleTensor): The predicted values
        num_true (DoubleTensor): The true values
        eps (float): A small value to prevent division by zero
        correct_threshold (float): Threshold for correct prediction
        **kwargs: Additional arguments
    Returns:
        accuracy (float): The accuracy
        correct_samples (BoolTensor): The correct samples
        acc (DoubleTensor): The sMAPE for each sample in the batch
    """
    sMAP = (torch.abs(num_pred - num_true) / (torch.abs(num_true) + torch.abs(num_pred) + eps))
    acc = 1 - sMAP
    # replace nan values with 0
    acc = acc.nan_to_num(0.0)
    correct_samples = acc > correct_threshold
    return acc.mean().item(), correct_samples, acc.float()

def logSMAPE(
        num_pred: torch.DoubleTensor,
        num_true: torch.DoubleTensor,
        max_num_digits=15,
        eps: float=1e-100,
        correct_threshold: float=0.5,
        y_true: Optional[torch.LongTensor]=None,
        **kwargs
    ) -> tuple[float, torch.BoolTensor, torch.FloatTensor]:
    r"""Calculate log Symmetric Mean Absolute Percentage Error (sMAPE).
    .. math::
        \text{sMAPE} = \frac{1}{n}\sum_{j=1}^n\frac{\left|A_j-P_j\right|}{\left|A_j\right|+\left|P_j\right|}

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.
    Args:
        num_pred (DoubleTensor): The predicted values
        num_true (DoubleTensor): The true values
        max_num_digits (int): The maximum number of digits to consider
        eps (float): A small value to prevent division by zero
        correct_threshold (float): Threshold for correct prediction
        y_true (LongTensor, optional): The true tokens of shape (batch_size, seq_length). Used to determine the batch size if num_pred is empty.
        **kwargs: Additional arguments
    Returns:
        accuracy (float): The mean log sMAPE metric between 0 and 1 (higher is better)
        correct_samples (BoolTensor): The correct samples
        acc (DoubleTensor): The log sMAPE for each sample in the batch
    """
    if num_pred.numel() == 0:
        assert y_true is not None, "y_true must be provided if num_pred is empty"
        return torch.nan, torch.full((y_true.size(0),), torch.nan), torch.full((y_true.size(0),), torch.nan)
    sMAPE = (num_pred - num_true).abs() / (abs(num_true) + abs(num_pred) + eps).abs()
    log_sMAPE = (sMAPE + eps).log10() / -max_num_digits
    acc = log_sMAPE.clamp(max=1)
    # replace nan values with 0
    acc = acc.nan_to_num(0.0)
    correct_samples = acc > correct_threshold
    return acc.mean().item(), correct_samples, acc.float()

def logSMAPE_32(
        num_pred: torch.DoubleTensor,
        num_true: torch.DoubleTensor,
        **kwargs
    ) -> tuple[float, torch.BoolTensor, torch.FloatTensor]:
    return logSMAPE(num_pred, num_true, **kwargs, max_num_digits=6)

def token_eqality(
        *,
        y_pred: torch.LongTensor,
        y_true: torch.LongTensor,
        correct_threshold: float=1,
        **kwargs
    ) -> tuple[float, torch.BoolTensor, torch.FloatTensor]:
    """Calculate the token equality accuracy.
    Args:
        y_pred (LongTensor): The predicted tokens of shape (batch_size, seq_length)
        y_true (LongTensor): The true tokens of shape (batch_size, seq_length)
        correct_threshold (float): Threshold for correct prediction
        **kwargs: Additional arguments

    Returns:
        accuracy (float): The accuracy
        correct_samples (BoolTensor): The correct samples
        equality (FloatTensor): The number of correct predictions divided by the total number of predictions
    """
    equality = torch.eq(y_pred, y_true).float()
    correct_samples = equality.mean(dim=-1) >= correct_threshold
    return torch.mean(equality).item(), correct_samples, equality.mean(dim=-1)

def exact_number_acc(
        *,
        num_pred: torch.DoubleTensor,
        num_true: torch.DoubleTensor,
        significant_digits: int=14,
        eps: float=1e-100,
        y_true: Optional[torch.LongTensor]=None,
        **kwargs
    ) -> tuple[float, torch.BoolTensor, torch.FloatTensor]:
    r"""Rounds y_pred to d decimals and calculate the exact number accuracy.
    Args:
        num_pred (DoubleTensor): The predicted values
        num_true (DoubleTensor): The true values
        significant_digits (int): Number of significant digits to consider for equality
        eps (float): A small value to prevent division by zero
        y_true (LongTensor, optional): The true tokens of shape (batch_size, seq_length). Used to determine the batch size if num_pred is empty.
        **kwargs: Additional arguments

    Returns:
        accuracy (float): The accuracy
        correct_samples (BoolTensor): The correct samples
        diff (DoubleTensor): The absolute difference between the predicted and true values
    """
    if num_pred.numel() == 0:
        assert y_true is not None, "y_true must be provided if num_pred is empty"
        return torch.nan, torch.full((y_true.size(0),), torch.nan), torch.full((y_true.size(0),), torch.nan)
    max_decimals = torch.zeros_like(num_true).fill_(significant_digits)
    min_decimals = torch.zeros_like(num_true)
    num_decimals = -torch.minimum(torch.maximum(significant_digits-torch.ceil(torch.log10(torch.abs(num_true) + eps)), min_decimals), max_decimals)
    diff = torch.abs(num_pred - num_true)
    correct_samples = diff < (10**num_decimals)
    acc = correct_samples.float().mean()
    return acc.item(), correct_samples, correct_samples.float()

def normalized_class_acc(
        *,
        y_pred: torch.LongTensor,
        y_true: torch.LongTensor,
        num_classes: float=2.8,
        **kwargs) -> tuple[float, torch.BoolTensor, torch.FloatTensor]:
    r"""Calculate the normalized class accuracy. The accuracy is 0 if the prediction is random and 1 if the prediction is perfect.

    Args:
        y_pred (LongTensor): The predicted classes of shape (batch_size,)
        y_true (LongTensor): The true classes of shape (batch_size,)
        num_classes (float): Number of classes on average

    Returns:
        accuracy (float): The accuracy
        correct_samples (BoolTensor): The correct samples
        equality (FloatTensor): The number of correct predictions divided by the total number of predictions
    """
    acc, correct_samples, _ = token_eqality(y_pred=y_pred[:,:-1], y_true=y_true[:,:-1]) # pyright: ignore[reportArgumentType]
    return max(0,(num_classes * acc-1)/(num_classes-1)), correct_samples, correct_samples.float()

def normalized_bin_class_acc(**kwargs) -> tuple[float, torch.BoolTensor, torch.FloatTensor]:
    return normalized_class_acc(num_classes=2, **kwargs)

def normalized_quint_class_acc(**kwargs) -> tuple[float, torch.BoolTensor, torch.FloatTensor]:
    return normalized_class_acc(num_classes=5, **kwargs)

def scaled_ppl(
        *,
        logits: torch.FloatTensor,
        y_pred: torch.LongTensor,
        y_true: torch.LongTensor,
        correct_threshold: float=1,
        min_ppl: float=40,
        ppl_scale: float=100,
        **kwargs
    ) -> tuple[float, torch.BoolTensor, torch.FloatTensor]:
    """ Scales the perplexity to the range [0,1] where 1 is the best and 0 is the worst.
    This function assumes that perplexity is larger than min_ppl.

    Args:
        logits (FloatTensor): The logits of the predicted values of shape (batch_size, seq_length, num_classes)
        y_pred (LongTensor): The predicted values of shape (batch_size, seq_length)
        y_true (LongTensor): The true values of shape (batch_size, seq_length)
        correct_threshold (float): Threshold for correct prediction
        min_ppl (float): Minimum perplexity
        ppl_range (float): The range of perplexity
        **kwargs: Additional arguments

    Returns:
        accuracy (float): The accuracy
        correct_samples (BoolTensor): The correct
        equality (FloatTensor): The number of correct predictions divided by the total number of predictions
    """
    mask = y_true != -100
    ppl = torch.nn.functional.cross_entropy(logits[mask], y_true[mask]).exp().item()
    ppl_score = exp((min_ppl-ppl)/ppl_scale)
    equality = torch.eq(y_pred, y_true).float().mean(dim=-1)
    return min(1., ppl_score), equality >= correct_threshold, equality

def scaled_ppl_hard(
        *,
        logits: torch.FloatTensor,
        y_pred: torch.LongTensor,
        y_true: torch.LongTensor,
        correct_threshold: float=1,
        ppl_scale: float=100,
        **kwargs
    ) -> tuple[float, torch.BoolTensor, torch.FloatTensor]:
    """ Scales the perplexity to the range [0,1] where 1 is the best and 0 is the worst.
    This function assumes that perplexity is larger than min_ppl.

    Args:
        logits (FloatTensor): The logits of the predicted values of shape (batch_size, seq_length, num_classes)
        y_pred (LongTensor): The predicted values of shape (batch_size, seq_length)
        y_true (LongTensor): The true values of shape (batch_size, seq_length)
        correct_threshold (float): Threshold for correct prediction
        min_ppl (float): Minimum perplexity
        ppl_range (float): The range of perplexity
        **kwargs: Additional arguments

    Returns:
        accuracy (float): The accuracy
        correct_samples (BoolTensor): The correct
        equality (FloatTensor): The number of correct predictions divided by the total number of predictions
    """
    return scaled_ppl(logits=logits, y_pred=y_pred, y_true=y_true, correct_threshold=correct_threshold, min_ppl=20, ppl_scale=ppl_scale, **kwargs)

def sig_bits_acc(
        *,
        num_pred: torch.DoubleTensor,
        num_true: torch.DoubleTensor,
        max_bits: int=53,
        correct_threshold: float=0.95,
        eps: float=1e-100,
        y_true: Optional[torch.LongTensor]=None,
        **kwargs
    ) -> tuple[float, torch.BoolTensor, torch.FloatTensor]:
    r"""Calculate the accuracy based on the number of correct significant bits.
    
    This metric compares the binary representation of predicted and true values
    to determine how many significant bits match, providing a measure of numerical
    precision in the prediction.
    
    Args:
        num_pred (DoubleTensor): The predicted values
        num_true (DoubleTensor): The true values  
        max_bits (int): Maximum number of significant bits to consider (default 53 for double precision)
        correct_threshold (float): Threshold for correct prediction
        eps (float): A small value to prevent division by zero
        y_true (LongTensor, optional): The true tokens of shape (batch_size, seq_length). Used to determine the batch size if num_pred is empty.
        **kwargs: Additional arguments
        
    Returns:
        accuracy (float): The mean fraction of correct significant bits
        correct_samples (BoolTensor): The correct samples based on threshold
        acc (FloatTensor): The fraction of correct significant bits for each sample
    """
    if num_pred.numel() == 0:
        assert y_true is not None, "y_true must be provided if num_pred is empty"
        return torch.nan, torch.full((y_true.size(0),), torch.nan), torch.full((y_true.size(0),), torch.nan)
    # Handle edge cases where both values are zero or very close to zero
    both_zero_mask = (torch.abs(num_true) < eps) & (torch.abs(num_pred) < eps)
    
    # Handle perfect matches (exact equality)
    perfect_match_mask = (num_pred == num_true) & (~both_zero_mask)
    
    # For non-zero values, calculate the relative error
    rel_error = torch.abs((num_pred - num_true) / (torch.abs(num_true) + eps))
    
    # Initialize correct_bits tensor
    correct_bits = torch.zeros_like(rel_error)
    
    # For perfect matches or both zero, assign max_bits
    correct_bits[both_zero_mask | perfect_match_mask] = max_bits
    
    # For cases with actual errors, calculate bits based on relative error
    error_mask = (~both_zero_mask) & (~perfect_match_mask) & (rel_error > eps)
    if error_mask.any():
        # If relative error is 2^(-k), then we have k correct significant bits
        correct_bits[error_mask] = torch.clamp(-torch.log2(rel_error[error_mask]), min=0, max=max_bits)
    
    # Calculate accuracy as fraction of maximum possible bits
    acc = correct_bits / max_bits
    
    # Replace nan values with 0
    acc = acc.nan_to_num(0.0)
    
    correct_samples = acc > correct_threshold
    return acc.mean().item(), correct_samples, acc.float()



def generalized_mean(
        *,
        task_accs: torch.FloatTensor,
        p: float=1,
        eps: float=1e-2,
        **kwargs
    ) -> float:
    r"""Calculate the generalized mean of the accuracies.
    .. math::
        \text{Generalized Mean} = \left(\frac{1}{n}\sum_{i=1}^n\left(x_i+\epsilon\right)^p\right)^{\frac{1}{p}}

    where :math:`x_i` is the accuracy of the i-th task and :math:`\epsilon` is a small value to prevent division by zero.
    Args:
        task_accs (FloatTensor): The accuracies of the tasks of shape (num_tasks,)
        p (float): The power of the mean. Example: p=1 is the arithmetic mean, p=0 is the geometric mean, p=-1 is the harmonic mean
        eps (float): A small value to prevent division by zero
        **kwargs: Additional arguments
    Returns:
        accuracy (float): The generalized mean of the accuracies
    """
    if p == 0:
        return task_accs.clamp(min=eps).log().mean().exp().item() # Special case for geometric mean
    return task_accs.clamp(min=eps).pow(p).mean().pow(1/p).item()

def inverse_generalized_mean(
        *,
        task_accs: torch.FloatTensor,
        p: float=1,
        eps: float=1e-2,
        **kwargs
    ) -> torch.FloatTensor:
    r"""Calculate the normalized importance score to improve the generalized mean of the accuracies based on the possible improvement.
    .. math::
        \text{Inverse Generalized Mean} I(\tau_k) =\frac{(1-\rho(\tau))^{1-\lambda}}{\sum_{\tau' \in T} (1-\rho(\tau'))^{1-\lambda}}

    where :math:`x_i` is the accuracy of the i-th task and :math:`\epsilon` is a small value to prevent division by zero.
    Args:
        task_accs (FloatTensor): The accuracies of the tasks of shape (num_tasks,)
        p (float): The power of the mean. Example: p=1 is the arithmetic mean, p=0 is the geometric mean, p=-1 is the harmonic mean
        eps (float): A small value to prevent division by zero
        **kwargs: Additional arguments
    Returns:
        accuracy (float): The inverse generalized mean of the accuracies
    """
    return (1-task_accs).clamp(min=eps).pow(1-p) / (1-task_accs).clamp(min=eps).pow(1-p).sum()
    
class MetricFunction:
    TOKEN_EQUALITY = "token_eqality"
    S_MAPE= "sMAPE"
    LOG_SMAPE = "logSMAPE"
    LOG_SMAPE_32 = "logSMAPE_32"
    LOG_SMAPE_BASE2 = "logSMAPE_base2"
    EXACT_NUMBER_ACC = "exact_number_acc"
    NORMALIZED_CLASS_ACC = "normalized_class_acc"
    NORMALIZED_BIN_CLASS_ACC = "normalized_bin_class_acc"
    NORMALIZED_QUINT_CLASS_ACC = "normalized_quint_class_acc"
    SCALED_PPL = "scaled_ppl"
    SCALED_PPL_HARD = "scaled_ppl_hard"
    SIG_BITS_ACC = "sig_bits_acc"
    
    def __init__(self, func_name):
        try:
            self.func: Callable = globals()[func_name]
        except KeyError:
            raise ValueError(f"Unknown function: {func_name}")
        
    def __str__(self):
        return self.func.__name__
    
    def __repr__(self):
        return self.func.__name__
    
    def __call__(
        self, 
        *,
        y_pred: Optional[torch.LongTensor] = None,
        y_true: Optional[torch.LongTensor] = None,
        logits: Optional[torch.FloatTensor] = None,
        num_pred: Optional[torch.DoubleTensor] = None,
        num_true: Optional[torch.DoubleTensor] = None,
        correct_threshold: Optional[float] = None,
        num_classes: Optional[float] = None,
        mag: Optional[int] = None,
        **kwargs
    ) -> tuple[float, torch.BoolTensor, torch.FloatTensor]:
        """
        Abstract method to calculate the accuracy of the given predictions.
        Args:
            y_pred (LongTensor): The predicted tokens of shape (batch_size, seq_length)
            logits (torch.FloatTensor): The logits of shape (batch_size, seq_length, num_classes)
            y_true (LongTensor): The true tokens of shape (batch_size, seq_length)
            num_pred (DoubleTensor): The predicted of shape values (batch_size, seq_length)
            num_true (DoubleTensor): The true values of shape (batch_size, seq_length)
            correct_threshold (float): Threshold for correct prediction
            num_classes (float): Number of classes on average
            mag (int): The magnitude of the error where accuracy is 0.5
            **kwargs: Additional arguments
        Returns:
            accuracy (float): The accuracy
            correct_samples (BoolTensor): The correct samples
            acc (FloatTensor): The accuracy/error for each sample in the batch
        """
        kwargs = {k: v for k, v in locals().items() if k not in ["self", "kwargs"] and v is not None}
        return self.func(**kwargs)

class CostSensitiveLoss(torch.nn.Module):
    def __init__(self, cost_matrix: torch.FloatTensor, reduction="mean"):
        """
        Initialize the CostSensitiveLoss with a cost matrix.
        Args:
            cost_matrix (FloatTensor): A cost matrix of shape (num_classes, num_classes)
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super().__init__()
        self.register_buffer("cost_matrix", cost_matrix)  # (C, C)
        self.cost_matrix: torch.FloatTensor
        self.reduction = reduction

    def forward(self, logits: torch.FloatTensor, targets: torch.LongTensor) -> torch.FloatTensor:
        """
        Compute the cost-sensitive loss using the provided cost matrix.
        Args:
            logits (FloatTensor): The unnormalized logits of shape (batch_size, num_classes)
            targets (LongTensor): The true class indices of shape (batch_size,)
        Returns:
            loss (FloatTensor): The computed cost-sensitive loss
        """
        # softmax probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)  # (N, C)

        # lookup cost rows for each target
        # cost_rows[i] = C[y_i, :]
        cost_rows = self.cost_matrix[targets]  # (N, C)

        # expected cost per sample
        loss = torch.sum(probs * cost_rows, dim=-1)  # (N,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
