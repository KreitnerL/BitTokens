import functools
from typing import Any, Callable, Optional

import torch


def dynamic_compile(func: Optional[Callable] = None, *, dynamic: Optional[bool] = None, **kwargs_compile):
    """
    Function that can be used as a decorator or directly to compile a function.
    When used as a decorator:
        @dynamic_compile
        def func(...): ...
        
        @dynamic_compile(dynamic=True, fullgraph=True)
        def func(...): ...
    
    When used directly:
        compiled_func = dynamic_compile(func)
        compiled_func = dynamic_compile(func, dynamic=True, fullgraph=True)
    
    This will compile a function for training but use dynamic compilation for inference.
    """
    def _compile(fn: Callable) -> Callable:
        # Compile the function for training
        compiled_func = torch.compile(fn, **kwargs_compile)
        dynamic_compiled_func = torch.compile(fn, dynamic=True, **kwargs_compile)
        
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return torch.cond(
                torch.is_grad_enabled(),
                lambda: compiled_func(*args, **kwargs),
                lambda: dynamic_compiled_func(*args, **kwargs)
            )
        
        return wrapper
    
    # If used as @dynamic_compile without parentheses
    if func is not None:
        return _compile(func)
    
    # If used as @dynamic_compile(...) with parentheses or as dynamic_compile(func, ...)
    return _compile