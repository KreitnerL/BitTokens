"""
Reusable test utility for number embedding modules.

This module provides a standardized testing framework for evaluating
the accuracy and robustness of number embedding implementations.
"""

import math
from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from eval_scripts.utils import logSMAPE
from networks.number_embedding_modules.abc_embedding import ABCEmbedding


class DummyModelOutput:
    """Mock model output class for testing."""
    def __init__(self):
        self.hidden_states = None


def test_number_embedding(
    embedding_module: ABCEmbedding,
    max_value: float = 1e14,
    tiny_value: float = 1e-14,
    relative_tolerance: float = 1e-14,
    absolute_tolerance: float = 1e-14,
    epsilon_for_tiny: bool = False,
    bottleneck_dtype=torch.bfloat16,
    noise_level: float = 0.0,
    bernoulli_noise_level: float = 0.0,
    grace_factor: int = 1,
    samples_per_exponent: int = 1000,
    fixed_test_values: Optional[list] = None,
    verbose: bool = True,
    test_name: str = "Custom test",
    seed: int = 42,
    input_emb_mode = "zeros",
    clamp=True,
    encoding_to_prediction_map=None
):
    """
    Test a number embedding module for accuracy and robustness.
    
    Args:
        embedding_module: The embedding module to test (should inherit from ABCEmbedding)
        max_value (float): Maximum value to test
        tiny_value (float): Smallest value to test
        relative_tolerance (float): Relative tolerance for comparison
        absolute_tolerance (float): Absolute tolerance for comparison
        epsilon_for_tiny (bool): Whether to use epsilon for tiny values
        bottleneck_dtype: Data type for bottleneck testing
        noise_level (float): Level of white noise to add (0.0 to 1.0)
        bernoulli_noise_level (float): Level of bernoulli noise to add (0.0 to 1.0)
        grace_factor (int): Multiplier for tolerance values
        samples_per_exponent (int): Number of random samples per exponent
        fixed_test_values (list): Additional fixed values to test
        verbose (bool): Whether to print detailed output
        test_name (str): Name of the test for logging
        seed (int): Random seed for reproducible results
        encoding_to_prediction_map (callable): Optional function to map encoding to prediction format.
                                             Default is identity. For cross-entropy models, use one-hot mapping.
        
    Returns:
        dict: Test results containing accuracy metrics and error statistics
    """
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set up mock model output
    out: CausalLMOutputWithCrossAttentions = DummyModelOutput()
    
    # Default fixed test values if none provided
    if fixed_test_values is None:
        fixed_test_values = [*np.array([
            -max_value, -5, -1.000001, -0.999999999, -tiny_value, 0, 
            tiny_value, 0.999999999, 1, 1.000001, math.pi, max_value, 1e-7
        ], dtype=np.float64)]
    
    # Override num_head for testing (flatten to avoid prediction complexity)
    original_num_head = embedding_module.num_head
    embedding_module.num_head = torch.nn.Flatten(start_dim=1)
    num_tests = 0
    try:
        print(f"Test name: {test_name}, embedding name: {embedding_module.__class__.__name__}")
        if verbose:
            print(f"White noise: {noise_level:.2%}, Binomial noise: {bernoulli_noise_level:.2%}")
            print(f"Tolerance: {relative_tolerance * grace_factor:.2e}")
        
        correct = []
        wrong = []
        log_smapes = []
        
        # Test across different orders of magnitude
        for e in tqdm(
            range(math.ceil(math.log10(tiny_value)), int(math.log10(max_value))), 
            desc="Running tests...",
            disable=not verbose
        ):
            # Prepare test values for this exponent
            test_values = fixed_test_values.copy()
            fixed_test_values = []  # Clear for next iteration
            
            # Add random values at this exponent
            for sign in [-1, 1]:
                for m in np.random.uniform(1, 10, size=samples_per_exponent):
                    test_values.append(sign * m * 10**e)
            
            # Convert to tensor
            x_tensor: torch.DoubleTensor = torch.tensor(test_values, dtype=torch.float64)
            
            # Forward pass: encode numbers
            num_encodings: torch.FloatTensor  = embedding_module.forward(x_tensor).unsqueeze(1)
            number_mask: torch.BoolTensor = torch.ones(
                (num_encodings.shape[0], 1), 
                dtype=torch.bool, 
                device=num_encodings.device
            )
            
            # Combine embeddings
            if input_emb_mode == "zeros":
                fun = torch.zeros
            elif input_emb_mode == "ones":
                fun = torch.ones
            inputs_embeds: torch.FloatTensor = fun(
                    num_encodings.shape[0], 
                    1, 
                    embedding_module.n_embed, 
                    dtype=bottleneck_dtype, 
                    device=num_encodings.device
                )
            combined_embeds = embedding_module.combine_embeds(
                inputs_embeds=inputs_embeds,
                num_encoding=num_encodings,
                number_mask=number_mask
            )
            
            # Extract the relevant part (excluding padding)
            pad_size = embedding_module.pad_size if hasattr(embedding_module, 'pad_size') else -1
            enc = combined_embeds[:, 0, :-pad_size]
            
            # Apply encoding to prediction mapping if provided
            if encoding_to_prediction_map is not None:
                enc = encoding_to_prediction_map(enc)
            
            # Add noise for robustness testing
            if noise_level > 0:
                noise = torch.zeros(enc.shape, dtype=bottleneck_dtype).uniform_(-1, 1) * noise_level
                enc = enc + noise
            if clamp:
                enc = enc.clamp(-1, 1)
            
            if bernoulli_noise_level > 0:
                bernoulli_noise = enc.clone().bernoulli(bernoulli_noise_level).bool()
                enc[bernoulli_noise] = enc.clone().uniform_(-1., 1.)[bernoulli_noise]
            
            # Prepare mock output for decoding
            out.hidden_states = [torch.zeros(enc.shape[0], 1, enc.shape[1], dtype=bottleneck_dtype)]
            out.hidden_states[-1][..., -1, :] = enc # pyright: ignore[reportOptionalSubscript]
            
            # Decode back to numbers
            if num_tests==356013:
                pass
            decoded = embedding_module.decode(out, slice(None)) # pyright: ignore[reportArgumentType]
            
            # Evaluate results
            for i, (input_val, recon_val) in enumerate(zip(test_values, decoded)):
                recon_val = recon_val.item()
                log_smapes.append(logSMAPE(recon_val,input_val))
                
                # Check if reconstruction is within tolerance
                if math.isclose(
                    input_val, 
                    recon_val, 
                    rel_tol=relative_tolerance * grace_factor,
                    abs_tol=0 if epsilon_for_tiny else absolute_tolerance * grace_factor
                ):
                    correct.append((input_val, recon_val))
                else:
                    if noise_level == 0 and bernoulli_noise_level==0 and verbose:
                        rel_error = abs(np.longdouble(input_val) - np.longdouble(recon_val)) / abs(np.longdouble(input_val))
                        print(f"Reconstruction error: Input: {repr(input_val.item())}, Recon: {recon_val}, Relative error: {rel_error:.2e}")
                    wrong.append((num_tests, i, input_val, recon_val))
            num_tests += len(test_values)
        
        # Calculate results
        total_tests = len(correct) + len(wrong)
        accuracy = len(correct) / total_tests if total_tests > 0 else 0
        mean_log_smape = np.nanmean(log_smapes) if log_smapes else float('inf')
        
        # Print results
        print(f"Test results: Total: {total_tests}, Correct: {len(correct)}, Wrong: {len(wrong)}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Mean logSMAPE: {mean_log_smape:.2%}")
        
        return {
            'total_tests': total_tests,
            'correct': len(correct),
            'wrong': len(wrong),
            'accuracy': accuracy,
            'mean_log_smape': mean_log_smape,
            'correct_samples': correct,
            'wrong_samples': wrong,
            'log_smapes': log_smapes
        }
        
    finally:
        # Restore original num_head
        embedding_module.num_head = original_num_head

def run_standard_test(embedding_module: ABCEmbedding, **kwargs):
    """
    Run the standard test configuration for number embedding modules.
    Args:
        embedding_module: The embedding module to test
    Returns:
        dict: Test results
    """
    return test_number_embedding(embedding_module=embedding_module, test_name="Standard test", **kwargs)


def run_dtype_test(embedding_module: ABCEmbedding, dtype: torch.dtype, encoding_to_prediction_map=None, **kwargs):
    """
    This test style uses dtype-based precision limits and different default parameters:
    - Uses torch.finfo() to determine MAX, TINY, and EPSILON values based on torch.float64
    - Uses EPSILON for tolerance checking instead of relative_tolerance
    - Higher default noise level (0.3 vs 0.0)
    
    Args:
        embedding_module: The embedding module to test
        dtype: The data type to use for testing (e.g., torch.float64)
        **kwargs: Additional arguments to override defaults
        
    Returns:
        dict: Test results
    """
    # Calculate precision limits from dtype
    finfo = torch.finfo(dtype)
    max_value = finfo.max * (1 - 1e-11)
    tiny_value = finfo.tiny * 2
    epsilon = finfo.eps * 2
    
    default_config = {
        'max_value': max_value,
        'tiny_value': tiny_value,
        'relative_tolerance': epsilon,  # Use EPSILON for tolerance
        'absolute_tolerance': epsilon,
        'epsilon_for_tiny': True,  # epsilon for tiny values
        'noise_level': 0.3,  # Higher noise level
        'test_name': f"{str(dtype).split('.')[-1]} Test",
        'encoding_to_prediction_map': encoding_to_prediction_map,
    }
    
    # Override defaults with provided kwargs
    config = {**default_config, **kwargs}
    
    return test_number_embedding(embedding_module, **config)


def run_minimal_test(embedding_module: ABCEmbedding, **kwargs):
    """
    Only run a minimal test with a few fixed values.
    Args:
        embedding_module: The embedding module to test
    Returns:
        dict: Test results
    """
    return test_number_embedding(embedding_module, **{"samples_per_exponent": 0, "max_value": 10,"tiny_value": 1, **kwargs})

def run_specific_test(embedding_module: ABCEmbedding, fixed_test_values: list, **kwargs):
    """
    Only run a minimal test with a few fixed values.
    Args:
        embedding_module: The embedding module to test
    Returns:
        dict: Test results
    """
    return test_number_embedding(embedding_module, samples_per_exponent=0, max_value=10,tiny_value=1, fixed_test_values=fixed_test_values, **kwargs)
