if __name__ == "__main__":
    # Add project root to path when running directly 
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from math import sqrt
from typing import override

import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from networks.number_embedding_modules.abc_embedding import ABCEmbedding
from utils.enums import COMBINE_STRATEGY
from utils.metrics import CostSensitiveLoss


class Base10Embedding(ABCEmbedding):
    """
    Number embedding module that represents numbers using base-10 scientific notation.
    Numbers are represented as: number = M × 10^E where:
    - 1 digit for the sign (0 for positive, 1 for negative)
    - 3 digits for the exponent (allows range 10^-499 to 10^499, biased by 500)
    - 17 digits for the mantissa (covers full float64 precision of 15-17 digits with safety margin)
    
    This results in a 22-dimensional vector of decimal digits for each number.
    The embedding can optionally include the reciprocal (1/x) of the number as well.
    """
    
    def __init__(
            self,
            n_embed: int,
            device="cuda:0",
            add_reciprocal: bool=True,
            combination_method: COMBINE_STRATEGY="sum",
            loss_type: str ="ce",
            mantissa_digits: int=17,
            exponent_digits: int=3
        ):
        """
        Initialize the Base10Embedding module.
        
        Args:
            n_embed (int): Embedding dimension of the model
            device (str): Device to run computations on
            add_reciprocal (bool): Whether to include 1/x in the embedding
            combination_method (COMBINE_STRATEGY): How to combine with input embeddings
            loss_type (Literal["bce", "mse", "l1", "ce"]): Loss function type
            mantissa_digits (int): Number of mantissa digits (18 covers float64 precision)
            exponent_digits (int): Number of exponent digits (3 allows ±499 range)
        """
        super().__init__()
        self.n_embed = n_embed
        self.device = device
        self.device_str = str(device).split(":")[0]
        self.add_reciprocal = add_reciprocal
        self.loss_type = loss_type
        self.mantissa_digits = mantissa_digits
        self.exponent_digits = exponent_digits
        
        # Digit allocation
        self.sign_digits = 1
        self.freq_size = self.sign_digits + self.exponent_digits + self.mantissa_digits
        
        # Exponent bias (to handle negative exponents)
        # For 3 digits: bias = 500, allowing range [-499, 500]
        self.exponent_bias = 10 ** (exponent_digits - 1) * 5
        
        # Loss function setup
        if loss_type == "ce":
            self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")
        elif loss_type == "cs":
            # Cost-sensitive loss to consider the distance between digits using rotary schema
            # In rotary schema, digits wrap around: distance between 9 and 1 is min(8, 2) = 2
            linear_distance = torch.abs(torch.arange(10).unsqueeze(1) - torch.arange(10).unsqueeze(0)).float()
            wrap_distance = 10 - linear_distance
            distance_matrix: torch.FloatTensor = torch.min(linear_distance, wrap_distance)
            distance_matrix = distance_matrix / distance_matrix.max()  # Normalize to [0, 1]
            self.loss_func = CostSensitiveLoss(cost_matrix=distance_matrix, reduction="none")
        elif loss_type == "mse":
            raise NotImplementedError("MSE loss not implemented for Base10Embedding.")
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        # Embedding size calculation - now using one-hot encoding
        self.embedding_size = self.freq_size * 10  # One-hot: 22 digits × 10 classes = 220
        assert n_embed >= self.embedding_size * (2 if self.add_reciprocal else 1), (
            f"n_embed ({n_embed}) must be at least {self.embedding_size * (2 if self.add_reciprocal else 1)} "
            f"to accommodate {'number and reciprocal' if self.add_reciprocal else 'number only'} embeddings."
        )
            
        self.pad_size = n_embed - self.embedding_size * (2 if self.add_reciprocal else 1)
        self.combination_method: COMBINE_STRATEGY = combination_method
        
        # Neural network head for decoding
        if loss_type in ["ce", "cs"]:
            # Predict digit probabilities (10 classes per digit position)
            self.num_head = torch.nn.Linear(n_embed, self.freq_size * 10)
        else:
            raise NotImplementedError(f"Loss type {loss_type} not implemented for Base10Embedding.")
        
        # Precomputed powers for efficiency
        self.exponent_powers = torch.tensor([10**i for i in range(exponent_digits-1, -1, -1)], dtype=torch.int64, device=device)
        self.mantissa_powers = torch.tensor([10**i for i in range(mantissa_digits-1, -1, -1)], dtype=torch.float64, device=device)
        
        # Precomputed powers for digit extraction optimization
        self.exponent_digit_powers = torch.tensor([10**i for i in range(exponent_digits-1, -1, -1)], dtype=torch.int64, device=device)
        self.mantissa_digit_powers = torch.tensor([10**i for i in range(mantissa_digits-1, -1, -1)], dtype=torch.int64, device=device)
        
        # Precomputed constants for float_to_base10_digits optimization
        self.zero_tensor_f64 = torch.tensor(0.0, dtype=torch.float64, device=device)
        self.tiny_epsilon_f64 = torch.tensor(1e-300, dtype=torch.float64, device=device)
        self.zero_tensor_i32 = torch.tensor(0, dtype=torch.int32, device=device)
        self.bias_tensor_i32 = torch.tensor(self.exponent_bias, dtype=torch.int32, device=device)
        self.mantissa_scale_f64 = torch.tensor(10.0 ** (self.mantissa_digits-1), dtype=torch.float64, device=device)
        
        # Precomputed tensors for loss computation optimization
        self.digit_indices = torch.arange(10, device=device).unsqueeze(0)  # Shape: (1, 10) for broadcasting
        
        # Loss weights for each digit position (can be adjusted for importance)
        self.freq_loss_weights = torch.ones(1, self.freq_size, dtype=torch.float32, device=device)

        if self.combination_method == "weighted" or self.combination_method == "weighted_sum":
            self.bit_weight_matrix = torch.nn.Linear(self.embedding_size, n_embed, bias=False)

    def float_to_base10_digits(self, float_numbers: torch.Tensor) -> torch.LongTensor:
        """Convert float numbers to base-10 digit representation."""
        # Ensure the input is on the same device as the embedding
        float_numbers = float_numbers.to(self.device)
        
        # Handle special cases
        is_zero = (float_numbers == 0.0)
        is_positive = (float_numbers >= 0.0)  # Include zero as positive
        
        # Get absolute values for processing
        abs_numbers = torch.abs(float_numbers)
        
        # Calculate exponent using proper scientific notation normalization
        # Use small epsilon to avoid log(0) - using precomputed constant
        safe_abs = torch.where(is_zero, self.tiny_epsilon_f64, abs_numbers)
        log10_abs = torch.where(is_zero, self.zero_tensor_f64, torch.log10(safe_abs))
        
        # For scientific notation M×10^E where 1 ≤ M < 10, exponent is floor(log10(abs_number))
        exponent = torch.where(
            is_zero,
            self.zero_tensor_i32,
            torch.floor(log10_abs).to(torch.int32)
        )
        
        # Bias the exponent by 500 to make it positive - using precomputed constant
        biased_exponent = exponent + self.bias_tensor_i32
        
        # Calculate mantissa in proper scientific notation: M = abs_number / 10^exponent
        # This ensures M is always in range [1.0, 10.0) for non-zero numbers
        mantissa_multiplier = torch.where(
            is_zero,
            self.zero_tensor_f64,
            abs_numbers / (10.0 ** exponent.to(torch.float64))
        )
        
        # Now mantissa_multiplier is in [1.0, 10.0), so we multiply by 10^17 to get 18 digits
        # This gives us digits like: 1.23456... -> 123456...000000000000000000
        # Using precomputed scale constant
        scaled_mantissa = mantissa_multiplier * self.mantissa_scale_f64
        mantissa_int = torch.round(scaled_mantissa).to(torch.int64)
        
        # Use vectorized digit extraction for better performance
        batch_size = float_numbers.shape[0]
        result = torch.zeros((batch_size, self.freq_size), dtype=torch.long, device=self.device)
        
        # Sign digit (0 for negative, 1 for positive or zero)
        result[:, 0] = is_positive.long()
        
        # Extract exponent digits using vectorized method with precomputed powers
        exp_digits = self._extract_decimal_digits(biased_exponent.long(), self.exponent_digit_powers)
        result[:, 1:1+self.exponent_digits] = exp_digits
        
        # Extract mantissa digits using vectorized method with precomputed powers
        mantissa_digits = self._extract_decimal_digits(mantissa_int, self.mantissa_digit_powers)
        result[:, 1+self.exponent_digits:] = mantissa_digits
        
        return result

    def _extract_decimal_digits(self, numbers: torch.Tensor, digit_powers: torch.Tensor) -> torch.LongTensor:
        """
        Extract decimal digits from integer numbers using vectorized operations.
        Fully vectorized version that avoids loops for better performance.
        
        Args:
            numbers (torch.LongTensor): Integer numbers to extract digits from
            digit_powers (torch.Tensor): Precomputed powers of 10 for digit extraction
            
        Returns:
            torch.LongTensor: Digits with shape (*numbers.shape, num_digits)
        """
        # Reshape numbers for broadcasting: (*batch_shape, 1)
        numbers_expanded = numbers.unsqueeze(-1)
        
        # Extract digits using integer division and modulo operations
        # For each power of 10, get the digit at that position
        digits = (numbers_expanded // digit_powers) % 10
        
        return digits

    def digits_to_float64(self, digits: torch.Tensor) -> torch.DoubleTensor:
        """
        Reconstruct float64 numbers from base-10 digit representation.
        
        Args:
            digits (torch.LongTensor): Digit representation with shape (N, total_digits)
            
        Returns:
            torch.DoubleTensor: Reconstructed numbers with shape (N,)
        """
        # Split digits into components
        sign_digits = digits[:, 0]
        exponent_digits = digits[:, 1:1+self.exponent_digits]
        mantissa_digits = digits[:, 1+self.exponent_digits:]
        
        # Reconstruct exponent from digits and remove bias
        exponent_biased = torch.sum(exponent_digits * self.exponent_powers, dim=-1)
        exponent = exponent_biased - self.exponent_bias
        
        # Reconstruct mantissa from digits
        mantissa_int = torch.sum(mantissa_digits.double() * self.mantissa_powers, dim=-1)
        mantissa = mantissa_int / (10 ** (self.mantissa_digits - 1))
        
        # Handle zero case
        is_zero = (mantissa == 0.0)
        
        # Reconstruct number: M × 10^E
        abs_number = torch.where(
            is_zero,
            torch.tensor(0.0, dtype=torch.float64, device=self.device),
            mantissa * (10.0 ** exponent.to(torch.float64))
        )
        
        # Apply sign: 0 = negative, 1 = positive (fixed the logic)
        result = torch.where(sign_digits.bool(), abs_number, -abs_number)
        return result

    @override
    def forward(self, x: torch.Tensor) -> torch.FloatTensor | torch.BFloat16Tensor:
        """Encode numbers to base-10 one-hot digit representation"""
        # Get raw digit representation
        x_digits = self.float_to_base10_digits(x)  # Shape: (batch_size, total_digits)
        
        # Convert to one-hot encoding: (batch_size, total_digits, 10) -> (batch_size, total_digits * 10)
        x_encoding = torch.nn.functional.one_hot(x_digits, num_classes=10).float()
        x_encoding = x_encoding.view(x_digits.shape[0], -1)  # Flatten to (batch_size, 220)
        
        if self.add_reciprocal:
            # Compute reciprocal - this will naturally give inf for zero, which is mathematically correct
            x_reciprocal = x.reciprocal()
            x_reciprocal_digits = self.float_to_base10_digits(x_reciprocal.to(torch.float64))
            x_reciprocal_encoding = torch.nn.functional.one_hot(x_reciprocal_digits, num_classes=10).float()
            x_reciprocal_encoding = x_reciprocal_encoding.view(x_reciprocal_digits.shape[0], -1)
            
            # Concatenate main number and reciprocal encodings
            combined = torch.cat([x_encoding, x_reciprocal_encoding], dim=-1)
        else:
            combined = x_encoding
        
        return combined
    
    @override
    def combine_embeds(
        self,
        inputs_embeds: torch.FloatTensor | torch.BFloat16Tensor,
        num_encoding: torch.FloatTensor | torch.BFloat16Tensor,
        number_mask: torch.BoolTensor,
    ) -> torch.FloatTensor | torch.BFloat16Tensor:
        combined_embeds = inputs_embeds.clone()
        num_encoding = num_encoding.to(inputs_embeds.dtype)
        num_encoding[number_mask] = num_encoding[number_mask]*2-1
        if self.combination_method == "sum":
            combined_embeds[number_mask] = inputs_embeds[number_mask] + torch.nn.functional.pad(num_encoding[number_mask], (0, self.pad_size), value=0.)
        elif self.combination_method == "prod":
            combined_embeds[number_mask] = inputs_embeds[number_mask] * torch.nn.functional.pad(num_encoding[number_mask], (0, self.pad_size), value=1.)
        elif self.combination_method == "sum_scaled":
            token_rms = torch.sqrt(torch.mean(inputs_embeds[number_mask]**2, dim=-1, keepdim=True)).to(inputs_embeds.dtype)
            combined_embeds[number_mask] = inputs_embeds[number_mask] + torch.nn.functional.pad(num_encoding[number_mask] * token_rms, (0, self.pad_size), value=0.)
        elif self.combination_method == "concat":
            token_rms = torch.sqrt(torch.mean(inputs_embeds[number_mask][:, self.embedding_size:]**2, dim=-1, keepdim=True))
            combined_embeds[number_mask] = torch.cat([
                num_encoding[number_mask],
                inputs_embeds[number_mask][:, self.embedding_size:] / token_rms.to(inputs_embeds.dtype)
            ], dim=1)
        elif self.combination_method == "zero_pad":
            combined_embeds[number_mask] = torch.nn.functional.pad(num_encoding[number_mask] * sqrt(self.n_embed / self.embedding_size), (0, self.pad_size), value=0.)
        elif self.combination_method == "weighted":
            combined_embeds[number_mask] = self.bit_weight_matrix(num_encoding[number_mask])
        elif self.combination_method == "weighted_sum":
            weighted_num_encoding = self.bit_weight_matrix(num_encoding[number_mask])
            combined_embeds[number_mask] = inputs_embeds[number_mask] + weighted_num_encoding
        else:
            raise NotImplementedError(f"Combination method {self.combination_method} not implemented.")
        return combined_embeds
    
    @override
    def compute_num_loss(
        self,
        out: CausalLMOutputWithCrossAttentions,
        num_encodings: torch.FloatTensor | torch.BFloat16Tensor,
        number_mask: torch.BoolTensor,
        numbers: torch.DoubleTensor,
        hidden_states_slice=slice(0, -1),
        **kwargs
    ) -> dict:
        """Compute loss for number prediction"""
        assert out.hidden_states is not None, "Model output must contain hidden states for number loss computation."
        
        # Get hidden states for positions with numbers
        hidden = out.hidden_states[-1][:, hidden_states_slice][number_mask].float()
        
        # Early return if no numbers to process
        if hidden.size(0) == 0:
            num_loss_per_sample = torch.zeros_like(number_mask, dtype=torch.float32, device=self.device)
            digit_loss_means = torch.zeros(self.freq_size, dtype=torch.float32, device=self.device)
            return {
                "num_loss": num_loss_per_sample,
                "num_loss_per_frequency": digit_loss_means,
            }
        
        # Cross-entropy loss for digit classification - keep flat for efficiency
        digit_logits_flat = self.num_head(hidden)  # Shape: (N, total_digits * 10)
        
        # Get target digits - work directly with one-hot encoding to avoid argmax
        target_encoding = num_encodings[number_mask][:, :self.embedding_size]  # Shape: (N, 220)
        target_one_hot = target_encoding.view(-1, self.freq_size, 10)  # Shape: (N, 22, 10)
        
        # Reshape logits to match target structure for efficient loss computation
        digit_logits = digit_logits_flat.view(-1, self.freq_size, 10)  # Shape: (N, 22, 10)
        
        target_digits = torch.argmax(target_one_hot, dim=-1)  # Shape: (N, 22)
        # Compute cross-entropy loss using the configured loss function
        digit_losses = self.loss_func(digit_logits.view(-1, 10), target_digits.view(-1))
        digit_losses = digit_losses.view(-1, self.freq_size)  # Shape: (N, 22)
        
        # Apply per-digit weights efficiently
        weighted_digit_losses = digit_losses * self.freq_loss_weights
        
        # Combine losses (weighted sum across digit positions)
        total_loss = weighted_digit_losses.sum(dim=1)
        digit_loss_means = weighted_digit_losses.detach().mean(dim=0)
        
        # Efficiently populate loss tensor using advanced indexing
        num_loss_per_sample = torch.zeros_like(number_mask, dtype=torch.float32, device=self.device)
        num_loss_per_sample[number_mask] = total_loss
        
        train_metrics = {
            "num_loss": num_loss_per_sample,
            "num_loss_per_frequency": digit_loss_means,
        }
        return train_metrics
    
    @override
    def decode(self, out: CausalLMOutputWithCrossAttentions, number_mask: torch.BoolTensor) -> torch.DoubleTensor:
        """Decode hidden states back to numbers"""
        assert out.hidden_states is not None, "Model output must contain hidden states for decoding."
        
        # Get hidden states for positions with numbers
        hidden = out.hidden_states[-1][..., -1:, :][number_mask].float()
        head_output = self.num_head(hidden)
        
        # Always predict only the main number (first total_digits), never the reciprocal
        digit_logits = head_output.view(-1, self.freq_size, 10)
        predicted_digits = torch.argmax(digit_logits, dim=-1)
        
        # Reconstruct numbers from digits
        reconstructed_numbers = self.digits_to_float64(predicted_digits)
        return reconstructed_numbers


if __name__ == "__main__":
    # Test the Base10Embedding class using the dtype-based test framework
    from test_utils import run_dtype_test
    
    # Initialize embedding with cross-entropy loss (works better than MSE)
    base_embedding = Base10Embedding(
        n_embed=768, 
        device="cpu",
        add_reciprocal=False,  # Simplified for testing
        loss_type="ce"         # Cross-entropy works perfectly
    )

    # Run comprehensive dtype test (uses torch.float64 precision limits)
    print("\nRunning comprehensive dtype test with torch.float64 precision...")
    results = run_dtype_test(
        embedding_module=base_embedding,
        dtype=torch.float64,
        encoding_to_prediction_map=None,  # No conversion needed - forward already returns one-hot
        grace_factor=2,
    )
