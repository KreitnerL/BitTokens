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
from utils.enums import COMBINE_STRATEGY, NUMBER_HEAD


class Float64Embedding(ABCEmbedding):
    """
    Number embedding module that represents numbers using the IEEE 754 float64 binary representation.
    - 1 bit for the sign (0 for positive, 1 for negative)
    - 11 bits for the exponent (biased by 1023)
    - 52 bits for the fraction (also known as the mantissa)
    This results in a 64-dimensional binary vector for each number. The embedding can optionally include the reciprocal (1/x) of the number as well.
    The embedding is the scaled to the range [-1, 1] and concatenated with its negation to form a 128-dimensional vector per number (256 if reciprocal is included).
    """
    
    def __init__(
            self,
            n_embed: int,
            device="cuda:0",
            add_reciprocal: bool=False,
            combination_method: COMBINE_STRATEGY="sum",
            loss_type: str="bce",
            frequency_weight_slope: float=0.,
            number_head_type: NUMBER_HEAD=NUMBER_HEAD.LINEAR,
            precision_type: torch.dtype=torch.float64,
        ):
        """
        Initialize the Float64Embedding module.
        
        Args:
            n_embed (int): Embedding dimension of the model
            device (str): Device to run computations on
            add_reciprocal (bool): Whether to include 1/x in the embedding
            combination_method (COMBINE_STRATEGY): How to combine embeddings
            loss_type (str): Loss function type - "bce" or "mse"
            frequency_weight_slope (float): Slope for frequency loss weighting
        """
        super().__init__()
        self.n_embed = n_embed
        self.device = device
        self.device_str = str(device).split(":")[0]
        self.add_reciprocal = add_reciprocal
        self.loss_type = loss_type
        self.precision_type = precision_type
        
        # Calculate embedding dimensions first
        if self.precision_type == torch.float64:
            self.freq_size = 64
        elif self.precision_type == torch.float32:
            self.freq_size = 32
        elif self.precision_type in [torch.float16, torch.bfloat16]:
            self.freq_size = 16
        else:
            raise ValueError(f"Unsupported precision type: {self.precision_type}")
        self.embedding_size = self.freq_size * (2 if self.add_reciprocal else 1)
        self.output_size = self.freq_size
        self.pad_size = n_embed - self.embedding_size
        self.combination_method: COMBINE_STRATEGY = combination_method

        num_head_list: list = []
        match number_head_type:
            case NUMBER_HEAD.LINEAR:
                num_head_list.append(torch.nn.Linear(n_embed, self.output_size))
            case NUMBER_HEAD.MLP:
                num_head_list.append(torch.nn.Linear(n_embed, n_embed//2))
                num_head_list.append(torch.nn.ReLU())
                num_head_list.append(torch.nn.Linear(n_embed//2, self.output_size))
            case NUMBER_HEAD.NONE:
                class ChannelSlice(torch.nn.Module):
                    def __init__(self, start=0, end=-1):
                        super().__init__()
                        self.start = start
                        self.end = end
                    def forward(self, x):
                        return x[:, self.start:self.end]
                num_head_list.append(ChannelSlice(0, self.output_size))
    
        match loss_type:
            case "bce":
                self.loss_func = torch.nn.BCEWithLogitsLoss(reduction="none")
            case "mse":
                num_head_list.append(torch.nn.Sigmoid())
                self.loss_func = torch.nn.MSELoss(reduction="none")
            case _:
                raise ValueError(f"Unsupported loss type: {loss_type}")
            
        self.num_head: torch.nn.Module = torch.nn.Sequential(*num_head_list)
        # self.num_head = torch.nn.Linear(n_embed, self.output_size)
            
        if self.combination_method == "weighted" or self.combination_method == "weighted_sum":
            self.bit_weight_matrix = torch.nn.Linear(self.embedding_size, n_embed, bias=False)
        
        self.float64_bit_shifts: torch.LongTensor = torch.arange(self.freq_size-1, -1, -1, device=device, dtype=torch.int64)
        self.freq_loss_weights = torch.linspace(1, 1+self.freq_size*frequency_weight_slope, self.freq_size, device=device, dtype=torch.float32).unsqueeze(0)
        self.freq_loss_weights = self.freq_loss_weights - self.freq_loss_weights.min() + 1
        self.freq_loss_weights = self.freq_loss_weights / self.freq_loss_weights.mean()

    # @torch.compile(fullgraph=True)
    def float64_tensor_to_binary_tensor(self, tensor_in: torch.DoubleTensor) -> torch.LongTensor:
        """
        Converts a float64 PyTorch tensor to its IEEE 754 binary representation,
        returning the result as a new PyTorch integer tensor of bits.

        Args:
            tensor_in (torch.DoubleTensor): A PyTorch tensor with dtype torch.float64.

        Returns:
            base_extension (torch.LongTensor): A PyTorch tensor of dtype torch.int64 with shape (*tensor_in.shape, 64)
        """
        int_representation = tensor_in.view(torch.int64).unsqueeze(-1)
        bits = (int_representation >> self.float64_bit_shifts) & 1
        return bits
    
    # @torch.compile(fullgraph=True)
    def binary_tensor_to_float64_tensor(self, bits_int64: torch.LongTensor) -> torch.DoubleTensor:
        """
        Reconstructs a float64 tensor from its IEEE 754 binary representation.
        This function is the inverse of `float64_tensor_to_binary_tensor`.
        Args:
            bits_int64: A PyTorch tensor with an integer dtype and shape (N, 64),
                        where the last dimension holds the 64 bits (0s or 1s)
                        of each float number, from most-significant to least-significant.
        Returns:
            num_tensor (torch.DoubleTensor): A tensor of reconstructed float64 values with shape (N).
        """
        exponents = self.float64_bit_shifts
        weights = torch.tensor(1, dtype=torch.int64, device=self.device) << exponents
        reconstructed_int = torch.sum(bits_int64 * weights, dim=-1)
        int_type = torch.int64 if self.precision_type == torch.float64 else torch.int32
        return reconstructed_int.to(dtype=int_type).view(self.precision_type).to(torch.float64)
    
    @override
    def forward(self, x: torch.DoubleTensor) -> torch.FloatTensor | torch.BFloat16Tensor:
        x_encoding = self.float64_tensor_to_binary_tensor(x)
        if self.add_reciprocal:
            x_reciprocal: torch.DoubleTensor = x.reciprocal()
            x_reciprocal_encoding = self.float64_tensor_to_binary_tensor(x_reciprocal)
        else:
            x_reciprocal_encoding = torch.empty(0, dtype=x_encoding.dtype, device=x_encoding.device)
        x_norm = torch.cat([x_encoding, x_reciprocal_encoding], dim=-1)
        return x_norm
    
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
    ) -> torch.FloatTensor:
        assert out.hidden_states is not None, "Model output must contain hidden states for number loss computation."
        x_base_digits_norm_pred = self.num_head(out.hidden_states[-1][:, hidden_states_slice][number_mask]).float()
        x_base_digits_norm: torch.DoubleTensor = num_encodings[number_mask][:,:self.output_size].float()
        
        frequency_loss: torch.Tensor = self.loss_func.forward(x_base_digits_norm_pred, x_base_digits_norm)
        frequency_loss = frequency_loss * self.freq_loss_weights
        
        num_loss_per_sample = torch.zeros_like(number_mask, dtype=torch.float32)
        num_loss_per_sample[number_mask] = frequency_loss.mean(dim=-1)
        train_metrics = {
            "num_loss": num_loss_per_sample,
            "num_loss_per_frequency": frequency_loss.detach().mean(dim=0),
        }
        return train_metrics
    
    @override
    def decode(self, out: CausalLMOutputWithCrossAttentions, number_mask: torch.BoolTensor) -> torch.DoubleTensor:
        assert out.hidden_states is not None, "Model output must contain hidden states for decoding."
        x_base_digits_norm_pred: torch.BFloat16Tensor = self.num_head(out.hidden_states[-1][...,-1:,:][number_mask]).double()
        threshold = 0 if self.loss_type == "bce" else 0.5
        x_base_digits_pred: torch.LongTensor = (x_base_digits_norm_pred > threshold).to(torch.int64)
        num_preds = self.binary_tensor_to_float64_tensor(x_base_digits_pred)
        return num_preds



if __name__ == "__main__":
    # Test the Float64Embedding class using the reusable test utility
    from test_utils import run_dtype_test
    
    # Test parameters
    BOTTLENECK_DTYPE = torch.bfloat16
    
    # Initialize embedding
    base_embedding = Float64Embedding(
        n_embed=384, 
        device="cpu",
    ).to(dtype=BOTTLENECK_DTYPE)
    
    results = run_dtype_test(embedding_module=base_embedding, dtype=torch.float64)
