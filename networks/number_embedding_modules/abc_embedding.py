from abc import ABC, abstractmethod

import torch.nn as nn
from torch import BFloat16Tensor, BoolTensor, DoubleTensor, FloatTensor
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class ABCEmbedding(ABC, nn.Module):
    n_embed: int
    num_head: nn.Module
    pad_size: int
    freq_size: int
    freq_loss_weights: FloatTensor | BFloat16Tensor

    @abstractmethod
    def forward(self, x: DoubleTensor) -> FloatTensor | BFloat16Tensor:
        """
        Encode a batch of numerical values.
        Args:
            x (DoubleTensor): The batch of numerical values to encode of shape (N,)
        Returns:
            num_encoding (FloatTensor): The number encoding of shape (N, E_f), where E_f is the number of encoding dimensions
        """
        pass

    @abstractmethod
    def combine_embeds(self, inputs_embeds: FloatTensor | BFloat16Tensor, num_encoding: FloatTensor | BFloat16Tensor, number_mask: BoolTensor) -> FloatTensor | BFloat16Tensor:
        """
        Combine the input embeddings with the number encoding.
        Args:
            inputs_embeds (LongTensor): The input embeddings of shape (B, S, E_t)
            num_encoding (FloatTensor | BFloat16Tensor): The encoding of the numbers of shape (B, S, E_f)
            number_mask (LongTensor): Boolean mask of num_tokens in output tokens (B, S)
        Returns:
            combined_embeds (FloatTensor | BFloat16Tensor): The combined embeddings of shape (B, S, E_t)
        """
        pass


    @abstractmethod
    def decode(self, out: CausalLMOutputWithCrossAttentions, number_mask: BoolTensor) -> DoubleTensor:
        """
        Decode the number encoding back to the original values
        Args:
            out (CausalLMOutputWithCrossAttentions): The output of the model containing the hidden states
            number_mask (BoolTensor): Boolean mask of num_tokens in output tokens (B, S)
        Returns:
            num_decoded (DoubleTensor): The decoded values of shape (N,)
        """
        pass
    
    @abstractmethod
    def compute_num_loss(
        self,
        out: CausalLMOutputWithCrossAttentions,
        num_encodings: FloatTensor | BFloat16Tensor,
        number_mask: BoolTensor,
        numbers: DoubleTensor,
        hidden_states_slice=slice(0,-1),
        **kwargs
    ) -> FloatTensor:
        """
        Compute the L1 loss for the number prediction.
        Args:
            out (CausalLMOutputWithCrossAttentions): The output of the model containing the hidden states
            num_encodings (FloatTensor): The number encodings of shape (B, S, E_f)
            number_mask (BoolTensor):  Boolean mask of num_tokens in output tokens (B, S)
            numbers (DoubleTensor): The numerical values of the number tokens of shape (B, S)
            hidden_states_slice (slice): The slice to select the target sequence of the hidden states
        Returns:
            num_loss_per_sample (FloatTensor): The L1 loss per sample of shape (B,S)
        """
        pass