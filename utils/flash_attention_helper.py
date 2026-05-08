import inspect
import os
from typing import Optional, Tuple

import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

_flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
deterministic_g = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"

def _upad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    max_seqlen_in_batch_k: int,
):
    """
    > NOTE: We use this function as a temporary replacement instead of the original `transformers.modeling_flash_attention_utils._upad_input` to avoid a compile graph break.

    Unpads query, key, and values tensors, using a single dimension for all tokens even though they belong to different batches.

    This function is used instead of `flash_attn.bert_padding.unpad_input` in order to avoid the recomputation of the same intermediary
    tensors for query, key, value tensors.

    Arguments:
        query_layer (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
        query_length (`int`):
            Target length.
        max_seqlen_in_batch_k (`int`):
            Maximum sequence length in batch (`max_seqlen_in_batch_k` for the source sequence i.e. key/value).

    Return:
        query_layer (`torch.Tensor`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`Tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`Tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    if value_layer.shape != key_layer.shape:
        raise AssertionError(
            "_upad_input expects key_layer and value_layer to have identical shapes. "
            f"(key_layer={tuple(key_layer.shape)}, value_layer={tuple(value_layer.shape)})."
        )
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices_k = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, -1, head_dim), indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        # flash_attn.bert_padding.unpad_input returns 5 values:
        #   (unpadded_hidden_states, indices, cu_seqlens, max_seqlen_in_batch, unused_mask_or_last)
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q, _ = unpad_input(
            query_layer, attention_mask
        )

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )

def compilable_flash_attention_2_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    cu_seq_lens: Optional[torch.Tensor],
    max_seq_length: int,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """
    Wrapper function for `flash_attn.flash_attn_varlen_func` that is a temporary replacement for `transformers.integrations.flash_attention.flash_attention_forward`.
    This function supports compilation and sequence packing.
    Args:
        module (torch.nn.Module): transformers AttentionModule
        query (torch.Tensor): 
    """
    batch_size = query.size(0)
    use_sliding_windows = (
        _flash_supports_window_size and sliding_window is not None and key.shape[2] > sliding_window
    )
    query_length = query.shape[2]
    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    flash_kwargs = {"window_size": (sliding_window, 0)} if use_sliding_windows else {}
    flash_kwargs["deterministic"] = deterministic_g # pyright: ignore[reportArgumentType]
    if softcap is not None:
        flash_kwargs["softcap"] = softcap # pyright: ignore[reportArgumentType]

    if cu_seq_lens is None and attention_mask is not None:
        query, key, value, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
            query, key, value, attention_mask, query_length, max_seqlen_in_batch_k=max_seq_length
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens # pyright: ignore[reportGeneralTypeIssues]
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=True,
            **flash_kwargs, # pyright: ignore[reportArgumentType]
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    elif cu_seq_lens is not None:
        query = query.contiguous().view(-1, query.size(-2), query.size(-1))
        key = key.contiguous().view(-1, key.size(-2), key.size(-1))
        value = value.contiguous().view(-1, value.size(-2), value.size(-1))

        attn_output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seq_lens,
            cu_seqlens_k=cu_seq_lens,
            max_seqlen_q=max_seq_length,
            max_seqlen_k=max_seq_length,
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=True,
            **flash_kwargs # pyright: ignore[reportArgumentType]
        )
        attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1)) # pyright: ignore[reportOptionalMemberAccess]
    else:
        attn_output = flash_attn_func(
            query, key, value, dropout, softmax_scale=scaling, causal=True, **flash_kwargs # pyright: ignore[reportArgumentType]
        )
    return attn_output, None
