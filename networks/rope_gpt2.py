# This is a modified version of the GPT2 model from the Hugging Face Transformers library.
# For the original code, see:
# https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/models/gpt2/modeling_gpt2.py

from typing import Any, Callable, Dict, Optional, Tuple, Union, cast, override

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2Model
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.masking_utils import create_bidirectional_mask, create_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2MLP,
    GPT2Attention,
    GPT2Block,
    GPT2LMHeadModel,
    eager_attention_forward,
    logging,
)
from transformers.utils.generic import ModelOutput

from utils.flash_attention_helper import compilable_flash_attention_2_forward

logger = logging.get_logger(__name__)
try:
    # Transformers >=5 provides a registry-like interface.
    ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", compilable_flash_attention_2_forward)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
except Exception:
    # Fallback for older interface shapes.
    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = compilable_flash_attention_2_forward  # pyright: ignore[reportArgumentType]

class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) as introduced in "RoFormer: Enhanced Transformer with Rotary Position Embedding" by Su et al.
    https://arxiv.org/abs/2104.09864.
    """
    def __init__(self, dim: int, max_position_embeddings=2048, base=10000):
        """
        Rotary Positional Embedding (RoPE). The class computes the cosine and sine embeddings for the positional encodings and caches them.
        Args:
            dim (int): The dimension of the input tensor.
            max_position_embeddings (int): The maximum position embeddings to cache.
            base (int): The base of the sinusoidal function.
            device (torch.device): The device to use.
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.inv_freq: torch.FloatTensor
        self._initialize(max_position_embeddings)

    def _initialize(self, max_seq_len: int):
        self.max_seq_len_cached = max_seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.cos_cached: torch.FloatTensor
        self.register_buffer("sin_cached", freqs.sin())
        self.sin_cached: torch.FloatTensor

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor | None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): The input tensor.
            seq_len (int): The sequence length.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The cosine and sine embeddings.
        """
        assert position_ids is not None
        return (
            self.cos_cached[position_ids, ...].unsqueeze(1),
            self.sin_cached[position_ids, ...].unsqueeze(1),
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the tensor by 90 degrees.
    Args:
        x (torch.Tensor): The input tensor.
    Returns:
        torch.Tensor: The rotated tensor.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotary positional embeddings to the tensor.
    Args:
        x (torch.Tensor): The input tensor.
        cos (torch.Tensor): The cosine embeddings.
        sin (torch.Tensor): The sine embeddings.
    Returns:
        torch.Tensor: The tensor with the positional embeddings applied.
    """
    x1, x2 = x.chunk(2, dim=-1)
    # Apply rotation: [cos -sin; sin cos] @ [x1; x2]
    
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class RopeGPT2Attention(GPT2Attention):
    @override
    def forward(
        self,
        hidden_states: torch.FloatTensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        cos: Optional[torch.FloatTensor]=None,
        sin: Optional[torch.FloatTensor]=None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        assert self.layer_idx is not None
        is_cross_attention = encoder_hidden_states is not None

        curr_past_key_values: Cache | None = None
        is_updated: bool | None = None
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                curr_past_key_values = (
                    past_key_values.cross_attention_cache if is_cross_attention else past_key_values.self_attention_cache
                )
            else:
                curr_past_key_values = past_key_values

        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query_states = self.q_attn(hidden_states)
            attention_mask = encoder_attention_mask

            # Try to get key/value states from cache if possible.
            if curr_past_key_values is not None and is_updated:
                key_states = curr_past_key_values.layers[self.layer_idx].keys
                value_states = curr_past_key_values.layers[self.layer_idx].values
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        ##############################################
        # QK-norm
        norm_func = torch.nn.functional.rms_norm if self.config.norm_class == "rms" else torch.nn.functional.layer_norm
        query_states = norm_func(query_states, (query_states.size(-1),)).to(query_states.dtype)
        key_states = norm_func(key_states, (key_states.size(-1),)).to(query_states.dtype)
        ##############################################

        ##############################################
        # RoPE
        if cos is not None and sin is not None:
            query_states = apply_rotary_pos_emb(query_states, cos, sin)
            key_states = apply_rotary_pos_emb(key_states, cos, sin)
        ##############################################

        if curr_past_key_values is not None and (
            (not is_cross_attention) or (is_cross_attention and not is_updated)
        ):
            # Save all key/value to cache for fast auto-regressive generation.
            cache_position_for_update = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_values.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position_for_update}
            )
            if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                past_key_values.is_updated[self.layer_idx] = True

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                **kwargs,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights
    
class RopeGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        
        norm_class = nn.RMSNorm if config.norm_class=="rms" else nn.LayerNorm
        self.ln_1 = norm_class(hidden_size)
        self.attn: RopeGPT2Attention = RopeGPT2Attention(config=config, layer_idx=layer_idx)
        self.ln_2 = norm_class(hidden_size)

        if config.add_cross_attention:
            self.crossattention: RopeGPT2Attention = RopeGPT2Attention(config=config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = norm_class(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    @override
    def forward(
        self,
        hidden_states: torch.FloatTensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cos: Optional[torch.FloatTensor]=None,
        sin: Optional[torch.FloatTensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        cu_seq_lens: Optional[torch.Tensor] = None,
        max_seq_length: Optional[int] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, _ = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            use_cache=use_cache,
            sin=sin,
            cos=cos,
            position_ids=position_ids,
            cu_seq_lens=cu_seq_lens,
            max_seq_length=max_seq_length,
        )
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_output, _ = self.crossattention(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                sin=sin,
                cos=cos,
                position_ids=position_ids,
                cu_seq_lens=cu_seq_lens,
                max_seq_length=max_seq_length,
            )
            # residual connection
            hidden_states = residual + cross_attn_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states

class RopeGPT2Model(GPT2Model):
    """
    A GPT2 model that uses the Rotary Positional Embedding (RoPE) instead of the standard sinusoidal positional embeddings.
    """
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        del self.h
        self.h = nn.ModuleList([RopeGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings
        )
        del self.wpe
        self.wpe = lambda x: 0  # Dummy positional embeddings
        norm_class = nn.RMSNorm if config.norm_class=="rms" else nn.LayerNorm
        self.ln_f = norm_class(config.hidden_size)
        self.post_init()

    @override
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Cache | None = None,  # pyright: ignore[reportRedeclaration]
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None, # pyright: ignore[reportRedeclaration]
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None, # pyright: ignore[reportRedeclaration]
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cu_seq_lens: Optional[torch.Tensor] = None,
        max_seq_length: Optional[int] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0] # pyright: ignore[reportOptionalMemberAccess]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")


        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

         # based on pattern from src/transformers/models/whisper/modeling_whisper.py::WhisperDecoder
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)

            if self.config.add_cross_attention and not isinstance(past_key_values, EncoderDecoderCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache(config=self.config))

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        assert inputs_embeds is not None

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        assert cache_position is not None
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        assert position_ids is not None

        position_embeds = self.wpe(position_ids)
        if isinstance(position_embeds, torch.Tensor):
            hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        else:
            hidden_states = inputs_embeds


        ######################################
        # RoPE
        cos, sin = self.rotary_emb(hidden_states, position_ids=position_ids)
        ######################################

        # Attention mask.
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        if self._attn_implementation == "flash_attention_2":
            # Flash-attn path: we forward the 2D attention mask to the attention backend (or None).
            causal_mask = attention_mask
            encoder_attention_mask = encoder_attention_mask
        else:
            assert inputs_embeds is not None
            assert cache_position is not None
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

            encoder_attention_mask = None
            if encoder_hidden_states is not None:
                assert inputs_embeds is not None
                encoder_attention_mask = create_bidirectional_mask(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                )
        
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False


        all_hidden_states = () if output_hidden_states else None

        for i, block in enumerate(self.h):
            if output_hidden_states and all_hidden_states is not None:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = block(
                hidden_states,
                past_key_values if not (self.gradient_checkpointing and self.training) else None,
                cache_position,
                causal_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                position_ids=position_ids,
                sin=sin,
                cos=cos,
                cu_seq_lens=cu_seq_lens,
                max_seq_length=max_seq_length,
            )

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states and all_hidden_states is not None:
            all_hidden_states = all_hidden_states + (hidden_states,)

        past_key_values = past_key_values if use_cache else None
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=None,
            cross_attentions=None,
        )

class RopeGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.transformer = RopeGPT2Model(config)
        self.post_init()
        self.loss_fct = CrossEntropyLoss(reduction="none")

    @override
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        next_sequence_length: int | None = None,
        past_key_values: Cache | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        is_first_iteration: bool | None = False,
        position_ids: torch.LongTensor | None = None,
        cu_seq_lens: torch.Tensor | None = None,
        max_seq_length: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            next_sequence_length=next_sequence_length,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            is_first_iteration=is_first_iteration,
            position_ids=position_ids,
            **kwargs,
        )
        model_inputs["cu_seq_lens"] = cu_seq_lens
        model_inputs["max_seq_length"] = max_seq_length
        return model_inputs

    @override
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens
        )
        attention_mask: torch.Tensor | None = model_kwargs.get("attention_mask")
        if attention_mask is not None and attention_mask.ndim == 2:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            model_kwargs["position_ids"] = position_ids
            model_kwargs["max_seq_length"] = position_ids.max().item() + 1
        else:
            cache_position = model_kwargs.get("cache_position")
            if cache_position is not None:
                # `cache_position` is 1D; expand to batch for `position_ids` if needed.
                position_ids = cache_position.unsqueeze(0)
                if attention_mask is not None and attention_mask.ndim >= 2:
                    batch_size = attention_mask.shape[0]
                    position_ids = position_ids.expand(batch_size, -1)
                model_kwargs["position_ids"] = position_ids
                model_kwargs["max_seq_length"] = int(cache_position.max().item()) + 1
        return model_kwargs

    @override
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Cache | None = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cu_seq_lens: Optional[torch.Tensor] = None,
        max_seq_length: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cu_seq_lens=cu_seq_lens,
            max_seq_length=max_seq_length,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism (legacy HF API)
        if getattr(self, "model_parallel", False):
            first_device = getattr(self.transformer, "first_device", None)
            if first_device is not None:
                torch.cuda.set_device(cast(torch.device, first_device))
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous() # pyright: ignore[reportOptionalSubscript]
            # Flatten the tokens
            loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        out = CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
        return out
