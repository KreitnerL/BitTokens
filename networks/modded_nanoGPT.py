from typing import Any, Dict, Optional, Tuple, Union, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from networks.rope_gpt2 import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Model,
    ModelOutput,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    compilable_flash_attention_2_forward,
)

# https://github.com/KellerJordan/modded-nanogpt/blob/46bf4cbe2b4ddab2f1c2fdac9e88e47e81b74872/train_gpt2.py


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_k = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_v = CastedLinear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.lamb = nn.Parameter(torch.tensor(0.5)) # @Grad62304977

    def forward(self, x, v1,
        cu_seq_lens: Optional[torch.Tensor],
        max_seq_length: int,
        attention_mask: Optional[torch.FloatTensor] = None,
        cos: Optional[torch.FloatTensor]=None,
        sin: Optional[torch.FloatTensor]=None,
        **kwargs):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        if v1 is None:
            v1 = v # This happens if we are in the first block. v needs to be accessed by subsequent blocks
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v) # @Grad62304977
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977


        ##############################################
        # RoPE
        shape_q = q.shape
        shape_kv = k.shape
        q = q.view(shape_q).transpose(1, 2)
        k = k.view(shape_kv).transpose(1, 2)
        v = v.view(shape_kv).transpose(1, 2)
        if cos is not None and sin is not None:
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)
        ##############################################

        attn_output, attn_weights = compilable_flash_attention_2_forward(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0,
            cu_seq_lens=cu_seq_lens,
            max_seq_length=max_seq_length,
            sliding_window=128,
            **kwargs,
        )
        y = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        y = self.c_proj(y)
        return y, v1



class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = CastedLinear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = CastedLinear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, v1, x0,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cos: Optional[torch.FloatTensor]=None,
        sin: Optional[torch.FloatTensor]=None,
        cu_seq_lens: Optional[torch.Tensor] = None,
        max_seq_length: int = 1024,
    ):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn.forward(
            F.rms_norm(x, (x.size(-1),)),
            v1,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            sin=sin,
            cos=cos,
            cu_seq_lens=cu_seq_lens,
            max_seq_length=max_seq_length,
        )
        # x1, v1 = self.attn(F.rms_norm(x, (x.size(-1),)), v1, block_mask)
        x = x + x1
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x, v1


class NanoGPTModel(GPT2Model):
    """
    A GPT2 model that uses the Rotary Positional Embedding (RoPE) instead of the standard sinusoidal positional embeddings.
    """
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        del self.h
        self.h = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings
        )
        del self.wpe
        self.wpe = lambda x: 0  # Dummy positional embeddings
        self.ln_f = nn.RMSNorm(config.hidden_size)
        self.post_init()
         # U-net design by @brendanh0gan
        self.num_encoder_layers = config.n_layer // 2 # Half of the layers for encoder
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

    @override
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
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

        device = input_ids.device if input_ids is not None else inputs_embeds.device # pyright: ignore[reportOptionalMemberAccess]

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0) # pyright: ignore[reportOptionalMemberAccess]

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds # pyright: ignore[reportOptionalOperand]

        ######################################
        # RoPE
        cos, sin = self.rotary_emb(hidden_states, position_ids=position_ids)
        ######################################

        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask) # pyright: ignore[reportArgumentType]
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        ######################################
        # NanoGPT
        ######################################

        # forward the GPT model itself
        hidden_states = self.ln_f(hidden_states) # @Grad62304977
        x0 = hidden_states
        v1 = None

        # Store outputs for U-Net skip connections
        skip_connections = []
        # Encoder pass - process only the first half of the blocks
        for i in range(self.num_encoder_layers):
            ######################################
            # RoPE
            block: Block = self.h[i]  # for type checker
            hidden_states ,v1 = block.forward(
                hidden_states,
                v1,
                x0,
                layer_past=past_key_values[i], # pyright: ignore[reportOptionalSubscript]
                attention_mask=attention_mask,
                head_mask=head_mask[i],  # type: ignore
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                sin=sin,
                cos=cos,
                cu_seq_lens=cu_seq_lens,
                max_seq_length=max_seq_length, # pyright: ignore[reportArgumentType]
            )
            ######################################
            skip_connections.append(hidden_states)

        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.num_decoder_layers):
            hidden_states = hidden_states + self.skip_weights[i] * skip_connections.pop()
            block: Block = self.h[self.num_encoder_layers + i]  # for type checker
            hidden_states, v1 = block.forward(
                hidden_states,
                v1,
                x0,
                layer_past=past_key_values[i], # pyright: ignore[reportOptionalSubscript]
                attention_mask=attention_mask,
                head_mask=head_mask[i], # type: ignore
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                sin=sin,
                cos=cos,
                cu_seq_lens=cu_seq_lens,
                max_seq_length=max_seq_length, # pyright: ignore[reportArgumentType]
            )

        hidden_states = self.ln_f(hidden_states)



        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,) # pyright: ignore[reportOptionalOperand]

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents, # pyright: ignore[reportArgumentType]
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class NanoGPTLMHeadModel(GPT2LMHeadModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.transformer = NanoGPTModel(config)
        self.post_init()
        self.loss_fct = CrossEntropyLoss(reduction="none")
        
        self.lm_head.weight.data.zero_() # @Grad62304977

    @override
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None, position_ids=None, cu_seq_lens=None, max_seq_length=None, **kwargs) -> dict[str, Any]:
        # This is necessary to supress a Value error because of unused generation arguments.
        return {
            **super().prepare_inputs_for_generation(
                input_ids,
                past_key_values=None, # This causes trouble when past_key_values is not None
                inputs_embeds=inputs_embeds,
                **kwargs),
            "attention_mask": attention_mask,
            "position_ids": position_ids, # GPT ereases the position_ids from the kwargs for some reason
            "cu_seq_lens": cu_seq_lens,
            "max_seq_length": max_seq_length
        }

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
        attention_mask: torch.LongTensor =model_kwargs["attention_mask"]
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        model_kwargs["position_ids"] = position_ids
        model_kwargs["max_seq_length"] = position_ids.max().item() + 1
        return model_kwargs

    @override
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
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

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(lm_logits.device)
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