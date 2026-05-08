from os import PathLike
from typing import Any, Dict, Optional, Type, Union, cast, override

import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils.generic import ModelOutput

from networks.number_embedding_modules.abc_embedding import ABCEmbedding
from networks.number_embedding_modules.base10_embedding import Base10Embedding
from networks.number_embedding_modules.bittoken_embedding import (
    BitTokenEmbedding,
)
from networks.number_embedding_modules.fone_embedding import FoNE
from networks.number_embedding_modules.xVal_embedding import xValEmbedding
from utils.dynamic_compile import dynamic_compile
from utils.enums import (
    CausalLMOutputWithCrossAttentionsAndNumbers,
    GenerateOutput,
    GenerateOutputWithNumbers,
)
from utils.train_argument_parser import BaseArgumentParser
from utils.util_funcs import get_num_token_mask


def create_stem_head_model(superclass: Type[GPT2LMHeadModel], args: BaseArgumentParser, create_from_pretrained: bool=False, **kwargs) -> GPT2LMHeadModel:
    """
    Create a `StemHeadModel` from the given superclass of type `GPT2LMHeadModel`.

    Args:
        superclass (Type[GPT2LMHeadModel]): The superclass to extend
        create_from_pretrained (bool, optional): Whether to create the model from pretrained. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the model
    Returns:
        GPT2LMHeadModel: The StemHeadModel
    """
    class StemHeadModel(superclass):
        """
        `StemHeadModel` extends the `GPT2LMHeadModel` model with Single Token EMbeddings for
        numbers in a more structured way.
        The learned embedding of the number placeholder tokens with the id tokenizer.num_token_id 
        are combined with the encoding of the numbers.

        Note that the `.forward` and the `.generate` methods are overridden and require the `numbers`
        to be passed as an additional argument.
        """

        def __init__(
                self,
                config: GPT2Config,
                tokenizer: PreTrainedTokenizerFast,
                args: BaseArgumentParser
            ):
            """
            `StemHeadModel` extends the `GPT2LMHeadModel` model with with Single Token EMbeddings for
            numbers in a more structured way.
            Args:
                config: The configuration of the GPT2 model
                tokenizer (PreTrainedTokenizerFast): The tokenizer used for encoding the input
                args (BaseArgumentParser): The training arguments
            """
            super().__init__(config)
            self.initialize(
                config,
                tokenizer,
                args
            )

        def initialize(
                self,
                config: GPT2Config,
                tokenizer: PreTrainedTokenizerFast,
                args: BaseArgumentParser,
            ):
            """
            Initialize the model with the given configuration, tokenizer and training arguments.
            Args:
                config (GPT2Config): The configuration of the GPT2 model
                tokenizer (PreTrainedTokenizerFast): The tokenizer used for encoding the input
                args (BaseArgumentParser): The training arguments
            """
            self.tokenizer = tokenizer
            assert tokenizer.init_kwargs["model_specific_special_tokens"]["num_token"] is not None, "Tokenizer must have a num_token"
            num_tokens = tokenizer.init_kwargs["model_specific_special_tokens"]["num_token"]
            if not isinstance(num_tokens, list):
                num_tokens = [num_tokens]
            self._num_token_ids = tuple(cast(int, tokenizer.convert_tokens_to_ids(token)) for token in num_tokens)
            self.num_embedding_type = args.num_embedding_type
            self.num_embedding_module: ABCEmbedding
            device_str = str(args.device).split(":")[0]
            match args.num_embedding_type:
                case "bittoken":
                    self.num_embedding_module = BitTokenEmbedding(
                        n_embed=config.n_embd,
                        device=device_str,
                        add_reciprocal=args.add_reciprocal,
                        combination_method=args.combine_strategy,
                        loss_type=args.num_loss_type,
                        frequency_weight_slope=args.frequency_weight_slope if hasattr(args, "frequency_weight_slope") else 0.,
                        number_head_type=args.num_head_type,
                        precision_type=cast(torch.dtype, args.precision_type)
                    )
                case "base10":
                    self.num_embedding_module = Base10Embedding(
                        n_embed=config.n_embd,
                        device=device_str,
                        add_reciprocal=args.add_reciprocal,
                        combination_method=args.combine_strategy,
                        loss_type=args.num_loss_type
                    )
                case "fone":
                    self.num_embedding_module = FoNE(
                        n_embd=config.n_embd,
                        add_linear=False,
                        period_base_list=[args.base]
                    )
                case "xval":
                    self.num_embedding_module = xValEmbedding(n_embd=config.n_embd)
                case _:
                    raise NotImplementedError(f"Loss type {args.num_embedding_type} is not implemented")
            self.latest_numbers: torch.DoubleTensor=None
            self.super_forward = dynamic_compile(super().forward, disable=not args.compile)
            self.n_embed = config.n_embd

        @dynamic_compile(disable=not args.compile)
        def _get_input_embeds(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
            """
            Generate the input embeddings for the model. The input embeddings are generated by
            combining the token embeddings with the number encoding of the numbers.
            Args:
                input_ids (torch.LongTensor): The input token ids of shape (B, S)
            Returns:
                input_embeds (torch.FloatTensor): The input embeddings of shape (B, S, E_t)
            """
            return self.transformer.wte(input_ids).to(dtype=torch.get_autocast_gpu_dtype())

        @property
        def num_token_ids(self) -> torch.LongTensor:
            return torch.tensor(self._num_token_ids, dtype=torch.long)
        
        def get_padded_num_encoding(self, input_ids: torch.LongTensor, numbers: torch.DoubleTensor, number_mask: torch.BoolTensor) -> torch.FloatTensor:
            number_indices = number_mask.nonzero(as_tuple=True)
            num_encoding : torch.FloatTensor = self.num_embedding_module.forward(numbers[number_mask]) # pyright: ignore[reportArgumentType]
            if self.num_embedding_type == "xval":
                padded_num_encoding = torch.ones([*number_mask.shape, num_encoding.shape[-1]], dtype=num_encoding.dtype, device=input_ids.device)
            else:
                padded_num_encoding = torch.zeros([*number_mask.shape, *num_encoding.shape[1:]], dtype=num_encoding.dtype, device=input_ids.device)
            padded_num_encoding[number_indices[0], number_indices[1]] = num_encoding
            return padded_num_encoding
        
        def compute_input_embeddings(self, input_ids: torch.LongTensor, numbers: torch.DoubleTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
            """
            Compute the input embeddings by combining the token embeddings with the number encoding of the numbers.
            Args:
                input_ids (torch.LongTensor): The input token ids of shape (B, S)
                numbers (torch.DoubleTensor): The numerical values of the number tokens of shape (B, S)
            Returns:
               combined_embeds (torch.FloatTensor): The input embeddings of shape (B, S, E_t)
               padded_num_encoding (torch.FloatTensor): The number encodings of shape (B, S, E_f)
            """
            inputs_embeds = self._get_input_embeds(input_ids)
            num_token_ids: torch.LongTensor = torch.tensor(self._num_token_ids, dtype=torch.long, device=input_ids.device)
            number_mask: torch.BoolTensor = get_num_token_mask(input_ids, num_token_ids)
            padded_num_encoding = self.get_padded_num_encoding(input_ids, numbers, number_mask)
            combined_embeds = self.num_embedding_module.combine_embeds(inputs_embeds, padded_num_encoding, number_mask)
            return combined_embeds, padded_num_encoding

        @override
        def forward(
            self,
            input_ids: torch.LongTensor,
            numbers: torch.DoubleTensor,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            **kwargs,
        ) -> CausalLMOutputWithCrossAttentionsAndNumbers:
            combined_embeds, padded_num_encoding = self.compute_input_embeddings(input_ids, numbers)
            out: CausalLMOutputWithCrossAttentions = self.super_forward(
                inputs_embeds=combined_embeds, # pyright: ignore[reportCallIssue]
                labels=labels, # pyright: ignore[reportCallIssue]
                **{**kwargs, "output_hidden_states": True},
            )

            if labels is not None:
                label_num_token_ids: torch.LongTensor = torch.tensor(self._num_token_ids, dtype=torch.long, device=labels.device)
                label_number_mask: torch.BoolTensor = get_num_token_mask(labels[..., 1:].contiguous(), label_num_token_ids) # pyright: ignore[reportArgumentType]
                if label_number_mask.any():
                    train_metrics = self.num_embedding_module.compute_num_loss(out, padded_num_encoding[:, 1:], label_number_mask, numbers[..., 1:]) # pyright: ignore[reportArgumentType]
                    if isinstance(train_metrics, dict):
                        setattr(out, "num_loss", train_metrics.pop("num_loss"))
                        setattr(out, "additional_train_losses", train_metrics)
                    else: # legacy support
                        setattr(out, "num_loss", train_metrics)
            else:
                assert out.logits is not None
                pred_tokens = out.logits.argmax(-1)[...,-1:]
                pred_num_token_ids: torch.LongTensor = torch.tensor(self._num_token_ids, dtype=torch.long, device=pred_tokens.device)
                number_mask: torch.BoolTensor = get_num_token_mask(pred_tokens, pred_num_token_ids) # pyright: ignore[reportArgumentType]
                if number_mask.any():
                    num_decoded = self.num_embedding_module.decode(out, number_mask)
                    new_numbers = torch.zeros((input_ids.shape[0],1), dtype=numbers.dtype, device=numbers.device)
                    new_numbers[number_mask] = num_decoded
                else:
                    new_numbers = torch.zeros((self.latest_numbers.shape[0],1), dtype=self.latest_numbers.dtype, device=self.latest_numbers.device)
                self.latest_numbers = torch.cat((self.latest_numbers, new_numbers), dim=-1)
                setattr(out, "numbers", new_numbers)
            return out
        
        @override
        def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            next_sequence_length: int | None = None,
            past_key_values=None,
            attention_mask: torch.LongTensor | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            cache_position: torch.LongTensor | None = None,
            is_first_iteration: bool | None = False,
            numbers: torch.DoubleTensor | None = None,
            position_ids: torch.LongTensor | None = None,
            max_seq_length: int | None = None,
            **kwargs,
        ) -> dict[str, Any]:
            # Transformers 5.x will slice `input_ids` after prefill (typically to last token). We must slice `numbers`
            # the same way so that the per-token number embeddings stay aligned.
            if numbers is not None and next_sequence_length is not None:
                numbers = numbers[:, -next_sequence_length:]

            model_inputs = super().prepare_inputs_for_generation(
                input_ids=input_ids,
                next_sequence_length=next_sequence_length,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                is_first_iteration=is_first_iteration,
                position_ids=position_ids,
                max_seq_length=max_seq_length,
                **kwargs,
            )
            if is_first_iteration:
                self.latest_numbers = numbers
            model_inputs["numbers"] = numbers
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
            model_kwargs["numbers"] = getattr(outputs, "numbers", model_kwargs.get("numbers"))
            return model_kwargs
        
        @override
        def _sample(
            self,
            *args,
            **model_kwargs,
        ) -> GenerateOutputWithNumbers:
            out: GenerateOutput = super()._sample(*args, **model_kwargs)
            setattr(out, "numbers", self.latest_numbers)
            return out

    @override
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, PathLike],
        tokenizer: PreTrainedTokenizerFast,
        *init_inputs,
        args: BaseArgumentParser,
        cache_dir: Optional[Union[str, PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        trust_remote_code=False,
        **kwargs,
    ) -> StemHeadModel:
        model: StemHeadModel = superclass.from_pretrained(
            pretrained_model_name_or_path,
            *init_inputs,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        model.__class__ =  StemHeadModel
        model_config: GPT2Config = model.config
        model.initialize(
                model_config,
                tokenizer,
                args=args
            )
        return model
    if create_from_pretrained:
        return StemHeadModel.from_pretrained(
            args=args,
            **kwargs
        )
    return StemHeadModel(
        args=args,
        **kwargs
    )
