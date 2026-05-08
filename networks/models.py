import logging
from pathlib import Path
from typing import Optional, Type

import torch
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

from networks.modded_nanoGPT import CastedLinear, NanoGPTLMHeadModel
from networks.rope_gpt2 import RopeGPT2LMHeadModel
from networks.stem_head_model import create_stem_head_model
from utils.train_argument_parser import BaseArgumentParser


def _default_gpt2_config(tokenizer: PreTrainedTokenizerFast, args: BaseArgumentParser, **kwargs) -> GPT2Config:
    if not isinstance(tokenizer.bos_token_id, int) or not isinstance(tokenizer.eos_token_id, int) or not isinstance(tokenizer.pad_token_id, int):
        raise ValueError("Tokenizer must have valid bos_token_id, eos_token_id, and pad_token_id.")
    config = GPT2Config(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        n_embd = args.n_embd,
        n_head = args.n_head,
        n_layer = args.n_layer,
        vocab_size=len(tokenizer),
        max_position_embeddings=args.context_length,
        tie_word_embeddings=args.tie_word_embeddings, # Untie embedding layer from the output layer
        _attn_implementation=args.attn_implementation,
        **kwargs
    )
    setattr(config, "norm_class", args.norm)
    return config

def getGpt2Model(
        tokenizer: PreTrainedTokenizerFast,
        args: BaseArgumentParser,
        pretrained_model_dir: Optional[str|Path]=None,
        device=torch.device("cpu")
    ) -> GPT2LMHeadModel:
    """
    Get the default GPT2 model used for all experiments
    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use
        pretrained_model_dir (str, optional): The directory of the pretrained model. Defaults to None.
        device (torch.device, optional): The device to use. Defaults to torch.device("cpu").
    Returns:
        GPT2LMHeadModel: The model
    """
    if pretrained_model_dir is not None:
        return GPT2LMHeadModel.from_pretrained(pretrained_model_dir).to(device=device) # pyright: ignore[reportCallIssue]
    config = _default_gpt2_config(tokenizer, args)
    return AutoModelForCausalLM.from_config(config).to(device=device)

def getRopeGpt2Model(
        tokenizer: PreTrainedTokenizerFast,
        args: BaseArgumentParser,
        pretrained_model_dir: Optional[str|Path]=None,
        device=torch.device("cpu")
    ) -> RopeGPT2LMHeadModel:
    """
    Get the default GPT2 model used for all experiments
    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use
        pretrained_model_dir (str, optional): The directory of the pretrained model. Defaults to None.
        device (torch.device, optional): The device to use. Defaults to torch.device("cpu").
    Returns:
        RopeGPT2LMHeadModel: The model
    """
    config = _default_gpt2_config(tokenizer, args)
    if pretrained_model_dir is not None:
        return RopeGPT2LMHeadModel.from_pretrained(pretrained_model_dir, config=config).to(device=device) # pyright: ignore[reportCallIssue]
    return RopeGPT2LMHeadModel(config).to(device=device) # pyright: ignore[reportCallIssue]

def getNanoGptModel(
        tokenizer: PreTrainedTokenizerFast,
        args: BaseArgumentParser,
        pretrained_model_dir: Optional[str|Path]=None,
        device=torch.device("cpu")
    ) -> NanoGPTLMHeadModel:
    """
    Get the default GPT2 model used for all experiments
    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use
        pretrained_model_dir (str, optional): The directory of the pretrained model. Defaults to None.
        device (torch.device, optional): The device to use. Defaults to torch.device("cpu").
    Returns:
        NanoGPTLMHeadModel: The model
    """
    if pretrained_model_dir is not None:
        return NanoGPTLMHeadModel.from_pretrained(pretrained_model_dir).to(device=device) # pyright: ignore[reportCallIssue]
    config = _default_gpt2_config(tokenizer, args)
    return NanoGPTLMHeadModel(config).to(device=device) # pyright: ignore[reportCallIssue]

def getStemModel(
        superclass: Type[GPT2LMHeadModel],
        tokenizer: PreTrainedTokenizerFast,
        args: BaseArgumentParser,
        pretrained_model_dir: Optional[str|Path]=None,
        device=torch.device("cpu"),
        **kwargs
    ) -> GPT2LMHeadModel:
    """
    Get the StemHeadModel model used for all experiments
    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use
        pretrained_model_dir (str, optional): The directory of the pretrained model. Defaults to None.
        device (torch.device, optional): The device to use. Defaults to torch.device("cpu").
    Returns:
        StemHeadModel: The model
    """
    config = _default_gpt2_config(
        tokenizer,
        args,
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        summary_first_dropout=args.dropout,
    )
    if pretrained_model_dir is not None:
        return create_stem_head_model(superclass, args=args, config=config, create_from_pretrained=True, pretrained_model_name_or_path=pretrained_model_dir, tokenizer=tokenizer, **kwargs).to(device=device) # pyright: ignore[reportCallIssue]
    return create_stem_head_model(superclass, args=args, config=config, tokenizer=tokenizer, **kwargs).to(device=device) # pyright: ignore[reportCallIssue]

def get_model(
        tokenizer: PreTrainedTokenizerFast,
        args: BaseArgumentParser,
        pretrained_model_dir: Optional[str | Path]=None,
        device=torch.device("cpu"),
        **kwargs
    ) -> GPT2LMHeadModel:
    """
    Get the model based on the type

    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use
        pretrained_model_dir (str, optional): The directory of the pretrained model. Defaults to None.
        device (torch.device, optional): The device to use. Defaults to torch.device("cpu").
    Returns:
        GPT2LMHeadModel: The model
    """
    if args.model == "gpt2":
        model = getGpt2Model(tokenizer, args, pretrained_model_dir, device)
        if args.compile:
            model = torch.compile(model)
    elif args.model == "rope_gpt2":
        model = getRopeGpt2Model(tokenizer, args, pretrained_model_dir, device)
        if args.compile:
            model = torch.compile(model)
    elif args.model == "modded_nanoGPT":
        model = getNanoGptModel(tokenizer, args, pretrained_model_dir, device)
        if args.compile:
            model = torch.compile(model)
    elif args.model == "stem":
        model = getStemModel(GPT2LMHeadModel,tokenizer, args, pretrained_model_dir, device, **kwargs)
    elif args.model == "rope_stem":
        model = getStemModel(RopeGPT2LMHeadModel,tokenizer, args, pretrained_model_dir, device, **kwargs)
    elif args.model == "modded_nanoGPT_stem":
        model = getStemModel(NanoGPTLMHeadModel,tokenizer, args, pretrained_model_dir, device, **kwargs)

    else:
        raise ValueError(f"Model type {args.model} not supported")
    if pretrained_model_dir is not None:
        logging.info(f"Loaded model from {pretrained_model_dir}")
    model = model.to(dtype=args.data_type) # pyright: ignore[reportCallIssue]
    if args.model in ["modded_nanoGPT", "modded_nanoGPT_stem"]:
        for m in model.modules():
            if isinstance(m, CastedLinear):
                m.float()
    return model
