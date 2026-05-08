import argparse
import os
from math import inf, nan

import numpy as np
import tiktoken
from transformers import PreTrainedTokenizerFast

from tokenizers import Tokenizer, decoders, models

current_file = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file)

parser = argparse.ArgumentParser(description="Choose tokenizer type.")
parser.add_argument(
    "--tokenizer_type",
    type=str,
    choices=["sd", "fe", "td", "rm"],
    required=True,
    help="Specify the tokenizer type: sd, fe, td, or rm."
)
args = parser.parse_args()

enc = tiktoken.get_encoding("gpt2")
orig_vocab = enc._mergeable_ranks
float_toks = []
for t in orig_vocab:
    try:
        f = float(t)
        if not (f==nan or f==inf or f==-inf): 
            float_toks.append(t)
    except ValueError:
        continue
new_vocab = orig_vocab.copy()
for t in float_toks:
    new_vocab.pop(t, None)
len(new_vocab)
sd = {str(i).encode() for i in range(10)}
td = {str(i).encode() for i in range(1000)}
bos_token = "<|bos|>"
eos_token = "<|eos|>"
pad_token = "<|pad|>"
num_token = "<|num|>"
overflow_token = "<|overflow|>"
eoq_token = "<|eoq|>"
min_token = "<|min|>"
max_token = "<|max|>"
asc_token = "<|asc|>"
desc_token = "<|desc|>"
interval_assignment_token = "<|intva|>"
mean_token = "<|mean|>"
std_token = "<|std|>"
repeat_token = "<|repeat|>"
char_token = "<|char|>"

special_tokens = [bos_token, eos_token, pad_token, eoq_token, min_token, max_token, asc_token, desc_token, interval_assignment_token, repeat_token, mean_token, std_token, char_token]

match args.tokenizer_type:
    case "sd":
        tok = Tokenizer(models.BPE())
        tok.add_tokens([*special_tokens, *map(lambda x: x.decode(),sd), *map(lambda x: x.decode("ISO-8859-1"),new_vocab.keys())])
        tok.decoder = decoders.ByteLevel()
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tok,
            clean_up_tokenization_spaces=True, # To avoid warning
            # Add pad token
            bos_token=bos_token,
            bos_token_id=tok.token_to_id(bos_token),
            eos_token=eos_token,
            eos_token_id=tok.token_to_id(eos_token),
            pad_token=pad_token,
            pad_token_id=tok.token_to_id(pad_token),
            eoq_token=eoq_token,
            eoq_token_id=tok.token_to_id(eoq_token),
            min_token=min_token,
            min_token_id=tok.token_to_id(min_token),
            max_token=max_token,
            max_token_id=tok.token_to_id(max_token),
            asc_token=asc_token,
            asc_token_id=tok.token_to_id(asc_token),
            desc_token=desc_token,
            desc_token_id=tok.token_to_id(desc_token),
            interval_assignment_token=interval_assignment_token,
            interval_assignment_token_id=tok.token_to_id(interval_assignment_token),
            repeat_token=repeat_token,
            repeat_token_id=tok.token_to_id(repeat_token),
            mean_token=mean_token,
            mean_token_id=tok.token_to_id(mean_token),
            std_token=std_token,
            std_token_id=tok.token_to_id(std_token),
            char_token=char_token,
            char_token_id=tok.token_to_id(char_token),
        )
        wrapped_tokenizer.add_special_tokens({
            "bos_token": bos_token,
            "eos_token": eos_token,
            "pad_token": pad_token,
            "additional_special_tokens": special_tokens # pyright: ignore[reportArgumentType]
        })
    case "fe":
        tok = Tokenizer(models.BPE())
        # tok.add_tokens([*special_tokens, *map(lambda x: x.decode(),sd), *map(lambda x: x.decode("ISO-8859-1"),new_vocab.keys())])
        tok.add_tokens([*special_tokens, num_token, overflow_token, *map(lambda x: x.decode("ISO-8859-1"),new_vocab.keys())])
        tok.decoder = decoders.ByteLevel()
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tok,
            clean_up_tokenization_spaces=True, # To avoid warning
            # Add pad token
            bos_token=bos_token,
            bos_token_id=tok.token_to_id(bos_token),
            eos_token=eos_token,
            eos_token_id=tok.token_to_id(eos_token),
            pad_token=pad_token,
            pad_token_id=tok.token_to_id(pad_token),
            num_token=num_token,
            num_token_id=tok.token_to_id(num_token),
            overflow_token=overflow_token,
            overflow_token_id=tok.token_to_id(overflow_token),
            eoq_token=eoq_token,
            eoq_token_id=tok.token_to_id(eoq_token),
            min_token=min_token,
            min_token_id=tok.token_to_id(min_token),
            max_token=max_token,
            max_token_id=tok.token_to_id(max_token),
            asc_token=asc_token,
            asc_token_id=tok.token_to_id(asc_token),
            desc_token=desc_token,
            desc_token_id=tok.token_to_id(desc_token),
            interval_assignment_token=interval_assignment_token,
            interval_assignment_token_id=tok.token_to_id(interval_assignment_token),
            repeat_token=repeat_token,
            repeat_token_id=tok.token_to_id(repeat_token),
            mean_token=mean_token,
            mean_token_id=tok.token_to_id(mean_token),
            std_token=std_token,
            std_token_id=tok.token_to_id(std_token),
            char_token=char_token,
            char_token_id=tok.token_to_id(char_token),
        )
        wrapped_tokenizer.add_special_tokens({
            "bos_token": bos_token,
            "eos_token": eos_token,
            "pad_token": pad_token,
            "additional_special_tokens": [*special_tokens, num_token, overflow_token]  # pyright: ignore[reportArgumentType]
        })
    case "td":
        tok = Tokenizer(models.BPE())
        tok.add_tokens([*special_tokens, *map(lambda x: x.decode(),td), *map(lambda x: x.decode("ISO-8859-1"),new_vocab.keys())])
        tok.decoder = decoders.ByteLevel()
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tok,
            clean_up_tokenization_spaces=True, # To avoid warning
            # Add pad token
            bos_token=bos_token,
            bos_token_id=tok.token_to_id(bos_token),
            eos_token=eos_token,
            eos_token_id=tok.token_to_id(eos_token),
            pad_token=pad_token,
            pad_token_id=tok.token_to_id(pad_token),
            eoq_token=eoq_token,
            eoq_token_id=tok.token_to_id(eoq_token),
            min_token=min_token,
            min_token_id=tok.token_to_id(min_token),
            max_token=max_token,
            max_token_id=tok.token_to_id(max_token),
            asc_token=asc_token,
            asc_token_id=tok.token_to_id(asc_token),
            desc_token=desc_token,
            desc_token_id=tok.token_to_id(desc_token),
            interval_assignment_token=interval_assignment_token,
            interval_assignment_token_id=tok.token_to_id(interval_assignment_token),
            repeat_token=repeat_token,
            repeat_token_id=tok.token_to_id(repeat_token),
            mean_token=mean_token,
            mean_token_id=tok.token_to_id(mean_token),
            std_token=std_token,
            std_token_id=tok.token_to_id(std_token),
            char_token=char_token,
            char_token_id=tok.token_to_id(char_token),
        )
        wrapped_tokenizer.add_special_tokens({
            "bos_token": bos_token,
            "eos_token": eos_token,
            "pad_token": pad_token,
            "additional_special_tokens": special_tokens # pyright: ignore[reportArgumentType]
        })
        wrapped_tokenizer.padding_side = "left"
        wrapped_tokenizer.save_pretrained(f"{current_folder}/tokenizers/num_text/td_gpt2")
    case "rm":
        tok = Tokenizer(models.BPE())
        num_tokens = [f"<|num_{i}|>" for i in range(256)]
        tok.add_tokens([*special_tokens, *num_tokens, *map(lambda x: x.decode("ISO-8859-1"),new_vocab.keys())])
        tok.decoder = decoders.ByteLevel()

        num_token_bins = (2**np.linspace(-33, 34, 128, endpoint=True)).tolist()
        num_token_bins = [-inf, *[-i for i in reversed(num_token_bins[:-1])],*num_token_bins, inf]

        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tok,
            clean_up_tokenization_spaces=True, # To avoid warning
            # Add pad token
            bos_token=bos_token,
            bos_token_id=tok.token_to_id(bos_token),
            eos_token=eos_token,
            eos_token_id=tok.token_to_id(eos_token),
            pad_token=pad_token,
            pad_token_id=tok.token_to_id(pad_token),
            num_token=num_tokens,
            num_token_bins = num_token_bins,
            eoq_token=eoq_token,
            eoq_token_id=tok.token_to_id(eoq_token),
            min_token=min_token,
            min_token_id=tok.token_to_id(min_token),
            max_token=max_token,
            max_token_id=tok.token_to_id(max_token),
            asc_token=asc_token,
            asc_token_id=tok.token_to_id(asc_token),
            desc_token=desc_token,
            desc_token_id=tok.token_to_id(desc_token),
            interval_assignment_token=interval_assignment_token,
            interval_assignment_token_id=tok.token_to_id(interval_assignment_token),
            repeat_token=repeat_token,
            repeat_token_id=tok.token_to_id(repeat_token),
            mean_token=mean_token,
            mean_token_id=tok.token_to_id(mean_token),
            std_token=std_token,
            std_token_id=tok.token_to_id(std_token),
            char_token=char_token,
            char_token_id=tok.token_to_id(char_token),
        )
        wrapped_tokenizer.add_special_tokens({
            "bos_token": bos_token,
            "eos_token": eos_token,
            "pad_token": pad_token,
            "additional_special_tokens": [*special_tokens, *num_tokens] # pyright: ignore[reportArgumentType]
        })

wrapped_tokenizer.padding_side = "left"
wrapped_tokenizer.save_pretrained(f"{current_folder}/tokenizers/num_text/{args.tokenizer_type}_gpt2")
