"""Display helpers for the BitTokens interactive notebooks (bittokens.py, bittokens.ipynb).

IEEE float64 encode/decode live in the notebooks so the core bit conversion stays visible there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

REPO_ROOT = Path(__file__).resolve().parent.parent


TOKENIZER_DIRS: dict[str, Path] = {
    "sd_gpt2 (single-digit baseline)": REPO_ROOT / "tokenizers/num_text/sd_gpt2",
    "td_gpt2 (subword baseline)": REPO_ROOT / "tokenizers/num_text/td_gpt2",
    "bittoken_gpt2 (BitTokens)": REPO_ROOT / "tokenizers/num_text/bittoken_gpt2",
}


def load_tokenizer(tokenizer_dir: Path) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    tokenizer.padding_side = "left"
    return tokenizer


def load_all_tokenizers() -> dict[str, PreTrainedTokenizerFast]:
    return {name: load_tokenizer(path) for name, path in TOKENIZER_DIRS.items()}


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def looks_numeric_piece(token: str) -> bool:
    return bool(re.search(r"[-+]?\d|e\+|e\-|E\+|E\-", token))


def is_numeric_dot(tokens: list[str], idx: int) -> bool:
    if tokens[idx] != ".":
        return False
    prev_has_digit = idx > 0 and bool(re.search(r"\d", tokens[idx - 1]))
    next_has_digit = idx < len(tokens) - 1 and bool(re.search(r"\d", tokens[idx + 1]))
    return prev_has_digit and next_has_digit


@dataclass
class TokenizationView:
    name: str
    ids: list[int]
    tokens: list[str]
    numeric_mask: list[bool]


def get_num_token_ids_for_bittoken(tokenizer: PreTrainedTokenizerFast) -> list[int]:
    num_ids: list[int] = []

    num_token_id = tokenizer.init_kwargs.get("num_token_id")
    if isinstance(num_token_id, int):
        num_ids.append(num_token_id)

    num_tokens = tokenizer.init_kwargs.get("num_token", [])
    if isinstance(num_tokens, str):
        num_tokens = [num_tokens]
    num_ids.extend(tokenizer.convert_tokens_to_ids(t) for t in num_tokens)

    placeholder_id = tokenizer.convert_tokens_to_ids("<|num|>")
    if placeholder_id >= 0:
        num_ids.append(placeholder_id)

    return sorted(set(num_ids))


def replace_numbers_with_num_token(text: str, regex_pattern: re.Pattern, num_token: str = "<|num|>") -> str:
    return re.sub(regex_pattern, num_token, text)


def tokenize_for_view(name: str, tokenizer: PreTrainedTokenizerFast, text: str, regex_pattern: re.Pattern) -> TokenizationView:
    text_for_tokenization = text

    if "bittoken" in name:
        text_for_tokenization = replace_numbers_with_num_token(text, regex_pattern)

    encoded = tokenizer(text_for_tokenization, padding="do_not_pad", return_tensors=None)
    ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)

    if "bittoken" in name:
        num_ids = set(get_num_token_ids_for_bittoken(tokenizer))
        numeric_mask = [token_id in num_ids for token_id in ids]
    else:
        numeric_mask = [
            looks_numeric_piece(tok) or is_numeric_dot(tokens, i) for i, tok in enumerate(tokens)
        ]

    return TokenizationView(name=name, ids=ids, tokens=tokens, numeric_mask=numeric_mask)


def _highlight_token(tok: str, is_numeric: bool) -> str:
    style = (
        "background:#ffe082; color:#000; padding:2px 4px; margin:1px; border-radius:4px; display:inline-block;"
        if is_numeric
        else "background:#eceff1; color:#000; padding:2px 4px; margin:1px; border-radius:4px; display:inline-block;"
    )
    return f"<span style='{style}'>{html_escape(tok)}</span>"


def _render_token_panel(view: TokenizationView) -> str:
    token_html = "".join(_highlight_token(t, m) for t, m in zip(view.tokens, view.numeric_mask))
    total = len(view.ids)
    numeric_count = int(sum(view.numeric_mask))
    return (
        f"<h4>{html_escape(view.name)}</h4>"
        f"<div style='font-size:13px; line-height:1.8;'>{token_html}</div>"
        f"<p><b>Total tokens:</b> {total} &nbsp; | &nbsp; <b>Numeric-focused tokens:</b> {numeric_count}</p>"
    )


def tokenization_comparison_html(
    sample_text: str, tokenizers: dict[str, PreTrainedTokenizerFast], regex_pattern: re.Pattern
) -> str:
    views = [tokenize_for_view(name, tok, sample_text, regex_pattern) for name, tok in tokenizers.items()]
    panels = "".join(
        f"<div style='flex:1; min-width:280px; border:1px solid #ddd; padding:10px; border-radius:8px;'>{_render_token_panel(v)}</div>"
        for v in views
    )
    return f"<div style='display:flex; gap:12px; flex-wrap:wrap;'>{panels}</div>"


def extract_raw_numbers(text: str, regex_pattern: re.Pattern) -> list[str]:
    return [m.group(0).strip() for m in regex_pattern.finditer(text)]


def _render_ieee754_bits_row(value: float, bit_row: torch.Tensor) -> str:
    bit_chars = [str(int(b)) for b in bit_row.tolist()]

    def chip(bit: str, bg: str) -> str:
        return (
            f"<span style='background:{bg}; color:#000; padding:2px 4px; margin:1px; "
            "border-radius:4px; display:inline-block; font-family:monospace;'>"
            f"{bit}</span>"
        )

    sign_bits = "".join(chip(b, "#ffcc80") for b in bit_chars[:1])
    exponent_bits = "".join(chip(b, "#90caf9") for b in bit_chars[1:12])
    mantissa_bits = "".join(chip(b, "#c5e1a5") for b in bit_chars[12:])

    return (
        "<div style='margin-bottom:10px;'>"
        f"<div><b>value:</b> {value}</div>"
        "<div style='line-height:1.8;'>"
        f"{sign_bits} {exponent_bits} {mantissa_bits}"
        "</div>"
        "</div>"
    )


def ieee754_bits_table_html(values: torch.Tensor, bits_tensor: torch.Tensor) -> str:
    legend = (
        "<div style='margin-bottom:10px;'>"
        "<span style='background:#ffcc80; padding:2px 6px; margin-right:8px; color:#000; border-radius:4px;'>sign (1b)</span>"
        "<span style='background:#90caf9; padding:2px 6px; margin-right:8px; color:#000; border-radius:4px;'>exponent (11b)</span>"
        "<span style='background:#c5e1a5; padding:2px 6px; color:#000; border-radius:4px;'>mantissa (52b)</span>"
        "</div>"
    )
    rows = "".join(
        _render_ieee754_bits_row(float(values[i].item()), bits_tensor[i]) for i in range(values.shape[0])
    )
    return legend + rows


def combine_embeddings(
    inputs_embeds: torch.Tensor,
    num_encoding: torch.Tensor,
    number_mask: torch.Tensor,
) -> torch.Tensor:
    combined = inputs_embeds.clone()
    n_embed = inputs_embeds.shape[-1]
    emb_size = num_encoding.shape[-1]
    pad_size = n_embed - emb_size

    num_encoding = num_encoding.to(inputs_embeds.dtype)
    num_encoding = num_encoding.clone()
    num_encoding[number_mask] = num_encoding[number_mask] * 2 - 1

    padded = torch.nn.functional.pad(num_encoding[number_mask], (0, pad_size), value=0.0)
    combined[number_mask] = inputs_embeds[number_mask] + padded
    return combined
