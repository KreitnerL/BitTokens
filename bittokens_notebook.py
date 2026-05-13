# This file should be opened using the marimo extension: `marimo edit bittokens_notebook.py`

import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # BitTokens: An Interactive Introduction

    This notebook introduces the core contribution of **BitTokens** in three steps:

    - Traditional tokenizers break numbers into many pieces.
    - BitTokens keeps numbers as a dedicated single token pathway.
    - The model uses an efficient bit-level encoding for training and inference.

    ## What you will see

    1. Side-by-side tokenization for single-digit, subword, and bittokens.
    2. How `BitTokenEmbedding` encodes numbers and combines embeddings.
    3. How the linear numeric head predicts bits, computes training loss, and decodes numbers.
    """)
    return


@app.cell
def _():
    import re

    import torch

    from utils import notebook_utils as btn


    tokenizers = btn.load_all_tokenizers()
    print("Loaded tokenizers:", ", ".join(tokenizers.keys()))
    return btn, re, tokenizers, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Tokenization comparison: SD vs TD vs BitTokens

    Traditional tokenizers often split long numbers into many pieces. The BitToken tokenizer uses dedicated number tokens so numeric text can be represented more compactly.

    This section tokenizes the same text with three tokenizers and compares:

    - token pieces
    - token IDs
    - numeric-focused tokens (highlighted)
    - total token counts

    Related code:

    - [`analyze_token_counts.py`](analyze_token_counts.py)
    - [`utils/util_funcs.py`](utils/util_funcs.py)
    - [`dataloader/datasets/pretokenized_number_dataset.py`](dataloader/datasets/pretokenized_number_dataset.py)
    """)
    return


@app.cell
def _(btn, mo, re, tokenizers):
    sample_text = (
        "Sensor A reported 1234567890.123456, while sensor B dropped to -0.0000003141592653. "
        "A third run produced 602214076000000000000000.0 and 9.109383701531 in the same line."
    )
    NUMERIC_SPAN_REGEX = re.compile(r"[-]?(?:(?:0(?!\.[0-9]))|(?:[0-9]*[.][0-9]+)|(?:[1-9][0-9]*))")
    raw_numbers = btn.extract_raw_numbers(sample_text, regex_pattern=NUMERIC_SPAN_REGEX)
    mo.Html(btn.tokenization_comparison_html(sample_text, tokenizers, regex_pattern=NUMERIC_SPAN_REGEX))
    return NUMERIC_SPAN_REGEX, raw_numbers, sample_text


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why this matters

    - Fewer tokens for numeric-heavy text means more effective context length.
    - Numeric values can be handled by a dedicated pathway instead of only subword pieces.
    - This separation helps the model treat numeric magnitude and language context differently.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) How BitToken embedding works

    Relevant implementation paths:

    - `StemHeadModel.compute_input_embeddings`: [`networks/stem_head_model.py`](networks/stem_head_model.py)
    - `BitTokenEmbedding.forward` and `combine_embeds`: [`networks/number_embedding_modules/bittoken_embedding.py`](networks/number_embedding_modules/bittoken_embedding.py)

    At a high level:

    1. Detect number-token positions (`number_mask`).
    2. Encode only those numeric values into bit vectors.
    3. Combine token embeddings and numeric encodings at masked positions.

    The embedding path reinterprets `float64` storage as `int64` to access the IEEE-754 bit pattern directly, then extracts bits with vectorized tensor ops.
    """)
    return


@app.cell
def _(torch):
    float64_bit_shifts: torch.LongTensor = torch.arange(64 - 1, -1, -1, dtype=torch.int64)

    def float64_tensor_to_binary_tensor(tensor_in: torch.DoubleTensor) -> torch.LongTensor:
        """
        Converts a float64 PyTorch tensor to its IEEE 754 binary representation,
        returning the result as a new PyTorch integer tensor of bits.

        Args:
            tensor_in (torch.DoubleTensor): A PyTorch tensor with dtype torch.float64.

        Returns:
            base_extension (torch.LongTensor): A PyTorch tensor of dtype torch.int64 with shape (*tensor_in.shape, 64)
        """
        int_representation = tensor_in.view(torch.int64).unsqueeze(-1)
        bits = (int_representation >> float64_bit_shifts) & 1
        return bits

    def binary_tensor_to_float64_tensor(bits_int64: torch.Tensor) -> torch.Tensor:
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
        exponents = float64_bit_shifts
        weights = torch.tensor(1, dtype=torch.int64) << exponents
        reconstructed_int = torch.sum(bits_int64 * weights, dim=-1)
        return reconstructed_int.to(dtype=torch.int64).view(torch.float64).to(torch.float64)

    return binary_tensor_to_float64_tensor, float64_tensor_to_binary_tensor


@app.cell
def _(float64_tensor_to_binary_tensor, raw_numbers, torch):
    nums = torch.tensor([float(n) for n in raw_numbers], dtype=torch.float64)
    bits = float64_tensor_to_binary_tensor(nums)

    print("Numbers extracted from sample_text:", [float(n) for n in raw_numbers])
    print("Input shape:", tuple(nums.shape))
    print("Bit encoding shape (64-bit):", tuple(bits.shape))

    reciprocal_bits = float64_tensor_to_binary_tensor(nums.reciprocal())
    full_encoding = torch.cat([bits, reciprocal_bits], dim=-1)
    print("With reciprocal concat shape (128-bit):", tuple(full_encoding.shape))
    return bits, full_encoding, nums


@app.cell
def _(bits, btn, mo, nums):
    mo.Html(btn.ieee754_bits_table_html(nums, bits))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `StemHeadModel.compute_input_embeddings` follows this flow:

    - build normal token embeddings
    - compute `number_mask` from number token IDs
    - encode numeric values only at masked positions
    - combine token + numeric vectors into `combined_embeds`

    That keeps non-numeric token processing unchanged, while enriching numeric positions.
    """)
    return


@app.cell
def _(NUMERIC_SPAN_REGEX, btn, full_encoding, sample_text, tokenizers, torch):
    bittoken_tokenizer = tokenizers["bittoken_gpt2 (BitTokens)"]
    text_for_bittoken = btn.replace_numbers_with_num_token(sample_text, regex_pattern=NUMERIC_SPAN_REGEX)
    encoded = bittoken_tokenizer(text_for_bittoken, padding="do_not_pad", return_tensors=None)
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long).unsqueeze(0)

    vocab_size = int(len(bittoken_tokenizer))
    assert vocab_size > 0, f"Invalid tokenizer size: {vocab_size}"
    assert int(input_ids.min()) >= 0, f"Negative token id found: {int(input_ids.min())}"
    assert int(input_ids.max()) < vocab_size, (
        f"Token id {int(input_ids.max())} is out of range for vocab_size={vocab_size}"
    )

    embedding_layer = torch.nn.Embedding(vocab_size, 384)
    inputs_embeds = embedding_layer(input_ids)

    num_token_ids = torch.tensor(btn.get_num_token_ids_for_bittoken(bittoken_tokenizer), dtype=torch.long)
    number_mask = torch.isin(input_ids, num_token_ids)
    num_positions = int(number_mask.sum().item())

    num_encoding = torch.zeros((1, input_ids.shape[1], 128), dtype=torch.float32)
    num_encoding[number_mask] = full_encoding[:num_positions].float()

    combined = btn.combine_embeddings(inputs_embeds, num_encoding, number_mask)

    print("text_for_bittoken:", text_for_bittoken)
    print("vocab_size:", vocab_size)
    print("input_ids shape:", tuple(input_ids.shape))
    print("inputs_embeds shape:", tuple(inputs_embeds.shape))
    print("num_encoding shape:", tuple(num_encoding.shape))
    print("number_mask shape:", tuple(number_mask.shape), "| numeric positions:", num_positions)
    print("combined_embeds shape:", tuple(combined.shape))
    return combined, number_mask


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) Numeric head, training loss, and inference decode

    The numeric branch is configured with Linear layer number head.

    Conceptual path:

    - hidden state at numeric token positions -> linear head -> predicted bits
    - training: compare predictions to target bit encodings (`compute_num_loss`)
    - inference: threshold predicted bits, then convert bits back to float (`decode`)
    """)
    return


@app.cell
def _(bits, combined, number_mask, torch):
    hidden_states = combined
    output_size = bits.shape[-1]

    num_head_linear = torch.nn.Linear(hidden_states.shape[-1], output_size)
    pred_bits_logits = num_head_linear(hidden_states[number_mask])

    print("Hidden states shape:", tuple(hidden_states.shape))
    print("Masked numeric positions:", int(number_mask.sum()))
    print("Predicted bit-logits shape:", tuple(pred_bits_logits.shape))
    return output_size, pred_bits_logits


@app.cell
def _(bits, output_size, pred_bits_logits, torch):
    target_bits = bits.to(torch.float32)
    assert target_bits.shape == pred_bits_logits.shape, (
        f"Target/pred shape mismatch: {target_bits.shape} vs {pred_bits_logits.shape}"
    )

    freq_weights = torch.ones(output_size).unsqueeze(0)

    bce_per_bit = torch.nn.functional.binary_cross_entropy_with_logits(
        pred_bits_logits,
        target_bits,
        reduction="none",
    )
    weighted_bce = bce_per_bit * freq_weights
    loss = weighted_bce.mean(dim=-1)

    print("Target bits shape:", tuple(target_bits.shape))
    print("Mean weighted BCE loss:", loss.mean().item())
    return


@app.cell
def _(binary_tensor_to_float64_tensor, bits, nums, torch):
    noisy_pred_logits = bits * 2 - 1 + torch.rand_like(bits.float()).sub_(0.5)
    x_base_digits_pred: torch.LongTensor = (noisy_pred_logits > 0).to(torch.int64)
    num_preds = binary_tensor_to_float64_tensor(x_base_digits_pred)
    target_value = nums[0]
    print("Pred bits shape (first 64):", tuple(x_base_digits_pred.shape))
    print("Decoded float value:", float(num_preds[0]))
    print("Target sample_text value:", float(target_value))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Benefits recap

    - **D1 - Token efficiency:** Every number is represented by a single token. Fewer tokens for numeric-heavy text can preserve context for surrounding language.
    - **D2 - Uniqueness:** Each value has exactly one valid encoding, with a unique inverse mapping.
    - **D3 - Structured:** The encoding geometry reflects numeric order and distance, facilitating generalizable algorithms.
    - **D4 - Scale invariance:** A large range of input magnitudes and precisions can be represented.
    - **D5 - Normalization:** Encodings are bounded and information preserving under standard normalization functions used in language models (e.g., LayerNorm, RMSNorm)
    - **D6 - Numerical stability:** Representations remain accurate when using low-precision activations (e.g., FP8)
    - **D7 - Continuity:** Encodings vary relatively smoothly with the underlying value, making them compatible with gradient-based optimization
    - **D8 - Robustness:** Values can be decoded reliably under stochastic noise, allowing for stochastic training.
    - **D9 - Arithmetic:** Encodings admit learnable algorithms for core mathematical operations.

    ## Where to read next in the codebase

    1. [`networks/number_embedding_modules/bittoken_embedding.py`](networks/number_embedding_modules/bittoken_embedding.py)
    2. [`networks/stem_head_model.py`](networks/stem_head_model.py)
    3. [`tokenizers/create_tokenizers.py`](tokenizers/create_tokenizers.py)
    """)
    return


if __name__ == "__main__":
    app.run()
