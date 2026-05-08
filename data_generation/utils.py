import math
import random
import re
import struct
from typing import List, Literal, Optional, Tuple

import numpy as np

from data_generation.data_gen_utils import (
    Generation_settings,
    SignificantDigitsDistribution,
)


def get_number_of_significant_digits(base_number: str) -> int:
    """Count the number of significant digits in a base number string.
    
    Args:
        base_number: String representation of a number in any base
    Returns:
        Number of significant digits (non-zero digits)
    """    
    return len(base_number.removeprefix("-").replace(".", "").lstrip('0').rstrip('0'))

def quantize_base_str(base_str: str, num_digits: int) -> str:
    """Quantize a base string to by setting the num_digits least significant digits to zero.
    Args:
        base_str: String representation of a number in any base
        num_digits: Number of least significant digits to set to zero
    Returns:
        Quantized base string with specified number of least significant digits set to zero
    """
    base_str = base_str[:-num_digits or None] + "0" * num_digits
    return base_str.lstrip('0') or '0'

def get_pos_of_lsb(base_str: str) -> int:
    """Get the position of the least significant bit in a base string.
    
    Args:
        base_str: String representation of a number in any base
        base: The base of the number system (2-36)
    Returns:
        Position of the least significant bit (0-indexed from the right)
    """ 
    # Remove leading zeros
    base_str = base_str.lstrip('0')[::-1]
    
    # Find the position of the least significant digit
    match = re.search(r'[1-9A-Z]', base_str)
    if match:
        return match.start()
    return -1

def base_string_to_float(base_str: str, base: int) -> float:
    """Convert base string to float without intermediate int conversion.
    
    Args:
        base_str: String representation of a number in any base (e.g., "101.101" for binary)
        base: The base of the number system (2-36)
    
    Returns:
        float: The decimal representation of the base string
    """
    if not base_str:
        return 0.0
    
    # Handle negative numbers
    is_negative = base_str.startswith('-')
    if is_negative:
        base_str = base_str[1:]
    
    # Split into integer and fractional parts
    if '.' in base_str:
        int_part, frac_part = base_str.split('.', 1)
    else:
        int_part, frac_part = base_str, ""
    
    # Remove leading zeros from integer part
    int_part = int_part.lstrip('0') or '0'
    
    # Remove trailing zeros from fractional part
    frac_part = frac_part.rstrip('0')
    
    # If both parts are empty or zero, return 0
    if int_part == '0' and not frac_part:
        return 0.0
    
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    char_to_val = {digits[i]: i for i in range(base)}
    
    result = 0.0
    
    # Process integer part
    for char in int_part.upper():
        if char not in char_to_val:
            raise ValueError(f"Invalid character '{char}' for base {base}")
        digit_value = char_to_val[char]
        result = result * base + digit_value
    
    # Process fractional part
    if frac_part:
        frac_value = 0.0
        base_power = 1.0 / base
        
        for char in frac_part.upper():
            if char not in char_to_val:
                raise ValueError(f"Invalid character '{char}' for base {base}")
            digit_value = char_to_val[char]
            frac_value += digit_value * base_power
            base_power /= base
        
        result += frac_value
    
    return -result if is_negative else result

def float_to_base(number: float, base: int) -> str:
    """Convert a float number to specified base representation."""
    if base < 2 or base > 36:
        raise ValueError("Base must be between 2 and 36")
    
    int_number = int(number)
    if int_number != number:
        raise ValueError(f"Number {number} must be an integer (no fractional part)")
    
    if int_number == 0:
        return "0"
    
    if int_number < 0:
        raise ValueError("Number must be non-negative")
    
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = []
    
    while int_number > 0:
        int_number, remainder = divmod(int_number, base)
        result.append(digits[remainder])
    
    return ''.join(reversed(result))


def round_decimal_str_to_significant_digits(number_str: str, max_sig_digits: int) -> str:
    """
    Round a base10 string representation to the given maximum number of significant digits.

    Args:
        number_str (str): String representation of a number in base10.
        max_sig_digits (int): Maximum number of significant digits to keep.

    Returns:
        str: Rounded string representation of the number.
    """
    if not number_str or max_sig_digits <= 0:
        return "0"
    
    # Convert to float for mathematical operations
    try:
        x = float(number_str)
    except ValueError:
        return "0"
    
    if x == 0.0:
        return "0"
    
    # Extract sign
    sign = "-" if x < 0 else ""
    x = abs(x)
    
    # Find the order of magnitude of the most significant digit
    if x >= 1.0:
        magnitude = math.floor(math.log10(x))
    else:
        magnitude = math.floor(math.log10(x))
    
    # Scale the number so the most significant digit is in the ones place
    scaled = x / (10.0 ** magnitude)
    
    # Round to max_sig_digits decimal places (since we scaled to ones place)
    scale_factor = 10.0 ** (max_sig_digits - 1)
    rounded_scaled = round(scaled * scale_factor) / scale_factor
    
    # Scale back to original magnitude
    result = rounded_scaled * (10.0 ** magnitude)
    
    # If rounding caused us to exceed 1e15, round down instead
    if abs(result) >= 1e15:
        # Round down by truncating instead of rounding
        truncated_scaled = math.floor(scaled * scale_factor) / scale_factor
        result = truncated_scaled * (10.0 ** magnitude)
    elif abs(result) < 1e-15:
        result = 0
    
    # Convert back to string with appropriate precision
    if result == 0:
        return "0"
    
    # Format the result to avoid scientific notation
    if abs(result) >= 1e15 or abs(result) < 1e-15:
        # Use scientific notation for very large or very small numbers
        raise ValueError(f"result={result} is too large or too small for fixed-point representation. x={x}")
    else:
        # Use fixed point notation
        if magnitude >= 0:
            # Integer or mixed number
            decimals = max(0, max_sig_digits - magnitude - 1)
            result_str = f"{result:.{decimals}f}"
        else:
            # Pure decimal
            decimals = max_sig_digits - magnitude - 1
            result_str = f"{result:.{decimals}f}"
        
        # Remove trailing zeros
        if '.' in result_str:
            result_str = result_str.rstrip('0').rstrip('.')
    
    return sign + result_str

def get_rounded_base2_expansion_of_float(value: float, significant_bits: int, precision: Literal[32, 64] = 64) -> tuple[str, float]:
    """
    Returns the binary representation of a float as a string.
    Example: 0.15625 -> '0.00101'
    Args:
        value (float): The float value to convert.
        significant_bits (int): Number of significant bits to keep.
        precision (int): Bit precision (32 for float32, 64 for float64)
    Returns:
        tuple[str, float]: The binary representation and the (possibly rounded) float value.
    """
    # Handle special cases
    if value == 0.0:
        return '0.0', 0.0
    if not math.isfinite(value):
        nan_inf_str = 'nan' if math.isnan(value) else ('inf' if value > 0 else '-inf')
        return nan_inf_str, value
    assert significant_bits >= 0, f"significant_bits must be >= 0 but was {significant_bits}"
    
    # Set parameters based on precision
    if precision == 32:
        fmt = '>f'
        unpack_fmt = '>L'
        pack_fmt = '>L'
        exp_bits = 8
        mantissa_bits = 23
        bias = 127
    elif precision == 64:
        fmt = '>d'
        unpack_fmt = '>Q'
        pack_fmt = '>Q'
        exp_bits = 11
        mantissa_bits = 52
        bias = 1023
    else:
        raise ValueError("Precision must be 32 or 64")
    
    # Extract IEEE 754 components
    bits = struct.unpack(unpack_fmt, struct.pack(fmt, value))[0]
    sign = bits >> (exp_bits + mantissa_bits)
    raw_exponent = (bits >> mantissa_bits) & ((1 << exp_bits) - 1)
    mantissa = bits & ((1 << mantissa_bits) - 1)
    
    # Handle special/denormalized numbers
    if raw_exponent == (1 << exp_bits) - 1:  # All exponent bits set
        result_str = ('-inf' if sign else 'inf') if mantissa == 0 else 'nan'
        return result_str, value
    
    # Apply rounding using IEEE 754 "round to nearest, ties to even"
    rounded_value = value
    if significant_bits > 0 and raw_exponent > 0:  # Only round normalized numbers
        bits_to_keep = max(0, significant_bits - 1)  # -1 for implicit leading bit
        if bits_to_keep < mantissa_bits:
            bits_to_zero = mantissa_bits - bits_to_keep
            
            # Extract the bits we're about to remove
            removed_bits_mask = (1 << bits_to_zero) - 1
            removed_bits = mantissa & removed_bits_mask
            
            # Check if we need to round up
            # Round up if: removed bits > halfway point, or removed bits == halfway and LSB of kept bits is 1
            halfway_point = 1 << (bits_to_zero - 1)
            lsb_of_kept = (mantissa >> bits_to_zero) & 1
            
            should_round_up = (removed_bits > halfway_point) or \
                            (removed_bits == halfway_point and lsb_of_kept == 1)
            
            # Clear the bits we're removing
            mask = ~removed_bits_mask
            mantissa &= mask
            
            # Apply rounding if needed
            if should_round_up:
                # Add 1 to the truncated mantissa
                mantissa += (1 << bits_to_zero)
                
                # Check for mantissa overflow
                if mantissa >= (1 << mantissa_bits):
                    # Mantissa overflowed, increment exponent and reset mantissa
                    mantissa = 0
                    raw_exponent += 1
                    
                    # Check for exponent overflow
                    if raw_exponent >= (1 << exp_bits) - 1:
                        # Overflow to infinity
                        result_str = '-inf' if sign else 'inf'
                        return result_str, float('inf') if not sign else float('-inf')
            
            # Reconstruct float
            new_bits = (sign << (exp_bits + mantissa_bits)) | (raw_exponent << mantissa_bits) | mantissa
            rounded_value = struct.unpack(fmt, struct.pack(pack_fmt, new_bits))[0]
    
    # Generate binary string from rounded value
    if rounded_value == 0.0:
        return '0.0', rounded_value
    
    # Re-extract components for string generation
    bits = struct.unpack(unpack_fmt, struct.pack(fmt, rounded_value))[0]
    sign = bits >> (exp_bits + mantissa_bits)
    raw_exponent = (bits >> mantissa_bits) & ((1 << exp_bits) - 1)
    mantissa = bits & ((1 << mantissa_bits) - 1)
    
    if raw_exponent == 0:  # Denormalized
        if mantissa == 0:
            return '-0.0' if sign else '0.0', rounded_value
        actual_exponent, mantissa_str = 1 - bias, f"{mantissa:0{mantissa_bits}b}".lstrip('0')
    else:  # Normalized
        actual_exponent, mantissa_str = raw_exponent - bias, f"1{mantissa:0{mantissa_bits}b}".rstrip('0')
    
    # Position binary point and format result
    binary_point_pos = actual_exponent + 1
    if binary_point_pos <= 0:
        result = f"0.{'0' * (-binary_point_pos)}{mantissa_str}"
    elif binary_point_pos >= len(mantissa_str):
        result = f"{mantissa_str}{'0' * (binary_point_pos - len(mantissa_str))}.0"
    else:
        int_part, frac_part = mantissa_str[:binary_point_pos], mantissa_str[binary_point_pos:]
        result = f"{int_part}.{frac_part or '0'}"
    result = result.rstrip('0').rstrip('.') if '.' in result else result
    return (f"-{result}" if sign else result), rounded_value

def float_to_str(value: float) -> str:
    """Convert a float to a fixed-point string with sufficient precision to ensure it can be accurately represented.
    
    Args:
        value (float): The float value to convert.
    Returns:
        str: String representation of the float with sufficient precision.
    Raises:
        ValueError: If the float cannot be represented with sufficient precision.
    """
    repr_str = repr(value)
    if 'e' not in repr_str.lower():
        return repr_str.rstrip('0').rstrip('.')
    for p in range(53):
        s = f"{value:.{p}f}"
        if float(s) == value:
            return s.rstrip('0').rstrip('.')
    raise ValueError(f"Cannot convert {value} to string with sufficient precision")

def truncated_exponential(a, b, k, size=1) -> np.ndarray:
    """
    Samples from a truncated exponential distribution between a and b with rate k.
    Args:
        a (float): Lower bound of the distribution.
        b (float): Upper bound of the distribution.
        k (float): Rate parameter of the exponential distribution.
        size (int): Number of samples to generate.
    Returns:
        np.ndarray: Samples from the truncated exponential distribution.
    """
    u = np.random.uniform(0, 1, size)
    exp_kba = np.exp(k * (b - a))
    return a + (1 / k) * np.log(1 + u * (exp_kba - 1))

def difficulty_to_int(number_str: str) -> int:
    """
    Count the number of non zero digits in the difficulty string
    Args:
        number_str (str): Number string in any base
    Returns:
        int: Number of non zero digits
    """
    return sum(1 for c in number_str.upper() if (c.isdigit() and c != '0') or c.isalpha())

def generate_numbers(
    m: float, 
    s: float, 
    k: int, 
    max_num_significant_bits: Optional[int] = None, 
    max_num_significant_digits: Optional[int] = None
) -> List[float]:
    """
    Generate k IEEE-754 float64 values such that:
      • strict bounds: each value is in [m - s, m + s],
      • each value has at most `max_num_significant_bits` bits in the binary significand
        (i.e., p-bit precision, 1 <= p <= 53),
      • the resulting mean is the closest achievable mean to m under the above constraint
        on operands (we search a single global offset to minimize post-quantization error).

    Approach:
      1) Sample k base values uniformly in [L, U], with L = m - s, U = m + s.
      2) Use a monotone bounded p-bit quantizer Q_p^{[L,U]}(x) (nearest ties-to-even within bounds).
      3) Find a common shift δ in [δ_lo, δ_hi] via binary search to make
         sum_i Q_p^{[L,U]}(x_i + δ) closest to k*m.
      4) Return the quantized values at the best δ.

    Feasibility:
      • There must exist at least one p-bit number in [L, U], i.e. ceil_p(L) <= floor_p(U).
        Otherwise this is impossible and a ValueError is raised.

    Notes:
      • The quantizer ensures “at most p significant bits” by rounding to a p-bit significand
        (nearest, ties-to-even) and clamping to [L, U] on the p-bit grid.
      • The search is 1D and efficient (O(k log iterations)), and respects strict bounds.

    Args:
        m (float): Desired mean (target).
        s (float): Half-width of interval around m (s >= 0).
        k (int): Number of values (k >= 1).
        max_num_significant_bits (Optional[int]): Maximum number of significand bits allowed (1..53).
            Mutually exclusive with max_num_significant_digits.
        max_num_significant_digits (Optional[int]): Maximum number of significant digits in base-10 (1..15).
            Mutually exclusive with max_num_significant_bits.

    Returns:
        numbers (List[float]): k precision-limited floats in [m - s, m + s] whose mean is the closest
                     achievable to m under the precision constraint.

    Raises:
        ValueError: On invalid inputs, missing precision parameters, or infeasible constraints.
    """
    # --- Validation ---
    if not (isinstance(k, int) and k >= 1):
        raise ValueError("k must be an integer >= 1.")
    if not (isinstance(s, (int, float)) and s >= 0 and math.isfinite(s)):
        raise ValueError("s must be a finite number >= 0.")
    if not (isinstance(m, (int, float)) and math.isfinite(m)):
        raise ValueError("m must be a finite number.")
    
    # Validate precision parameters - exactly one must be provided
    if max_num_significant_bits is not None and max_num_significant_digits is not None:
        raise ValueError("Cannot specify both max_num_significant_bits and max_num_significant_digits.")
    if max_num_significant_bits is None and max_num_significant_digits is None:
        raise ValueError("Must specify either max_num_significant_bits or max_num_significant_digits.")
    
    # Determine precision type and validate range
    use_binary_precision = max_num_significant_bits is not None
    if use_binary_precision:
        p = max_num_significant_bits
        if not (isinstance(p, int) and 1 <= p <= 53):
            raise ValueError("max_num_significant_bits (p) must be an integer in [1, 53].")
    else:
        d = max_num_significant_digits
        if not (isinstance(d, int) and 1 <= d <= 17):
            raise ValueError("max_num_significant_digits (d) must be an integer in [1, 17].")

    L = m - s
    U = m + s
    if not (math.isfinite(L) and math.isfinite(U)):
        raise ValueError("Bounds must be finite.")
    if L > U:
        raise ValueError("Invalid bounds: m - s must be <= m + s.")

    # --- Feasibility: there must exist at least one value with specified precision in [L, U] ---
    if use_binary_precision:
        y_min = _ceil_p(L, p)
        y_max = _floor_p(U, p)
        if y_min > y_max:
            raise ValueError(
                f"Infeasible: No value with at most p={p} significant bits lies in [m - s, m + s]=[{m}-{s}, {m}+{s}]. "
                f"[L, U]=[{L}, {U}], p={p}"
            )
    else:
        y_min = _ceil_d(L, d)
        y_max = _floor_d(U, d)
        if y_min > y_max:
            raise ValueError(
                f"Infeasible: No value with at most d={d} significant digits lies in [m - s, m + s]=[{m}-{s}, {m}+{s}]. "
                f"[L, U]=[{L}, {U}], d={d}"
            )

    # --- 1) Sample k values uniformly within [L, U] ---
    samples = [random.uniform(L, U) for _ in range(k)]

    # Pre-compute sample bounds for delta calculation
    min_x = min(samples)
    max_x = max(samples)

    # --- 2) Define optimized helpers ---
    if use_binary_precision:
        def quantized_sum_only(delta: float) -> float:
            """Fast sum calculation without storing individual values.
            Args:
                delta (float): Shift to apply to each sample before quantization.
            Returns:
                float: Sum of quantized values.
            """
            return sum(_quantize_bounded_p(x + delta, p, L, U) for x in samples)

        def quantized_sum_with_values(delta: float) -> Tuple[float, List[float]]:
            """Sum calculation with individual values for final result.
            Args:
                delta (float): Shift to apply to each sample before quantization.
            Returns:
                Tuple[float, List[float]]: Sum of quantized values and the list of quantized values.
            """
            values = [_quantize_bounded_p(x + delta, p, L, U) for x in samples]
            return sum(values), values
    else:
        def quantized_sum_only(delta: float) -> float:
            """Fast sum calculation without storing individual values.
            Args:
                delta (float): Shift to apply to each sample before quantization.
            Returns:
                float: Sum of quantized values.
            """
            return sum(_quantize_bounded_d(x + delta, d, L, U) for x in samples)

        def quantized_sum_with_values(delta: float) -> Tuple[float, List[float]]:
            """Sum calculation with individual values for final result.
            Args:
                delta (float): Shift to apply to each sample before quantization.
            Returns:
                Tuple[float, List[float]]: Sum of quantized values and the list of quantized values.
            """
            values = [_quantize_bounded_d(x + delta, d, L, U) for x in samples]
            return sum(values), values

    # --- 3) δ search interval: keep all x_i + δ inside [L, U] before quantization ---
    # (Not strictly necessary with bounded quantizer, but keeps distribution stable.)
    delta_lo = L - max_x
    delta_hi = U - min_x
    if delta_lo > delta_hi:
        # This shouldn't happen because samples are within [L, U],
        # but guard against FP quirks.
        delta_lo, delta_hi = delta_hi, delta_lo

    target_sum = k * m

    # Evaluate endpoints
    s_lo, out_lo = quantized_sum_with_values(delta_lo)
    s_hi, out_hi = quantized_sum_with_values(delta_hi)

    # If monotone steps don't cross the target, pick the nearer endpoint
    if (s_lo >= target_sum and s_hi >= target_sum) or (s_lo <= target_sum and s_hi <= target_sum):
        # Choose whichever is closer to target_sum
        if abs(s_lo - target_sum) <= abs(s_hi - target_sum):
            return out_lo
        else:
            return out_hi

    # --- 4) Binary search for δ giving closest achievable sum ---
    lo = delta_lo
    hi = delta_hi
    best_err = float("inf")
    best_out = None

    # 60 iterations is plenty for float64 resolution
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        s_mid = quantized_sum_only(mid)

        # Update best if this is better
        err = abs(s_mid - target_sum)
        if err < best_err:
            best_err = err
            _, best_out = quantized_sum_with_values(mid)

        # Monotone step function: if below target, move right; else left
        if s_mid <= target_sum:
            lo = mid
        else:
            hi = mid

    # Also consider the final bracket endpoints to ensure nearest
    s_lo_final, out_lo_final = quantized_sum_with_values(lo)
    s_hi_final, out_hi_final = quantized_sum_with_values(hi)

    # Find the best result among all candidates
    candidates = [
        (abs(s_lo_final - target_sum), out_lo_final),
        (abs(s_hi_final - target_sum), out_hi_final),
        (best_err, best_out),
    ]
    result = min(candidates, key=lambda t: t[0])[1]

    # Final sanity check: strict bounds
    assert all(L <= y <= U for y in result), "Value out of bounds."

    return result


# ------------------------ d-digit (base-10) quantization helpers ------------------------ #

def _round_to_sigdigits_nearest(x: float, d: int) -> float:
    """
    Round x to a value with at most d significant digits in base-10
    (nearest with ties to even when possible).
    
    Args:
        x (float): Input float value.
        d (int): Maximum number of significant digits (1 <= d <= 17).
    
    Returns:
        float: Rounded float value with at most d significant digits.
    
    Raises:
        ValueError: If d is not in [1, 17].
    """
    if not (isinstance(d, int) and 1 <= d <= 17):
        raise ValueError("d must be an integer in [1, 17].")
    
    if x == 0.0:
        return 0.0
    
    # Convert to string, round using existing function, then back to float
    x_str = float_to_str(x)
    rounded_str = round_decimal_str_to_significant_digits(x_str, d)
    return float(rounded_str)


def _floor_d(x: float, d: int) -> float:
    """Greatest d-digit value <= x.
    
    Args:
        x (float): Input float value.
        d (int): Maximum number of significant digits (1 <= d <= 15).
    
    Returns:
        float: Greatest d-digit value less than or equal to x.
    
    Raises:
        ValueError: If d is not in [1, 15].
    """
    if x == 0.0:
        return 0.0
    if x < 0.0:
        # floor for negative = -ceil for positive
        return -_ceil_d(-x, d)
    
    # For positive x, first try rounding
    rounded = _round_to_sigdigits_nearest(x, d)
    if rounded <= x:
        return rounded
    
    # If rounding gives us something > x, we need to truncate
    # Use a more direct mathematical approach
    if x <= 0:
        return 0.0
    
    # Find the order of magnitude of the first significant digit
    log_x = math.log10(x)
    magnitude = math.floor(log_x)
    
    # Scale x to have first digit in the ones place
    scaled_x = x / (10.0 ** magnitude)
    
    # Truncate to d significant digits
    scale_factor = 10.0 ** (d - 1)
    truncated_scaled = math.floor(scaled_x * scale_factor) / scale_factor
    
    # Scale back
    result = truncated_scaled * (10.0 ** magnitude)
    
    # Ensure result has exactly d significant digits
    return _round_to_sigdigits_nearest(result, d)


def _ceil_d(x: float, d: int) -> float:
    """Smallest d-digit value >= x.
    
    Args:
        x (float): Input float value.
        d (int): Maximum number of significant digits (1 <= d <= 15).
    
    Returns:
        float: Smallest d-digit value greater than or equal to x.
    
    Raises:
        ValueError: If d is not in [1, 15].
    """
    if x == 0.0:
        return 0.0
    if x < 0.0:
        # ceil for negative = -floor for positive
        return -_floor_d(-x, d)
    
    # For positive x, first try rounding
    rounded = _round_to_sigdigits_nearest(x, d)
    if rounded >= x:
        return rounded
    
    # If rounding gives us something < x, we need to go up
    if x <= 0:
        return 0.0
    
    # Find the order of magnitude of the first significant digit
    log_x = math.log10(x)
    magnitude = math.floor(log_x)
    
    # Scale x to have first digit in the ones place
    scaled_x = x / (10.0 ** magnitude)
    
    # Ceiling to d significant digits
    scale_factor = 10.0 ** (d - 1)
    ceiled_scaled = math.ceil(scaled_x * scale_factor) / scale_factor
    
    # Scale back
    result = ceiled_scaled * (10.0 ** magnitude)
    
    # Ensure result has exactly d significant digits
    return _round_to_sigdigits_nearest(result, d)


def _quantize_bounded_d(x: float, d: int, L: float, U: float) -> float:
    """
    Nearest d-digit value to x, restricted to [L, U] over the set of d-digit values.
    If the nearest lies outside [L, U], choose the nearest within [L, U].
    Assumes there exists at least one d-digit value in [L, U].
    
    Args:
        x (float): Input float value.
        d (int): Maximum number of significant digits (1 <= d <= 15).
        L (float): Lower bound.
        U (float): Upper bound.
    
    Returns:
        float: Quantized float value within [L, U].
    
    Raises:
        ValueError: If d is not in [1, 15] or if no d-digit value exists in [L, U].
    """
    y = _round_to_sigdigits_nearest(x, d)
    if y < L:
        return _ceil_d(L, d)
    if y > U:
        return _floor_d(U, d)
    return y


# ------------------------ p-bit quantization helpers ------------------------ #

def _round_half_to_even_int(x: float) -> int:
    """Round a non-negative real x to nearest integer with ties to even.
    Args:
        x (float): Non-negative real number to round.
    Returns:
        int: Nearest integer to x, with ties rounded to the nearest even integer.
    Raises:
        ValueError: If x is negative.
    """
    # Assumes x >= 0
    floor = math.floor(x)
    diff = x - floor
    if diff > 0.5:
        return floor + 1
    if diff < 0.5:
        return floor
    # tie: return even
    return floor if (floor % 2 == 0) else floor + 1


def _round_to_sigbits_nearest(x: float, p: int) -> float:
    """
    Round x to a value whose normalized significand has at most p bits
    (nearest with ties to even). Returns a float64 value (still stored as Python float).
    Args:
        x (float): Input float value.
        p (int): Number of bits in the significand (1 <= p <= 53).
    Returns:
        float: Rounded float value with at most p bits in the significand.
    Raises:
        ValueError: If p is not in [1, 53].
    """
    if x == 0.0:
        return 0.0
    sign = -1.0 if x < 0 else 1.0
    ax = abs(x)

    # ax = m * 2^e2 with m in [0.5, 1)
    m, e2 = math.frexp(ax)
    # Normalize to [1, 2)
    m *= 2.0
    e = e2 - 1

    # Number of fractional bits to keep = p-1
    scale = 1 << max(p - 1, 0)  # p >= 1, so ok
    frac = m - 1.0
    scaled = frac * scale
    n = _round_half_to_even_int(scaled)

    if n >= scale:
        # Rounds up to 2.0 -> renormalize to 1.0 and bump exponent
        e += 1
        m_p = 1.0
    else:
        m_p = 1.0 + (n / scale)

    y = math.ldexp(m_p, e)
    return sign * y


def _floor_p(x: float, p: int) -> float:
    """Greatest p-bit value <= x.
    Args:
        x (float): Input float value.
        p (int): Number of bits in the significand (1 <= p <= 53).
    Returns:
        float: Greatest p-bit value less than or equal to x.
    Raises:
        ValueError: If p is not in [1, 53].
    """
    if x == 0.0:
        return 0.0
    if x < 0.0:
        # floor for negative = -ceil for positive
        return -_ceil_p(-x, p)

    # x > 0
    m, e2 = math.frexp(x)
    m *= 2.0
    e = e2 - 1
    scale = 1 << max(p - 1, 0)
    frac = m - 1.0
    n = int(math.floor(frac * scale))
    # n in [0, scale-1]
    m_p = 1.0 + (n / scale) if scale > 0 else 1.0
    return math.ldexp(m_p, e)


def _ceil_p(x: float, p: int) -> float:
    """Smallest p-bit value >= x.
    Args:
        x (float): Input float value.
        p (int): Number of bits in the significand (1 <= p <= 53).
    Returns:
        float: Smallest p-bit value greater than or equal to x.
    Raises:
        ValueError: If p is not in [1, 53].
    """
    if x == 0.0:
        return 0.0
    if x < 0.0:
        # ceil for negative = -floor for positive
        return -_floor_p(-x, p)

    # x > 0
    m, e2 = math.frexp(x)
    m *= 2.0
    e = e2 - 1
    scale = 1 << max(p - 1, 0)
    frac = m - 1.0
    n = int(math.ceil(frac * scale))  # may be == scale at top edge
    if n >= scale:
        # next is exactly 2.0 -> becomes 1.0 at exponent e+1
        return math.ldexp(1.0, e + 1)
    m_p = 1.0 + (n / scale) if scale > 0 else 1.0
    return math.ldexp(m_p, e)


def _quantize_bounded_p(x: float, p: int, L: float, U: float) -> float:
    """
    Nearest (ties-to-even) p-bit value to x, restricted to [L, U] over the set of p-bit values.
    If the nearest lies outside [L, U], choose the nearest within [L, U].
    Assumes there exists at least one p-bit value in [L, U].
    Args:
        x (float): Input float value.
        p (int): Number of bits in the significand (1 <= p <= 53).
        L (float): Lower bound.
        U (float): Upper bound.
    Returns:
        float: Quantized float value within [L, U].
    Raises:
        ValueError: If p is not in [1, 53] or if no p-bit value exists in [L, U].
    """
    y = _round_to_sigbits_nearest(x, p)
    if y < L:
        return _ceil_p(L, p)
    if y > U:
        return _floor_p(U, p)
    return y

def generate_quotient_with_precision(exp_range: tuple[float, float], quotient_sig_bits: int, gs: Generation_settings) -> tuple[str, float]:
    """Generate a quotient with controlled precision within the specified exponent range.
    
    Args:
        exp_range: (min_exp, max_exp) for the quotient magnitude
        quotient_sig_bits: Number of significant bits for the quotient
        gs: Generation settings
        
    Returns:
        tuple[str, float]: Binary representation and float value of the quotient
    """
    # With fixed exponent range [-15, 15] and base 10, simplify bounds
    exp_min, exp_max = exp_range
    # Clamp to safe range within [-15, 15]
    exp_min = max(exp_min, -14)
    exp_max = min(exp_max, 14)
    
    # Generate quotient in the range [10^exp_min, 10^exp_max)
    min_quotient = 10.0 ** exp_min
    max_quotient = 10.0 ** exp_max
    
    quotient = random.uniform(min_quotient, max_quotient)
    
    # Apply precision control
    if gs.significant_digits_distribution == SignificantDigitsDistribution.DECIMAL_UNIFORM:
        quotient_base_10_str = float_to_str(quotient)
        quotient_base_10_str = round_decimal_str_to_significant_digits(quotient_base_10_str, quotient_sig_bits)
        quotient = float(quotient_base_10_str)
        quotient_base2_str, _ = get_rounded_base2_expansion_of_float(quotient, gs.significant_bits)
    else:
        quotient_base2_str, quotient = get_rounded_base2_expansion_of_float(quotient, quotient_sig_bits)
    
    return quotient_base2_str, quotient

def generate_divisor_for_quotient(quotient_fXX: float, exp2: int, divisor_sig_bits: int, gs: Generation_settings, precision: int) -> tuple[str, float, str]:
    """Generate a divisor that will produce the target quotient.
    
    Args:
        quotient_fXX: Target quotient value
        exp2: Exponent range for the divisor
        divisor_sig_bits: Number of significant bits for the divisor
        gs: Generation settings
        precision: Precision for floating point representation
        
    Returns:
        tuple[str, float, str]: Binary representation, float value, and base-10 string of divisor
    """
    # With base 10 and exponent range [-15, 15], generate divisor in [10^exp2, 10^(exp2+1))
    min_number2 = 10.0 ** exp2
    max_number2 = 10.0 ** (exp2 + 1)
    
    num2 = random.uniform(min_number2, max_number2)
    
    if gs.significant_digits_distribution == SignificantDigitsDistribution.DECIMAL_UNIFORM:
        num2_base_10_str = float_to_str(num2)
        num2_base_10_str = round_decimal_str_to_significant_digits(num2_base_10_str, divisor_sig_bits)
        num2 = float(num2_base_10_str)
        num2_base2_str, _ = get_rounded_base2_expansion_of_float(num2, precision)
    else:
        num2_base2_str, num2 = get_rounded_base2_expansion_of_float(num2, divisor_sig_bits)
        num2_base_10_str = float_to_str(num2)
    
    return num2_base2_str, num2, num2_base_10_str

def compute_dividend_from_quotient_divisor(quotient_fXX: float, num2_fXX: float, exp1: int, gs: Generation_settings) -> tuple[bool, float, str, str]:
    """Compute dividend from quotient and divisor, validate it's within bounds.
    
    Args:
        quotient_fXX: Quotient value
        num2_fXX: Divisor value  
        exp1: Expected exponent range for dividend
        gs: Generation settings
        
    Returns:
        tuple[bool, float, str, str]: (is_valid, dividend_value, binary_repr, base10_str)
    """
    try:
        num1_fXX = quotient_fXX * num2_fXX

        if gs.significant_digits_distribution == SignificantDigitsDistribution.DECIMAL_UNIFORM:
            num1_base_10_str = float_to_str(num1_fXX)
            num1_base_10_str = round_decimal_str_to_significant_digits(num1_base_10_str, gs.significant_digits)
            num1_fXX = float(num1_base_10_str)
            num1_base2_str, _ = get_rounded_base2_expansion_of_float(num1_fXX, gs.significant_bits)
        else:
            num1_base2_str, num1_fXX = get_rounded_base2_expansion_of_float(num1_fXX, gs.significant_bits)
            num1_base_10_str = float_to_str(num1_fXX)
        
        # Check if dividend is within global bounds (should be guaranteed with [-15,15] range)
        if not (1e-15 <= abs(num1_fXX) <= 1e15):
            return False, num1_fXX, "", ""
        
        # Check if dividend is in the expected exponent range [10^exp1, 10^(exp1+1))
        # Allow some flexibility since we're working backwards from quotient
        min_number1 = 10.0 ** (exp1 - 1)  # Allow one order of magnitude flexibility
        max_number1 = 10.0 ** (exp1 + 2)   # Allow one order of magnitude flexibility
        
        if not (min_number1 <= abs(num1_fXX) < max_number1):
            return False, num1_fXX, "", ""
        
        # num1_base2_str, num1_fXX = get_rounded_base2_expansion_of_float(num1_fXX, gs.significant_bits)
        # num1_base_10_str = float_to_str(num1_fXX)
        
        return True, num1_fXX, num1_base2_str, num1_base_10_str
        
    except (OverflowError, ValueError, ZeroDivisionError):
        return False, 0.0, "", ""