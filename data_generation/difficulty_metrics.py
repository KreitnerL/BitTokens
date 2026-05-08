import math
from collections import Counter

from data_generation.utils import (
    base_string_to_float,
    float_to_str,
    get_rounded_base2_expansion_of_float,
)


def _difficulty_to_int(number_str: str) -> int:
    """
    Count the number of non zero digits in the number string
    Args:
        number_str (str): Number string in any base
    Returns:
        int: Number of non zero digits
    """
    return max(1,sum(1 for c in number_str.upper() if (c.isdigit() and c != '0') or c.isalpha()))

def multiplication_difficulty_score(*numbers: str) -> int:
    """
    Estimate the difficulty of multiplying fixed-point numbers (base-2 or base-10) with a
    heuristic that considers the number of non-zero digits in each operand.
    Args:
        numbers (list[str]): List of fixed-point number strings to be multiplied.
    Returns:
        int: Difficulty score based on the sum of non-zero digits in all operands.
    """
    score = sum(_difficulty_to_int(num) for num in numbers)
    return score

def division_difficulty_score(dividend: str, divisor: str, quotient: str) -> float:
    """
    Calculate division difficulty based on actual computational work.
    
    Optimized to focus on meaningful digits rather than leading zeros or common prefixes.
    
    Args:
        dividend: The number being divided (as string)
        divisor: The number dividing by (as string) 
        quotient: The result of division (as string)
        
    Returns:
        float: Difficulty score based on computational complexity
    """
    try:
        divisor_val = float(divisor)
        if divisor_val == 0:
            return 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0
    
    # Remove decimal points and signs for digit counting
    clean_dividend = dividend.replace('.', '').replace('-', '')
    
    # Count only meaningful (non-zero) digits in dividend
    meaningful_digits = len([d for d in clean_dividend if d != '0'])
    if meaningful_digits == 0:
        meaningful_digits = 1  # At least some minimal work
    
    # Check if divisor is a "nice" number (power of 2 or 10)
    divisor_val = abs(float(divisor))
    is_power_of_10 = divisor_val in [0.1, 0.01, 0.001, 1, 10, 100, 1000, 10000]
    
    # Check power of 2 for integers only
    is_power_of_2 = False
    if divisor_val.is_integer() and divisor_val > 0:
        int_val = int(divisor_val)
        is_power_of_2 = (int_val & (int_val - 1)) == 0
    
    # Base division difficulty - nice divisors are easier
    if is_power_of_10:
        divisor_complexity = 1  # Just shifting decimal point
    elif is_power_of_2:
        divisor_complexity = 2  # Bit shifts, still relatively easy
    else:
        divisor_complexity = 4  # Requires actual long division
    
    # Decimal handling penalty - but only if it adds real complexity
    decimal_penalty = 0
    if '.' in dividend or '.' in quotient:
        # Only add penalty if decimals create meaningful complexity
        if meaningful_digits > 2:  # Non-trivial decimal work
            decimal_penalty = 2
        else:
            decimal_penalty = 1  # Minimal decimal handling
    
    return meaningful_digits + divisor_complexity + decimal_penalty


def _normalize_numbers(numbers: list[str], align_integers: bool = False) -> list[str]:
    """
    Normalize numbers to same decimal places and optionally same integer width for proper alignment.
    
    Args:
        numbers: List of number strings to normalize
        align_integers: If True, pad integer parts to same width for decimal point alignment
        
    Returns:
        List of normalized number strings
    """
    if not numbers:
        return []
    
    # Find maximum decimal places and integer part length
    max_decimal_places = 0
    max_integer_length = 0
    
    for num in numbers:
        if '.' in num:
            integer_part, decimal_part = num.split('.')
            decimal_places = len(decimal_part)
            max_decimal_places = max(max_decimal_places, decimal_places)
            max_integer_length = max(max_integer_length, len(integer_part))
        else:
            max_integer_length = max(max_integer_length, len(num))
    
    # Normalize all numbers to same decimal places and optionally integer width
    normalized = []
    for num in numbers:
        if '.' in num:
            integer_part, decimal_part = num.split('.')
            # Optionally pad integer part to max length (left pad with zeros)
            if align_integers:
                integer_part = integer_part.zfill(max_integer_length)
            # Pad decimal part to max length (right pad with zeros)
            decimal_part = decimal_part.ljust(max_decimal_places, '0')
            normalized_num = integer_part + '.' + decimal_part
        else:
            # Optionally pad integer part and add decimal point with zeros if needed
            integer_part = num.zfill(max_integer_length) if align_integers else num
            if max_decimal_places > 0:
                normalized_num = integer_part + '.' + '0' * max_decimal_places
            else:
                normalized_num = integer_part
        normalized.append(normalized_num)
    
    return normalized


def addition_difficulty_score(operands: list[str], sum_result: str) -> int:
    """
    Calculate addition difficulty based on actual computational work.
    1. Align all operands to same decimal places
    2. Identify the largest operand
    3. Count digit changes between largest operand and sum
    4. Count carry operations - positions where multiple operands have non-zero digits
    Args:
        operands: List of number strings being added
        sum_result: The result of the addition as a string
    Returns:
        score (int): Difficulty score based on computational complexity
    """
    # Find the largest operand by magnitude
    max_magnitude = max(abs(float(op)) for op in operands)
    largest_operand = next(op for op in operands if abs(float(op)) == max_magnitude)
    
    # Remove decimal points for digit-by-digit comparison
    clean_operands = [op.replace('.', '') for op in operands]
    clean_sum = sum_result.replace('.', '')
    clean_largest = largest_operand.replace('.', '')
    
    # Count digit changes between largest operand and sum
    digit_changes = sum(1 for i in range(min(len(clean_largest), len(clean_sum))) 
                       if clean_largest[i] != clean_sum[i])
    
    # Count carry operations - positions where multiple operands have non-zero digits
    max_len = max(len(clean) for clean in clean_operands)
    carry_operations = 0
    
    for i in range(max_len):
        non_zero_count = sum(1 for clean in clean_operands 
                           if i < len(clean) and clean[i] != '0')
        if non_zero_count >= 2:
            carry_operations += 1
    
    # Carry operations are more computationally expensive than simple digit changes
    return int(round(digit_changes + (carry_operations * 3)))

def recursive_addition_difficulty_score(operands: list[str], base=10, precision=52) -> tuple[int, str]:
    """
    Calculate addition difficulty based on actual computational work using a recursive pairwise approach.
    1. Align all operands to same decimal places
    2. Recursively compute addition difficulty of pairs until one sum remains
    Args:
        operands: List of number strings being added
    Returns:
        score (int): Difficulty score based on computational complexity
    """
    if len(operands) == 0:
        raise ValueError("Operands list cannot be empty")
    if len(operands) == 1:
        return 0, operands[0]
    if len(operands) > 2:
        mid = len(operands)//2
        left_score, left_sum = recursive_addition_difficulty_score(operands[:mid], base, precision)
        right_score, right_sum = recursive_addition_difficulty_score(operands[mid:], base, precision)
        operands = [left_sum, right_sum]
    if base == 10:
        o1 = float(operands[0])
        o2 = float(operands[1])
        pair_sum = o1 + o2
        pair_sum_str = float_to_str(pair_sum)
    elif base == 2:
        o1 = base_string_to_float(operands[0], 2)
        o2 = base_string_to_float(operands[1], 2)
        pair_sum = o1 + o2
        pair_sum_str, _ = get_rounded_base2_expansion_of_float(pair_sum, precision)
    return addition_difficulty_score(operands, pair_sum_str), pair_sum_str


def _find_aligned_prefix_length(str1: str, str2: str) -> int:
    """Find the length of common prefix between two decimal-aligned strings."""
    min_len = min(len(str1), len(str2))
    for i in range(min_len):
        if str1[i] != str2[i]:
            return i
    return min_len

def min_max_difficulty_score(operands: list[str], min_max_result: str) -> int:
    """
    Calculate difficulty of finding min or max in a list of numbers.
    
    The score measures the longest common prefix with the min/max result,
    and the number of operands that share that prefix, as these represent
    the comparisons that must be made.
    
    Args:
        operands: List of fixed-point number strings to find min/max of
        min_max_result: The calculated min or max as a string
        
    Returns:
        int: Difficulty score based on number of comparisons
    """
    if len(operands) <= 1:
        return 0
    
    # Remove instances of the result from operands to avoid counting them
    operands_without_result = [op for op in operands if op != min_max_result]
    
    if not operands_without_result:
        return 0  # All operands are the same as the result
    
    # Normalize all numbers including the result for proper decimal alignment
    # Use align_integers=True for min/max comparison to ensure proper decimal point alignment
    all_numbers = operands_without_result + [min_max_result]
    normalized_numbers = _normalize_numbers(all_numbers, align_integers=True)
    
    # Extract normalized operands and result
    normalized_operands = normalized_numbers[:-1]
    normalized_result = normalized_numbers[-1]
    
    # Calculate prefix length for each normalized operand with the normalized result
    prefix_lengths = []
    for normalized_operand in normalized_operands:
        prefix_len = _find_aligned_prefix_length(normalized_operand, normalized_result)
        prefix_lengths.append(prefix_len)
    
    # Group operands by their common prefix length and calculate weighted score
    prefix_counts = Counter(prefix_lengths)
    
    # Calculate difficulty score: sum of (count × prefix_length) for each group
    difficulty_score = 0
    for prefix_length, count in prefix_counts.items():
        difficulty_score += count * prefix_length
    
    return math.ceil(difficulty_score**.8) if difficulty_score > 0 else 0

def sorting_difficulty_score(operands: list[str]) -> int:
    """
    Calculate difficulty of sorting a list of numbers.
    
    The score measures the comparison work needed to sort the numbers.
    Numbers with longer common prefixes are harder to distinguish and
    contribute more to the sorting difficulty.
    
    Args:
        operands: List of fixed-point number strings to sort
        
    Returns:
        int: Difficulty score based on pairwise comparison complexity
    """
    if len(operands) <= 1:
        return 0
    
    # Normalize all numbers for proper decimal alignment
    # Use align_integers=True to ensure proper decimal point alignment
    normalized_numbers = _normalize_numbers(operands, align_integers=True)
    
    # Calculate total comparison difficulty by examining all pairs
    total_difficulty = 0
    n = len(normalized_numbers)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Find common prefix length between each pair
            prefix_len = _find_aligned_prefix_length(normalized_numbers[i], normalized_numbers[j])
            # The longer the common prefix, the more work needed to distinguish
            total_difficulty += prefix_len
    
    return math.ceil(total_difficulty**.8) if total_difficulty > 0 else 0

def exponentiation_difficulty_score(base: str, exponent: str, result: str, dampening: float=0.4) -> int:
    """
    Estimate the difficulty of computing base^exponent with a heuristic that
    multiplies the complexities (number of non-zero digits) of base, exponent, and result
    and applies an exponential damping factor to moderate the score.
    
    Args:
        base (str): Base number as string
        exponent (str): Exponent as string (can be fractional)
        result (str): The computed result as string
        dampening (float): Exponential dampening factor to moderate score growth
        
    Returns:
        int: Difficulty score based on computational complexity
    """
    base_complexity = _difficulty_to_int(base)
    exponent_complexity = _difficulty_to_int(exponent)
    result_complexity = _difficulty_to_int(result)
    
    # Combine all factors
    total_score = base_complexity * exponent_complexity * result_complexity
    return round(total_score**dampening)

def _find_common_prefix(operands: list[str], preserve_magnitude: bool = False) -> tuple[str, list[str]]:
    """
    Find the common prefix across all operands and return the meaningful suffixes.
    
    Args:
        operands: List of number strings
        preserve_magnitude: If True, preserve the magnitude of suffixes for floating point numbers
        
    Returns:
        tuple: (common_prefix, list_of_suffixes)
    """
    if len(operands) <= 1:
        return "", operands
    
    # Normalize numbers to same decimal places (but don't align integers for this use case)
    normalized_operands = _normalize_numbers(operands, align_integers=False)
    
    # Find common prefix character by character
    if not normalized_operands:
        return "", []
    
    min_length = min(len(op) for op in normalized_operands)
    common_prefix = ""
    
    for i in range(min_length):
        char = normalized_operands[0][i]
        if all(op[i] == char for op in normalized_operands):
            common_prefix += char
        else:
            break
    
    # Extract meaningful suffixes
    suffixes = []
    for op in normalized_operands:
        suffix = op[len(common_prefix):]
        
        if preserve_magnitude and '.' in common_prefix:
            # For floating point numbers, preserve magnitude by calculating the decimal position
            # Count digits after decimal point in common prefix
            decimal_pos_in_prefix = common_prefix.find('.')
            if decimal_pos_in_prefix != -1:
                digits_after_decimal_in_prefix = len(common_prefix) - decimal_pos_in_prefix - 1
                
                # Create suffix with proper magnitude
                if suffix:
                    # Remove leading zeros but preserve the decimal positioning
                    suffix_clean = suffix.lstrip('0') or '0'
                    # Calculate the magnitude: 10^(-digits_after_decimal_in_prefix - leading_zeros_removed)
                    leading_zeros_removed = len(suffix) - len(suffix_clean)
                    
                    # Format as decimal with proper magnitude
                    if suffix_clean == '0':
                        magnitude_suffix = '0'
                    else:
                        magnitude_suffix = f"0.{'0' * (digits_after_decimal_in_prefix + leading_zeros_removed)}{suffix_clean}"
                else:
                    magnitude_suffix = '0'
                
                suffixes.append(magnitude_suffix)
            else:
                # No decimal in prefix, treat as regular suffix
                suffix = suffix.lstrip('0') or '0'
                suffixes.append(suffix)
        else:
            # Remove leading zeros but keep at least one digit
            suffix = suffix.lstrip('0') or '0'
            suffixes.append(suffix)
    
    return common_prefix, suffixes

def mean_difficulty_metric(operands: list[str], base=10):
    common_prefix, suffixes = _find_common_prefix(operands, preserve_magnitude=False)
    list_len_reciprocal = 1 / len(operands)
    list_len_reciprocal_base_string = str(list_len_reciprocal) if base == 10 else get_rounded_base2_expansion_of_float(list_len_reciprocal, 54)[0]
    sum_significant_digits = sum(_difficulty_to_int(suffix) for suffix in suffixes + [list_len_reciprocal_base_string])
    return sum_significant_digits
