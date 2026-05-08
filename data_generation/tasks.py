import math
from functools import partial
from multiprocessing.pool import Pool
from random import randint, uniform
from typing import Generator, Literal

import numpy as np

from data_generation.data_gen_utils import (
    Generation_settings,
    MockPool,
    SignificantDigitsDistribution,
    Task,
)
from data_generation.difficulty_metrics import (
    addition_difficulty_score,
    exponentiation_difficulty_score,
    mean_difficulty_metric,
    min_max_difficulty_score,
    multiplication_difficulty_score,
    sorting_difficulty_score,
)
from data_generation.utils import (
    compute_dividend_from_quotient_divisor,
    float_to_str,
    generate_divisor_for_quotient,
    generate_numbers,
    generate_quotient_with_precision,
    get_number_of_significant_digits,
    get_rounded_base2_expansion_of_float,
    round_decimal_str_to_significant_digits,
    truncated_exponential,
)


def wrapper(func, args):
    return func(*args)

computation_dtype = {
    53: np.float64,
    24: np.float32,
    11: np.float16,
}

def generate_floatXX_multiplication(exp1: int, exp2: int, gs: Generation_settings, N: int, sig_bits: int, precision: Literal[53, 24, 11]=53, allow_negative=True, mode: Task = Task.MULTIPLICATION) -> list[dict]:
    """
    Generate multiplication task data.
    Args:
        max_exp1: Maximum exponent for the first number.
        max_exp2: Maximum exponent for the second number.
        gs: Generation settings.
        N: Number of samples to generate.
        sig_bits: Number of significant bits for the numbers.
        precision: Precision for the floating point representation.
    Returns:
        List of dictionaries containing num1, num2, and their product in the specified base.
    """
    rows = []
    min_number1 = max(gs.min_number, gs.base**exp1)
    max_number1 = min(gs.max_number, gs.base ** (exp1 + 1))
    min_number2 = max(gs.min_number, gs.base**exp2)
    max_number2 = min(gs.max_number, gs.base ** (exp2 + 1))
    if gs.significant_digits_distribution == SignificantDigitsDistribution.FULL:
        sig_bits_array = np.full(N, 2*sig_bits, dtype=int).tolist()
    elif gs.significant_digits_distribution ==SignificantDigitsDistribution.BINARY_UNIFORM:
        sig_bits_array = np.random.randint(2, 2*sig_bits + 1, size=N).astype(int).tolist()
    elif gs.significant_digits_distribution == SignificantDigitsDistribution.BINARY_TRIANGULAR:
        sig_bits_array = np.random.triangular(2, 2*sig_bits, 2*sig_bits, size=N).astype(int).tolist()
    elif gs.significant_digits_distribution == SignificantDigitsDistribution.BINARY_EXPONENTIAL:
        sig_bits_array = truncated_exponential(2, 2*sig_bits, 0.06, size=N).astype(int).tolist()
    elif gs.significant_digits_distribution == SignificantDigitsDistribution.DECIMAL_UNIFORM:
        sig_bits_array = np.full(N, 2*sig_bits, dtype=int).tolist()
        sig_decimal_array = np.random.randint(2, 2*(gs.significant_digits+2), size=N).astype(int).tolist()
    else:
        raise ValueError(f"Unknown significant digits distribution: {gs.significant_digits_distribution}")
    
    for i in range(N):
        if gs.significant_digits_distribution in [SignificantDigitsDistribution.DECIMAL_UNIFORM]:
            min_s = max(1, sig_decimal_array[i]//2)
            max_s = min(sig_decimal_array[i]-1, gs.significant_digits+2)
        else:
            min_s = max(1, sig_bits_array[i]//2)
            max_s = min(sig_bits_array[i]-1, sig_bits)
        for j in range(100):
            num1 = uniform(min_number1, max_number1)
            num2 = uniform(min_number2, max_number2)
            sign_case = randint(0, 4) if allow_negative else 0 # 40% positive, 20% negative, 20% negative, 20% both negative
            if sign_case == 2:
                num1 = -num1
            elif sign_case == 3:
                num2 = -num2
            elif sign_case == 4:
                num1 = -num1
                num2 = -num2

            if gs.significant_digits_distribution in [SignificantDigitsDistribution.DECIMAL_UNIFORM]:
                s = int(np.random.triangular(min_s, max(min_s+1,sig_decimal_array[i]//2), max_s, size=1)[0]) if min_s < max_s else min_s
                num1_base_10_str = float_to_str(num1)
                num1_base_10_str = round_decimal_str_to_significant_digits(num1_base_10_str, s)
                num1 = float(num1_base_10_str)
                s2 = min(sig_decimal_array[i] - get_number_of_significant_digits(num1_base_10_str), gs.significant_digits+2)
                num2_base_10_str = float_to_str(num2)
                num2_base_10_str = round_decimal_str_to_significant_digits(num2_base_10_str, s2)
                num2 = float(num2_base_10_str)

                num1_base2_str, num1_fXX = get_rounded_base2_expansion_of_float(num1, precision)
                num2_base2_str, num2_fXX = get_rounded_base2_expansion_of_float(num2, precision)
            else:
                s = int(np.random.triangular(min_s, max(min_s+1,sig_bits_array[i]//2), max_s, size=1)[0]) if min_s < max_s else min_s
                num1_base2_str, num1_fXX = get_rounded_base2_expansion_of_float(num1, s)
                assert get_number_of_significant_digits(num1_base2_str) <= s, f"num1_base2_str={num1_base2_str} has more than {sig_bits} significant bits"
                s2 = min(sig_bits_array[i] - get_number_of_significant_digits(num1_base2_str), sig_bits)
                num2_base2_str, num2_fXX = get_rounded_base2_expansion_of_float(num2, s2)
                assert get_number_of_significant_digits(num2_base2_str) <= s2, f"num2_base2_str={num2_base2_str} has more than {s2} significant bits"
                num1_base_10_str = float_to_str(num1_fXX)
                num2_base_10_str = float_to_str(num2_fXX)

            product = (np.array(num1_fXX, dtype=computation_dtype[precision]) * np.array(num2_fXX, dtype=computation_dtype[precision])).item()
            if gs.significant_digits_distribution == SignificantDigitsDistribution.DECIMAL_UNIFORM:
                m = (int(math.floor(math.log10(abs(product)))) if product != 0 else 0)+1
                product=round(product, gs.significant_digits-m)
                product_base2_str, _ = get_rounded_base2_expansion_of_float(product, gs.significant_bits)
            else:
                product_base2_str, product = get_rounded_base2_expansion_of_float(product, gs.significant_bits)
            product_base_10_str = float_to_str(product)
            if gs.min_number <= abs(product) <= gs.max_number:
                break
            if j == 99:
                raise ValueError(f"Could not generate valid product within range after 100 tries. product={product}, num1={num1}, num2={num2}, min_number={gs.min_number}, max_number={gs.max_number}, s={s}, s2={s2}, sig_decimal_array[i]={sig_decimal_array[i] if gs.significant_digits_distribution in [SignificantDigitsDistribution.DECIMAL_UNIFORM] else 'N/A'}, sig_bits_array[i]={sig_bits_array[i] if gs.significant_digits_distribution not in [SignificantDigitsDistribution.DECIMAL_UNIFORM] else 'N/A'} ")
        
        if randint(0, 1) == 0:
            num1_base2_str, num2_base2_str = num2_base2_str, num1_base2_str
            num1_base_10_str, num2_base_10_str = num2_base_10_str, num1_base_10_str
            num1_exp, num2_exp = exp2, exp1
            num1_fXX, num2_fXX = num2_fXX, num1_fXX
        else:
            num1_exp, num2_exp = exp1, exp2
            
        if mode == Task.MULTIPLICATION:
            rows.append({
                "num1": num1_base_10_str,
                "num2": num2_base_10_str,
                "prod": product_base_10_str,
                "num1_base_2": num1_base2_str,
                "num2_base_2": num2_base2_str,
                "product_base_2": product_base2_str,
                "num1_significant_digits_base_2": get_number_of_significant_digits(num1_base2_str),
                "num2_significant_digits_base_2": get_number_of_significant_digits(num2_base2_str),
                "product_significant_digits_base_2": get_number_of_significant_digits(product_base2_str),
                "num1_exp": num1_exp,
                "num2_exp": num2_exp,
                "difficulty": min(60 if precision==53 else 32, multiplication_difficulty_score(num1_base2_str, num2_base2_str)),
                "difficulty_sd": min(32, multiplication_difficulty_score(num1_base_10_str, num2_base_10_str)),
            })
        elif mode == Task.DIVM:
            num2_reciprocal = 1. / num2_fXX
            # assert num1_fXX * num2_reciprocal == product_fXX, f"{num1_fXX} / {num2_reciprocal} = {num1_fXX/num2_reciprocal} != {product_fXX} = {num1_fXX} * {num2_fXX}"
            num2_reciprocal_base10_str = float_to_str(num2_reciprocal)
            num2_reciprocal_base2_str, _ = get_rounded_base2_expansion_of_float(num2_reciprocal, precision)
            rows.append({
                "num1": num1_base_10_str,
                "num2": num2_reciprocal_base10_str,
                "quot": product_base_10_str,
                "num1_base_2": num1_base2_str,
                "num2_base_2": num2_reciprocal_base2_str,
                "product_base_2": product_base2_str,
                "num1_significant_digits_base_2": get_number_of_significant_digits(num1_base2_str),
                "num2_significant_digits_base_2": get_number_of_significant_digits(num2_reciprocal_base2_str),
                "product_significant_digits_base_2": get_number_of_significant_digits(product_base2_str),
                "num1_exp": num1_exp,
                "num2_exp": -num2_exp,
                "difficulty": min(60 if precision==53 else 32, multiplication_difficulty_score(num1_base2_str, num2_reciprocal_base2_str)),
                # "difficulty2": min(60, multiplication_difficulty_score(num1_base2_str, num2_reciprocal_base2_str)),
                "difficulty_sd": min(32, multiplication_difficulty_score(num1_base_10_str, num2_base_10_str)),
            })

    return rows

def generate_floatXX_division(exp1: int, exp2: int, gs: Generation_settings, N: int, sig_bits: int, precision: Literal[53, 24, 11]=53, allow_negative=True) -> list[dict]:
    """
    Generate division task data with controlled quotient precision.
    
    Args:
        exp1: First number (dividend) is drawn from [gs.base**exp1, gs.base**(exp1+1) ).
        exp2: Second number (divisor) is drawn from [gs.base**exp2, gs.base**(exp2+1) ).
        gs: Generation settings.
        N: Number of samples to generate.
        sig_bits: Number of significant bits for the operands.
        precision: Precision for the floating point representation.
        allow_negative: Whether to allow negative numbers.
        
    Returns:
        List of dictionaries containing num1, num2, and their quotient with controlled precision.
    """
    rows = []
    
    # Calculate expected quotient exponent range (exp1 - exp2)
    # With exponents in [-15, 15], quotient will also be in [-15, 15]
    quotient_exp_min = exp1 - exp2 - 2
    quotient_exp_max = exp1 - exp2 + 2
    
    # Ensure quotient exponents stay within [-15, 15]
    quotient_exp_min = max(quotient_exp_min, -14)
    quotient_exp_max = min(quotient_exp_max, 14)
    
    # Set up significant bits distributions for quotient and operands
    if gs.significant_digits_distribution == SignificantDigitsDistribution.FULL:
        quotient_sig_bits_array = np.full(N, sig_bits, dtype=int).tolist()  # Use sig_bits for quotient
        operand_sig_bits_array = np.full(N, 2*sig_bits, dtype=int).tolist()
    elif gs.significant_digits_distribution == SignificantDigitsDistribution.BINARY_UNIFORM:
        quotient_sig_bits_array = np.random.randint(2, sig_bits + 1, size=N).astype(int).tolist()  # Use sig_bits for quotient
        operand_sig_bits_array = np.random.randint(2, 2*sig_bits + 1, size=N).astype(int).tolist()
    elif gs.significant_digits_distribution == SignificantDigitsDistribution.BINARY_TRIANGULAR:
        quotient_sig_bits_array = np.random.triangular(2, sig_bits, sig_bits, size=N).astype(int).tolist()  # Use sig_bits for quotient
        operand_sig_bits_array = np.random.triangular(2, 2*sig_bits, 2*sig_bits, size=N).astype(int).tolist()
    elif gs.significant_digits_distribution == SignificantDigitsDistribution.BINARY_EXPONENTIAL:
        quotient_sig_bits_array = truncated_exponential(2, sig_bits, 0.06, size=N).astype(int).tolist()  # Use sig_bits for quotient
        operand_sig_bits_array = truncated_exponential(2, 2*sig_bits, 0.06, size=N).astype(int).tolist()
    elif gs.significant_digits_distribution == SignificantDigitsDistribution.DECIMAL_UNIFORM:
        quotient_sig_bits_array = np.random.randint(2, gs.significant_digits+2, size=N).astype(int).tolist()
        operand_sig_decimal_array = np.random.randint(2, 2*(gs.significant_digits+2), size=N).astype(int).tolist()
    else:
        raise ValueError(f"Unknown significant digits distribution: {gs.significant_digits_distribution}")
    
    for i in range(N):
        # Get precision parameters for this sample
        if gs.significant_digits_distribution == SignificantDigitsDistribution.DECIMAL_UNIFORM:
            quotient_precision = quotient_sig_bits_array[i]
            min_s = max(1, operand_sig_decimal_array[i]//2)
            max_s = min(operand_sig_decimal_array[i]-1, gs.significant_digits+2)
            divisor_precision = int(np.random.triangular(min_s, max(min_s+1, operand_sig_decimal_array[i]//2), max_s, size=1)[0]) if min_s < max_s else min_s
        else:
            quotient_precision = quotient_sig_bits_array[i]
            min_s = max(1, operand_sig_bits_array[i]//2)
            max_s = min(operand_sig_bits_array[i]-1, sig_bits)
            divisor_precision = int(np.random.triangular(min_s, max(min_s+1, operand_sig_bits_array[i]//2), max_s, size=1)[0]) if min_s < max_s else min_s
        
        # Generate numbers using quotient-first approach
        for j in range(1000):
            try:
                # Step 1: Generate target quotient with controlled precision
                quotient_base2_str, quotient_fXX = generate_quotient_with_precision(
                    (quotient_exp_min, quotient_exp_max), quotient_precision, gs
                )
                
                # Step 2: Generate divisor with controlled precision
                num2_base2_str, num2_fXX, num2_base_10_str = generate_divisor_for_quotient(
                    quotient_fXX, exp2, divisor_precision, gs, precision
                )
                
                # Step 3: Compute dividend from quotient and divisor
                is_valid, num1_fXX, num1_base2_str, num1_base_10_str = compute_dividend_from_quotient_divisor(
                    quotient_fXX, num2_fXX, exp1, gs
                )
                
                if not is_valid:
                    continue
                
                # Apply sign variations
                sign_case = randint(0, 4) if allow_negative else 0
                if sign_case == 2:
                    num1_fXX = -num1_fXX
                    num1_base_10_str = float_to_str(num1_fXX)
                    num1_base2_str, _ = get_rounded_base2_expansion_of_float(num1_fXX, precision)
                elif sign_case == 3:
                    num2_fXX = -num2_fXX
                    num2_base_10_str = float_to_str(num2_fXX)
                    num2_base2_str, _ = get_rounded_base2_expansion_of_float(num2_fXX, precision)
                elif sign_case == 4:
                    num1_fXX = -num1_fXX
                    num2_fXX = -num2_fXX
                    num1_base_10_str = float_to_str(num1_fXX)
                    num2_base_10_str = float_to_str(num2_fXX)
                    num1_base2_str, _ = get_rounded_base2_expansion_of_float(num1_fXX, precision)
                    num2_base2_str, _ = get_rounded_base2_expansion_of_float(num2_fXX, precision)
                
                # Recompute quotient after sign changes
                quotient_fXX = (np.array(num1_fXX, dtype=computation_dtype[precision]) / np.array(num2_fXX, dtype=computation_dtype[precision])).item()
                if gs.significant_digits_distribution == SignificantDigitsDistribution.DECIMAL_UNIFORM:
                    m = (int(math.floor(math.log10(abs(quotient_fXX)))) if quotient_fXX != 0 else 0)+1
                    quotient_fXX=round(quotient_fXX, gs.significant_digits-m)
                    quotient_base2_str, _ = get_rounded_base2_expansion_of_float(quotient_fXX, precision)
                else:
                    quotient_base2_str, quotient_fXX = get_rounded_base2_expansion_of_float(quotient_fXX, precision)
                quotient_base_10_str = float_to_str(quotient_fXX)
                
                # Validate final quotient is within bounds [-15, 15] exponent range
                if 1e-15 <= abs(quotient_fXX) <= 1e15:
                    break
                    
            except (ZeroDivisionError, ValueError):
                continue
                
        else:
            raise ValueError(f"Could not generate valid division after 100 tries: exp1={exp1}, exp2={exp2}, quotient_precision={quotient_precision}, divisor_precision={divisor_precision}")

        # Final quotient computation with controlled precision
        # quotient_base2_str, _ = get_rounded_base2_expansion_of_float(quotient_fXX, precision)
        # quotient_base_10_str = float_to_str(quotient_fXX)
        
        # Compute reciprocal for difficulty calculation
        num2_reciprocal = 1. / num2_fXX
        num2_reciprocal_base2_str, _ = get_rounded_base2_expansion_of_float(num2_reciprocal, precision)

        rows.append({
            "num1": num1_base_10_str,
            "num2": num2_base_10_str,
            "quot": quotient_base_10_str,
            "num1_base_2": num1_base2_str,
            "num2_base_2": num2_base2_str,
            "quot_base_2": quotient_base2_str,
            "num1_significant_digits_base_2": get_number_of_significant_digits(num1_base2_str),
            "num2_significant_digits_base_2": get_number_of_significant_digits(num2_base2_str),
            "quot_significant_digits_base_2": get_number_of_significant_digits(quotient_base2_str),
            "num1_significant_digits_base_10": get_number_of_significant_digits(num1_base_10_str),
            "num2_significant_digits_base_10": get_number_of_significant_digits(num2_base_10_str),
            "quot_significant_digits_base_10": get_number_of_significant_digits(quotient_base_10_str),
            "num1_exp": exp1,
            "num2_exp": exp2,
            "quotient_precision": quotient_precision,
            "divisor_precision": divisor_precision,
            "difficulty": np.clip(multiplication_difficulty_score(num1_base2_str, num2_reciprocal_base2_str, quotient_base2_str), 30 if precision==53 else 15, 60 if precision==53 else 30),
            "difficulty_sd": multiplication_difficulty_score(num1_base_10_str, num2_base_10_str, quotient_base_10_str),
        })
    return rows

def generate_floatXX_addition(exp1: int, exp2: int, gs: Generation_settings, N: int,  precision: Literal[53, 24, 11]=53, allow_negative=True) -> list[dict]:
    """Generate addition task data.
    
    Args:
        exp1: Exponent for the first number.
        exp2: Exponent for the second number.
        gs: Generation settings.
        N: Number of samples to generate.
        precision: Precision for the floating point representation.
    Returns:
        List of dictionaries containing num1, num2, and their sum in the specified base.
    """
    rows = []
    min_number1 = max(gs.min_number, gs.base**exp1)
    max_number1 = min(gs.max_number, gs.base ** (exp1 + 1))
    min_number2 = max(gs.min_number, gs.base**exp2)
    max_number2 = min(gs.max_number, gs.base ** (exp2 + 1))
    if gs.significant_digits_distribution == SignificantDigitsDistribution.DECIMAL_UNIFORM or gs.significant_digits_distribution == SignificantDigitsDistribution.DECIMAL_UNIFORM:
        sig_decimal_array = np.random.randint(2, 2*(gs.significant_digits+2), size=N).astype(int).tolist()
    # elif gs.significant_digits_distribution != SignificantDigitsDistribution.FULL:
    #     raise NotImplementedError(f"Only {SignificantDigitsDistribution.FULL} and {SignificantDigitsDistribution.DECIMAL_UNIFORM} significant digits distribution supported for addition, got {gs.significant_digits_distribution}")
        
    for i in range(N):
        num1 = uniform(min_number1, max_number1)
        num2 = uniform(min_number2, max_number2)
        
        sign_case = randint(0, 4) if allow_negative else 0 # 40% positive, 20% negative, 20% negative, 20% both negative
        if sign_case == 2:
            num1 = -num1
        elif sign_case == 3:
            num2 = -num2
        elif sign_case == 4:
            num1 = -num1
            num2 = -num2

        if gs.significant_digits_distribution in [SignificantDigitsDistribution.DECIMAL_UNIFORM]:
            min_s = max(1, sig_decimal_array[i]//2)
            max_s = min(sig_decimal_array[i]-1, gs.significant_digits+2)
            s = int(np.random.triangular(min_s, max(min_s+1,sig_decimal_array[i]//2), max_s, size=1)[0]) if min_s < max_s else min_s
            num1_base_10_str = float_to_str(num1)
            num1_base_10_str = round_decimal_str_to_significant_digits(num1_base_10_str, s)
            num1 = float(num1_base_10_str)
            s2 = min(sig_decimal_array[i] - get_number_of_significant_digits(num1_base_10_str), gs.significant_digits+2)
            num2_base_10_str = float_to_str(num2)
            num2_base_10_str = round_decimal_str_to_significant_digits(num2_base_10_str, s2)
            num2 = float(num2_base_10_str)

        if randint(0, 1) == 0:
            num1, num2 = num2, num1
            num1_exp, num2_exp = exp2, exp1
        else:
            num1, num2 = num2, num1
            num1_exp, num2_exp = exp1, exp2

        num1_base_10_str = float_to_str(num1)
        num2_base_10_str = float_to_str(num2)

        num1_base2_str, num1_fXX = get_rounded_base2_expansion_of_float(num1, precision)
        num2_base2_str, num2_fXX = get_rounded_base2_expansion_of_float(num2, precision)

        if randint(0, 1) == 0:
            operator = '+'
            sum_fXX = (np.array(num1_fXX, computation_dtype[precision]) + np.array(num2_fXX, computation_dtype[precision])).item()
        else:
            operator = '-'
            sum_fXX = (np.array(num1_fXX, computation_dtype[precision]) - np.array(num2_fXX, computation_dtype[precision])).item()
        if gs.significant_digits_distribution == SignificantDigitsDistribution.DECIMAL_UNIFORM:
            m = (int(math.floor(math.log10(abs(sum_fXX)))) if sum_fXX != 0 else 0)+1
            sum_fXX=round(sum_fXX, gs.significant_digits-m)
            sum_base2_str, _ = get_rounded_base2_expansion_of_float(sum_fXX, precision)
        else:
            sum_base2_str, sum_fXX = get_rounded_base2_expansion_of_float(sum_fXX, precision)

        sum_base_10_str = float_to_str(sum_fXX)

        rows.append({
            "num1": num1_base_10_str,
            "num2": num2_base_10_str,
            "operator": operator,
            "sum": sum_base_10_str,
            "num1_base_2": num1_base2_str,
            "num2_base_2": num2_base2_str,
            "sum_base_2": sum_base2_str,
            "num1_significant_digits_base_2": get_number_of_significant_digits(num1_base2_str),
            "num2_significant_digits_base_2": get_number_of_significant_digits(num2_base2_str),
            "sum_significant_digits_base_2": get_number_of_significant_digits(sum_base2_str),
            "num1_significant_digits_base_10": get_number_of_significant_digits(num1_base_10_str),
            "num2_significant_digits_base_10": get_number_of_significant_digits(num2_base_10_str),
            "sum_significant_digits_base_10": get_number_of_significant_digits(sum_base_10_str),
            "num1_exp": num1_exp,
            "num2_exp": num2_exp,
            "difficulty": min(90 if precision==53 else 50, addition_difficulty_score([num1_base2_str, num2_base2_str], sum_base2_str)),
            "difficulty_sd": min(60 if precision==53 else 32, addition_difficulty_score([num1_base_10_str, num2_base_10_str], sum_base_10_str)),
        })
    return rows

def generate_floatXX_exponentiation(exp1: int, gs: Generation_settings, N: int, sig_bits: int, precision: Literal[53, 24, 11]=53, allow_negative=True) -> list[dict]:
    rows = list()
    min_n2 = math.ceil((gs.min_exponent+1)/(abs(exp1)+1))
    max_n2 = math.floor((gs.max_exponent-1)/(abs(exp1)+1))
    if min_n2 == max_n2:
        return []
    min_number1 = max(gs.min_number, gs.base**exp1)
    max_number1 = min(gs.max_number, gs.base ** (exp1 + 1))
    if gs.significant_digits_distribution == SignificantDigitsDistribution.FULL:
        sig_bits_array = np.full(N, sig_bits, dtype=int).tolist()
    elif gs.significant_digits_distribution ==SignificantDigitsDistribution.BINARY_UNIFORM:
        sig_bits_array = np.random.randint(2, sig_bits + 1, size=N).astype(int).tolist()
    elif gs.significant_digits_distribution == SignificantDigitsDistribution.DECIMAL_UNIFORM:
        sig_bits_array = np.full(N, sig_bits, dtype=int).tolist()
        sig_decimal_array = np.random.randint(2, gs.significant_digits+2, size=N).astype(int).tolist()
    else:
        raise ValueError(f"Unknown significant digits distribution: {gs.significant_digits_distribution}")
    for i in range(N):
        num1 = uniform(min_number1, max_number1)
        n_range = [i if i>1 else 1/i for i in range(min_n2, max_n2+1) if i != 0 and i!= 1]
        for num2 in n_range:
            if allow_negative and num2 > 1:
                if  randint(0, 2) == 0:
                    num1 = -num1
            if randint(0, 1) == 0:
                num2 = -num2
            
            if gs.significant_digits_distribution in [SignificantDigitsDistribution.DECIMAL_UNIFORM]:
                num1_base_10_str = float_to_str(num1)
                num1_base_10_str = round_decimal_str_to_significant_digits(num1_base_10_str, sig_decimal_array[i])
                num1 = float(num1_base_10_str)
                num2_base_10_str = float_to_str(num2)

                num1_base2_str, num1_fXX = get_rounded_base2_expansion_of_float(num1, precision)
                num2_base2_str, num2_fXX = get_rounded_base2_expansion_of_float(num2, precision)
            else:
                num1_base2_str, num1_fXX = get_rounded_base2_expansion_of_float(num1, sig_bits_array[i])
                num2_base2_str, num2_fXX = get_rounded_base2_expansion_of_float(num2, precision)
                num1_base_10_str = float_to_str(num1_fXX)
                num2_base_10_str = float_to_str(num2)

            exponentiation = (np.array(num1_fXX, dtype=computation_dtype[precision])**np.array(num2, dtype=computation_dtype[precision])).item()

            if gs.significant_digits_distribution == SignificantDigitsDistribution.DECIMAL_UNIFORM:
                m = (int(math.floor(math.log10(abs(exponentiation)))) if exponentiation != 0 else 0)+1
                exponentiation=round(exponentiation, gs.significant_digits-m)
                exponentiation_base2_str, _ = get_rounded_base2_expansion_of_float(exponentiation, gs.significant_bits)
            else:
                exponentiation_base2_str, exponentiation = get_rounded_base2_expansion_of_float(exponentiation, gs.significant_bits)
            exponentiation_base_10_str = float_to_str(exponentiation)

            if abs(num1_fXX)  == 0 or abs(num1_fXX) == 1 or abs(num2) == 0 or abs(num2) == 1 or not (gs.min_number <= abs(exponentiation) <= gs.max_number):
                continue

            rows.append({
                "num1": num1_base_10_str,
                "num2": num2_base_10_str,
                "exp": exponentiation_base_10_str,
                "num1_base_2": num1_base2_str,
                "num2_base_2": num2_base2_str,
                "exp_base_2": exponentiation_base2_str,
                "num1_significant_digits_base_2": get_number_of_significant_digits(num1_base2_str),
                "num2_significant_digits_base_2": get_number_of_significant_digits(num2_base2_str),
                "power_significant_digits_base_2": get_number_of_significant_digits(exponentiation_base2_str),
                "num1_exp": exp1,
                "num2_exp": float(abs(num2) if abs(num2)>1 else -1/abs(num2)),
                "difficulty": np.clip(exponentiation_difficulty_score(num1_base2_str, num2_base2_str, exponentiation_base2_str), 5 if precision==53 else 0, 58 if precision==53 else 23),
                "difficulty_sd": exponentiation_difficulty_score(num1_base_10_str, num2_base_10_str, exponentiation_base_10_str),
            })
    return rows

def generate_floatXX_mean(
        exp: int,
        gs: Generation_settings,
        N: int,
        mode: Task = Task.MEAN
    ) -> list[dict]:
    e_min = max(gs.min_exponent, gs.min_number_exponent)
    e_max = min(gs.max_exponent, gs.max_number_exponent)
    rows: list[dict] = list()
    if gs.significant_digits_distribution == SignificantDigitsDistribution.FULL:
        sig_bits_array = lambda *_: np.full(1, gs.significant_bits, dtype=int)  # noqa: E731
    elif gs.significant_digits_distribution ==SignificantDigitsDistribution.BINARY_UNIFORM:
        sig_bits_array = lambda spread: np.random.randint(np.clip( math.log2(10**(exp - spread + 2)), 10, gs.significant_bits), gs.significant_bits+1, size=1).astype(int)  # noqa: E731
    elif gs.significant_digits_distribution == SignificantDigitsDistribution.BINARY_TRIANGULAR:
        sig_bits_array = lambda spread: np.random.triangular(np.clip( math.log2(10**(exp - spread + 2)), 10, gs.significant_bits-1), gs.significant_bits, gs.significant_bits, size=1).astype(int)  # noqa: E731
    elif gs.significant_digits_distribution == SignificantDigitsDistribution.BINARY_EXPONENTIAL:
        sig_bits_array = lambda spread: truncated_exponential(np.clip( math.log2(10**(exp - spread + 2)), 10, gs.significant_bits-1), gs.significant_bits, 0.06, size=1).astype(int)  # noqa: E731
    elif gs.significant_digits_distribution == SignificantDigitsDistribution.DECIMAL_UNIFORM:
        sig_bits_array = lambda *_: np.full(1, gs.significant_bits, dtype=int)  # noqa: E731
        sig_decimal_array = lambda spread: np.random.randint(np.clip(exp - spread + 2, 1, gs.significant_digits+2), gs.significant_digits+3, size=1).astype(int)  # noqa: E731
    else:
        raise ValueError(f"Unknown significant digits distribution: {gs.significant_digits_distribution}")
    min_mean = max(gs.min_number, gs.base**exp)
    max_mean = min(gs.max_number, gs.base ** (exp + 1))
    for list_len in [2, 3, 4, 5]:
        for i in range(N):
            if mode in [Task.STD, Task.MEAN]:
                spread_range = range(max(e_min+2,exp-gs.significant_digits), min(exp+(gs.significant_digits+2), e_max-2))
            else:
                spread_range = range(max(e_min+2,exp-gs.significant_digits), min(exp+2, e_max-2))
            for spread in spread_range:
                skip=False
                if gs.significant_digits_distribution in [SignificantDigitsDistribution.DECIMAL_UNIFORM]:
                    precision=gs.significant_bits
                    precision_base10 = sig_decimal_array(spread).item()
                    max_num_significant_bits=None
                    max_num_significant_digits=precision_base10
                else:
                    precision: int = sig_bits_array(spread).item()
                    precision_base10 = math.log10(2.**precision)
                    max_num_significant_bits=precision
                    max_num_significant_digits=None
                list1: list[float] = list()
                for j in range(20):
                    mean_value = uniform(min_mean, max_mean)
                    try:
                        list1 = generate_numbers(
                            mean_value,
                            gs.base**spread,
                            list_len,
                            max_num_significant_bits=max_num_significant_bits,
                            max_num_significant_digits=max_num_significant_digits
                        )
                    except (AssertionError, ValueError):
                        if j>=19:
                            skip=True
                            continue
                        else:
                            continue
                    list1_base2_str = []
                    list1_base10_str = []
                    list1_values = []
                    for n in list1:
                        n_base2_str, n_fXX = get_rounded_base2_expansion_of_float(n, precision)
                        list1_base2_str.append(n_base2_str)
                        list1_base10_str.append(float_to_str(n_fXX))
                        list1_values.append(n_fXX)
                    mean_value_true = np.array(list1_values, dtype=computation_dtype[gs.significant_bits]).mean().item()
                    if gs.significant_digits_distribution == SignificantDigitsDistribution.DECIMAL_UNIFORM:
                        m = (int(math.floor(math.log10(abs(mean_value_true)))) if mean_value_true != 0 else 0)+1
                        mean_value_true=round(mean_value_true, gs.significant_digits-m)
                        mean_true_str_base2, _ = get_rounded_base2_expansion_of_float(mean_value_true, gs.significant_bits)
                    else:
                        mean_true_str_base2, mean_value_true = get_rounded_base2_expansion_of_float(mean_value_true, precision)

                    std = np.array(list1_values, dtype=computation_dtype[gs.significant_bits]).std().item()
                    if (
                        all([(gs.min_number <= abs(f) < gs.max_number) or f==0 for f in list1_values])
                        and (len(list1_values)%2 != 0 or mean_value_true != 0)
                        and (gs.min_number <= std < gs.max_number)
                        and not (mode == Task.INTERVAL and len(set(list1_base10_str)) < len(list1_base10_str))
                    ):
                        break
                    if j>=19:
                        skip=True
                        # raise Exception(f"Too many tries {j}: mean1={mean_value}, spread={spread}, precision={precision}, precision_base10={precision_base10}, list_len={list_len}, list1={list1}")
                if skip:
                    continue
                mean_true_str_base10 = float_to_str(mean_value_true)
                if mode == Task.STD:
                    assert gs.min_number <= std < gs.max_number, f"std={std} not in [{gs.min_number}, {gs.max_number}]"
                    if gs.significant_digits_distribution == SignificantDigitsDistribution.DECIMAL_UNIFORM:
                        m = (int(math.floor(math.log10(abs(std)))) if std != 0 else 0)+1
                        std=round(std, gs.significant_digits-m)
                        std_true_str_base2, _ = get_rounded_base2_expansion_of_float(std, gs.significant_bits)
                    else:
                        std_true_str_base2, std = get_rounded_base2_expansion_of_float(std, precision)
                    std_base_10_str = float_to_str(std)
                    record = {
                        "list1": str(list1_base10_str),
                        # "list1_base_2": str(list1_base2_str),
                        "std": std_base_10_str,
                        "std_base_2": std_true_str_base2,
                        "list_len": list_len,
                        "exp": exp,
                        "spread": spread,
                        "difficulty": min(155 if precision==53 else 95, mean_difficulty_metric(list1_base2_str, base=2)), # spread-exp+15,
                        "difficulty_sd": min(80, mean_difficulty_metric(list1_base10_str, base=10)), # spread-exp+15,
                        "max_num_significant_bits": max_num_significant_bits,
                        "max_num_significant_digits": max_num_significant_digits,
                    }
                elif mode == Task.MEAN:
                    record = {
                        "list1": str(list1_base10_str),
                        # "list1_base_2": str(list1_base2_str),
                        "mean": mean_true_str_base10,
                        "mean_base_2": mean_true_str_base2,
                        "list_len": list_len,
                        "exp": exp,
                        "spread": spread,
                        "difficulty": min(155 if precision==53 else 95, mean_difficulty_metric(list1_base2_str, base=2)), # spread-exp+15,
                        "difficulty_sd": min(80, mean_difficulty_metric(list1_base10_str, base=10)), # spread-exp+15,
                        "max_num_significant_bits": max_num_significant_bits,
                        "max_num_significant_digits": max_num_significant_digits,
                    }
                elif mode == Task.MIN_MAX:
                    record = {
                        "list1": str(list1_base10_str),
                        # "list1_base_2": str(list1_base2_str),
                        "minimum": list1_base10_str[np.argmin(list1)],
                        "min_base_2": list1_base2_str[np.argmin(list1)],
                        "maximum": list1_base10_str[np.argmax(list1)],
                        "max_base_2": list1_base2_str[np.argmax(list1)],
                        "list_len": list_len,
                        "exp": exp,
                        "spread": spread,
                        "max_num_significant_bits": max_num_significant_bits,
                        "max_num_significant_digits": max_num_significant_digits,
                        "minimum_difficulty": min_max_difficulty_score(list1_base2_str, list1_base2_str[np.argmin(list1)]),
                        "maximum_difficulty": min_max_difficulty_score(list1_base2_str,list1_base2_str[np.argmax(list1)]),
                        "minimum_difficulty_sd": min_max_difficulty_score(list1_base10_str, list1_base10_str[np.argmin(list1)]),
                        "maximum_difficulty_sd": min_max_difficulty_score(list1_base10_str,list1_base10_str[np.argmax(list1)]),
                    }
                elif mode == Task.INTERVAL:
                    sorted_l1 = sorted(list1_values)
                    sorted_l1_base10_str = [list1_base10_str[i] for i in np.argsort(list1_values)]
                    sorted_l1_base2_str = [list1_base2_str[i] for i in np.argsort(list1_values)]
                    
                    found=False
                    available_positions = list(range(len(sorted_l1)+1))
                    np.random.shuffle(available_positions)
                    for position in available_positions:
                        # Choose a random index for the interval
                        position = randint(0, len(sorted_l1))

                        skip=False
                        for j in range(20):
                            # Generate a random number between the two selected values
                            lower_bound = sorted_l1[position-1] if position > 0 else sorted_l1[position] - gs.base**spread/list_len
                            upper_bound = sorted_l1[position] if position < len(sorted_l1) else sorted_l1[position-1] + gs.base**spread/list_len
                            
                            # Generate random value in the interval with same precision properties
                            random_value = uniform(lower_bound, upper_bound)
                            
                            # Apply the same precision constraints as the original numbers
                            if gs.significant_digits_distribution in [SignificantDigitsDistribution.DECIMAL_UNIFORM]:
                                random_base_10_str = float_to_str(random_value)
                                random_base_10_str = round_decimal_str_to_significant_digits(random_base_10_str, int(precision_base10))
                                random_fXX = float(random_base_10_str)
                                # Convert to base2 representation with same precision
                                random_base2_str, _ = get_rounded_base2_expansion_of_float(random_value, precision)
                            else:
                                random_base2_str, random_fXX = get_rounded_base2_expansion_of_float(random_value, precision)
                                random_base_10_str = float_to_str(random_fXX)
                            
                            # Validate the generated number is within bounds
                            if (gs.min_number <= abs(random_fXX) < gs.max_number or random_fXX == 0) and (lower_bound <= random_fXX < upper_bound):
                                found=True
                                break
                        if found:
                            break
                    if not found:
                        continue
                        
                    record = {
                        "list1": str(sorted_l1_base10_str),
                        "ref": random_base_10_str,
                        "position": position,
                        # "list1_base_2": str(sorted_l1_base2_str),
                        "ref_base_2": random_base2_str,
                        "list_len": list_len,
                        "exp": exp,
                        "spread": spread,
                        "max_num_significant_bits": max_num_significant_bits,
                        "max_num_significant_digits": max_num_significant_digits,
                        "difficulty": min_max_difficulty_score(sorted_l1_base2_str, random_base2_str),
                        "difficulty_sd": min_max_difficulty_score(sorted_l1_base10_str, random_base_10_str),
                    }
                elif mode == Task.SORTING:
                    permuted_indices = np.random.permutation(len(list1_values))
                    list1_values = [list1_values[i] for i in permuted_indices]
                    list1_base10_str = [list1_base10_str[i] for i in permuted_indices]
                    list1_base2_str = [list1_base2_str[i] for i in permuted_indices]

                    sorted_l1 = sorted(list1_values)
                    asc_base10_str = [list1_base10_str[i] for i in np.argsort(list1_values)]
                    # asc_base2_str = [list1_base2_str[i] for i in np.argsort(list1_values)]
                    desc_base10_str = asc_base10_str[::-1]
                    # desc_base2_str = asc_base2_str[::-1]
                    record = {
                        "list1": str(list1_base10_str),
                        "asc": str(asc_base10_str),
                        "desc": str(desc_base10_str),
                        # "list1_base_2": str(list1_base2_str),
                        # "asc_base_2": str(asc_base2_str),
                        # "desc_base_2": str(desc_base2_str),
                        "list_len": list_len,
                        "exp": exp,
                        "spread": spread,
                        "max_num_significant_bits": max_num_significant_bits,
                        "max_num_significant_digits": max_num_significant_digits,
                        "difficulty": sorting_difficulty_score(list1_base2_str),
                        "difficulty_sd": sorting_difficulty_score(list1_base10_str),
                    }
                else:
                    raise NotImplementedError(f"Mode {mode} not implemented")
                rows.append(record)
    return rows


def generate_dataset(task: Task, split: str, N: int, gs: Generation_settings, pool: Pool|MockPool) -> Generator[dict, None, None]:
    all_combinations = []
    match task:
        case Task.MULTIPLICATION | Task.DIVM:
            # 0 if quantization = min_exponent. If quantization> min_exponent, then start at difference (positive)
            exp1_start = max(gs.min_exponent, gs.min_number_exponent)
            exp1_end = min(gs.max_exponent, gs.max_number_exponent)
            for exp1 in range(exp1_start, exp1_end):
                if (gs.base ** (exp1 + 1) < gs.min_number or gs.base**exp1 >= gs.max_number):
                    continue
                for exp2 in range(exp1_start, exp1_end):
                    if gs.base**exp2 < gs.min_number or gs.base**exp2 >= gs.max_number:
                        continue
                    if exp1 + exp2 + 2 <= gs.min_exponent or exp1 + exp2 >= gs.max_exponent:
                        continue
                    all_combinations.append((exp1, exp2, gs))
            task_func = partial(wrapper, partial(generate_floatXX_multiplication, sig_bits=gs.significant_bits, precision=gs.significant_bits, mode=task))
        case Task.ADDITION:
            # 0 if quantization = min_exponent. If quantization> min_exponent, then start at difference (positive)
            exp1_start = max(gs.min_exponent, gs.min_number_exponent)
            exp1_end = min(gs.max_exponent, gs.max_number_exponent)
            
            # Create weighted combinations based on exponent similarity
            weighted_combinations = []
            for exp1 in range(exp1_start, exp1_end):
                if (gs.base ** (exp1 + 1) < gs.min_number or gs.base**exp1 >= gs.max_number):
                    continue
                for exp2 in range(exp1_start, exp1_end):
                    if gs.base**exp2 < gs.min_number or gs.base**exp2 >= gs.max_number:
                        continue
                    # Weight based on exponent similarity - closer exponents get higher weight
                    exp_diff = abs(exp1 - exp2)
                    weight = math.exp(-exp_diff * 0.15)  # Exponential decay with similarity (less extreme)
                    weighted_combinations.append(((exp1, exp2, gs), weight))
            
            # Normalize weights and distribute samples
            total_weight = sum(weight for _, weight in weighted_combinations)
            for (exp1, exp2, gs_item), weight in weighted_combinations:
                # Calculate number of samples for this combination based on weight
                combo_samples = max(1, int(N * weight / total_weight))
                all_combinations.append((exp1, exp2, gs_item, combo_samples))
            
            task_func = partial(wrapper, partial(generate_floatXX_addition, precision=gs.significant_bits))
        case Task.DIVISION:
            # 0 if quantization = min_exponent. If quantization> min_exponent, then start at difference (positive)
            exp1_start = max(gs.min_exponent, gs.min_number_exponent)
            exp1_end = min(gs.max_exponent, gs.max_number_exponent)
            for exp1 in range(exp1_start, exp1_end):
                if (gs.base ** (exp1 + 1) < gs.min_number or gs.base**exp1 >= gs.max_number):
                    continue
                for exp2 in range(exp1_start, exp1_end):
                    if gs.base**exp2 < gs.min_number or gs.base**exp2 >= gs.max_number:
                        continue
                    # For division, the result exponent is exp1 - exp2, so we need different bounds
                    if exp1 - exp2 < gs.min_exponent or exp1 - exp2 >= gs.max_exponent:
                        continue
                    all_combinations.append((exp1, exp2, gs))
            task_func = partial(wrapper, partial(generate_floatXX_division, sig_bits=gs.significant_bits, precision=gs.significant_bits))
        case Task.EXPONENTIATION:
            exp_start = max(gs.min_exponent, gs.min_number_exponent)
            exp_end = min(gs.max_exponent, gs.max_number_exponent)
            weighted_combinations = []
            for exp in range(exp_start, exp_end):
                if (gs.base ** (exp + 1) < gs.min_number or gs.base**exp >= gs.max_number):
                    continue
                # Weight based on exponent proximity to 0 - closer exponents get higher weight
                exp_diff = abs(-exp)
                weight = math.exp(-exp_diff * 0.15)  # Exponential decay with similarity (less extreme)
                weighted_combinations.append(((exp, gs), weight))
            # Normalize weights and distribute samples
            total_weight = sum(weight for _, weight in weighted_combinations)
            for (exp1, gs_item), weight in weighted_combinations:
                # Calculate number of samples for this combination based on weight
                combo_samples = max(1, int(N * weight / total_weight))
                all_combinations.append((exp1, gs_item, combo_samples))
            task_func = partial(wrapper, partial(generate_floatXX_exponentiation, sig_bits=gs.significant_bits, precision=gs.significant_bits))
        case Task.MEAN | Task.STD | Task.MIN_MAX | Task.SORTING | Task.INTERVAL:
            e_min = max(gs.min_exponent, gs.min_number_exponent)
            e_max = min(gs.max_exponent, gs.max_number_exponent)
            for exp in range(e_min+2,e_max-2):
                 all_combinations.append((exp, gs))
            task_func = partial(wrapper, partial(generate_floatXX_mean, mode=task))
        case _:
            raise NotImplementedError(f"Task {task} not implemented")

    # Handle different combination formats - Addition has pre-calculated sample counts
    if task in [Task.ADDITION, Task.EXPONENTIATION]:
        task_args = [(*args[:-1], args[-1]) for args in all_combinations]  # args already includes sample count
    else:
        n = math.ceil(N / len(all_combinations))
        task_args = [(*args, n) for args in all_combinations]
    
    yield len(task_args)
    for result in pool.imap_unordered(task_func, task_args):
        yield result
