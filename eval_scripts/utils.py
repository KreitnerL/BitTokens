import json
import math
import re

AI_MESSAGE_PREFIX = """{"answer": """

AI_MESSAGE = """{"answer": {answer}}"""

NUMERIC_SYSTEM_MESSAGE = """You are an expert in numeracy. Return exactly one valid JSON object in this format: 
{"answer": <numeric_answer>}
Do not explain, show steps, or add any extra text. Do not use code blocks to output the answer.
DO NOT CALL ANY external APIs or use ANY external tool to solve the problem. DO NOT USE a calculator tool. DO NOT USE python. DO NOT USE Wolfram Alpha.
If the answer is not an integer, give it as a decimal (not a fraction), rounded to at most 15 significant digits."""

INTERVAL_SYSTEM_MESSAGE = """You are an expert in numeracy. For each problem, output only valid JSON in this format: 
{"answer": <interval_multiple_choice_answer>}
Do not explain, show steps, or add any extra text. Do not use code blocks to output the answer.
DO NOT CALL ANY external APIs or use ANY external tool to solve the problem. DO NOT USE a calculator tool. DO NOT USE python. DO NOT USE Wolfram Alpha.
The answer must be one of the following: A, B, C, D, E, F."""

SORTING_SYSTEM_MESSAGE = """You are an expert in numeracy. For each problem, output only valid JSON in this format: 
{"answer": <sorted_list>}
Do not explain, show steps, or add any extra text. Do not use code blocks to output the answer.
DO NOT CALL ANY external APIs or use ANY external tool to solve the problem. DO NOT USE a calculator tool. DO NOT USE python. DO NOT USE Wolfram Alpha.
The answer must be a list of numbers."""

MIN_MAX_SYSTEM_MESSAGE = """You are an expert in numeracy. For each problem, output only valid JSON in this format: 
{"answer": <numeric_answer>}
Do not explain, show steps, or add any extra text. Do not use code blocks to output the answer.
DO NOT CALL ANY external APIs or use ANY external tool to solve the problem. DO NOT USE a calculator tool. DO NOT USE python. DO NOT USE Wolfram Alpha.
The answer must be a single number, exactly as it appears in the list."""

def logSMAPE(
        num_pred: float,
        num_true: float,
        max_num_digits=15,
        eps: float=1e-100
    ) -> float:
    r"""Calculate log Symmetric Mean Absolute Percentage Error (sMAPE).
    .. math::
        \min\left(1, \frac{\log_{10}\left(\frac{|A_j - P_j|}{|A_j| + |P_j| + \epsilon} + \epsilon\right)}{-\min\left(1, \max\left(D, \left\lfloor D + \log_{10}(|A_j| + \epsilon)\right\rfloor\right)\right)}\right)
    where :math:`A_j` is the true value, :math:`P_j` is the predicted value, :math:`D` is the maximum number of digits to consider, and :math:`\epsilon` is a small value to prevent division by zero.

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.
    Args:
    -----
        num_pred (float): The predicted values
        num_true (float): The true values
        max_num_digits (int): The maximum number of digits to consider
        eps (float): A small value to prevent division by zero
    Returns:
    -------
         accuracy (float): The log sMAPE metric between 0 and 1 (lower is better)
    """
    # Check if num_pred is inf or nan
    if math.isinf(num_pred) or math.isnan(num_pred):
        return 0
    sMAPE = (abs(num_pred - num_true) / (abs(num_true) + abs(num_pred) + eps))
    # num_sig_digits = max(1,min(max_num_digits, int(max_num_digits + math.log10(abs(num_true)+eps))))
    log_sMAPE = min(1, math.log10(sMAPE + eps) / -max_num_digits)
    return log_sMAPE

def parse_numbers_from_text(text: str) -> list[float]:
    """
    Parse numbers from a text.
    Args:
        text (str): The text to parse numbers from
    Returns:
        list[float]: The list of numbers
    """
    numbers = []
    
    for match in re.finditer(r"(?:^| |)[+-]?(?:(?:0(?!\.[0-9]))|(?:[0-9]*[.][0-9]+)|(?:[1-9][0-9]*))(?:[eE][+-]?[0-9]+)?", text):
        num_str = match.group().strip()
        # Process the number string to handle precision
        extracted_num = float(num_str)
        numbers.append(extracted_num)
    return numbers

def eval_regression(response: str, true_answer: float) -> tuple[float, float]:
    """
    Evaluate the regression response.
    Args:
        response (str): The response from the model
        true_answer (float): The true answer
    Returns:
        bool: True if the response is correct, False otherwise
        float: The sMAPE accuracy
        float: The predicted number
    """
    # Parse numbers from the response
    pred_numbers = parse_numbers_from_text(response)
    if len(pred_numbers) == 0:
        return 0.0, 0.0
    pred_number = pred_numbers[0]
    logSMAPE_acc = logSMAPE(num_pred=pred_number, num_true=true_answer)
    # Check if the true answer is in the parsed numbers
    return logSMAPE_acc, pred_number

def parse_response(response: str, true_answer) -> bool:
    # response = response.lower()
    # true_answer = true_answer.lower()
    return true_answer in response

def parse_answer(response: str, keep_answer_raw: bool=False) -> str:
    """
    Extract an answer from a model response robustly.

    Tries to parse JSON and read the "answer" field. If that fails, falls back to
    extracting the first number from the text. Returns an empty string if nothing
    can be extracted.

    Parse floats and ints to strings without interpreting them as numbers if keep_answer_raw is True.
    """
    try:
        if keep_answer_raw:
            json_object = json.loads(response, parse_float=str, parse_int=str)
        else:
            json_object = json.loads(response)
        if isinstance(json_object, dict) and "answer" in json_object:
            return str(json_object["answer"])
    except Exception:
        pass
    # Fallback: attempt to extract first numeric token from text
    try:
        numbers = parse_numbers_from_text(response)
        return str(numbers[0]) if numbers else ""
    except Exception:
        return ""