import numpy as np
import re


def extract_llm_metric(text: str, metric: str) -> np.number:
    """Utiliy function for extracting scores from LLM output from grading of generated answers and retrieved document chunks.

    Args:
        text (str): LLM result
        metric (str): name of metric

    Returns:
        number: the found score as integer or np.nan
    """
    match = re.search(f"{metric}: (\d+)", text)
    if match:
        return int(match.group(1))
    return np.nan
