import evaluate

import logging

logger = logging.getLogger(__name__)


def grade_rouge(
    label_dataset: list[dict], predicted_answers: list[dict]
) -> tuple[float, float]:
    """Calculates rouge1 and rouge2 scores of the label and generated answers.

    Args:
        label_dataset (list[dict]): The evaluation ground truth dataset of QA pairs
        predictions (list[dict]): The predicted answers

    Returns:
        tuple[float, float]: The rouge1 and rouge2 average scores
    """
    logger.info("Calculating ROUGE scores.")

    label_answers = [qa_pair["answer"] for qa_pair in label_dataset]
    predicted_answers = [qa_pair["result"] for qa_pair in predicted_answers]

    rouge_score = evaluate.load("rouge")
    score = rouge_score.compute(references=label_answers, predictions=predicted_answers)

    return score["rouge1"], score["rouge2"]
