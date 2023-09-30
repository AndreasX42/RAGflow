import numpy as np

from langchain.evaluation.qa import QAEvalChain
from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel

import evaluate

from prompts import (
    GRADE_ANSWER_PROMPT_3CATEGORIES_ZERO_SHOT,
    GRADE_ANSWER_PROMPT_3CATEGORIES_ZERO_SHOT_WITH_REASON,
    GRADE_ANSWER_PROMPT_FAST,
    GRADE_DOCS_PROMPT,
)

from utils import extract_llm_metric

import logging

logger = logging.getLogger(__name__)


def grade_embedding_similarity(
    gt_dataset: list[dict[str, str]],
    predictions: list[dict[str, str]],
    embedding_model: Embeddings,
) -> float:
    """Calculate similarities of label answers and generated answers using the corresponding embeddings. We multiply the matrixes of the provided embeddings and take the average of the diagonal values, which should be the cosine similarites assuming that the embeddings were already normalized.
    TODO: Make sure embedding model returns normalized vectors.

    Args:
        gt_dataset (List[Dict[str, str]]): _description_
        predictions (List[str]): _description_
        embedding_model (Embeddings): _description_

    Returns:
        float: the average cosine similarities
    """
    logger.debug("Calculating embedding similarities.")

    num_qa_pairs = len(gt_dataset)

    label_answers = [qa_pair["answer"] for qa_pair in gt_dataset]
    predicted_answers = [qa_pair["result"] for qa_pair in predictions]

    target_embeddings = np.array(
        embedding_model.embed_documents(label_answers)
    ).reshape(num_qa_pairs, -1)

    predicted_embeddings = np.array(
        embedding_model.embed_documents(predicted_answers)
    ).reshape(num_qa_pairs, -1)

    # similarities between openai embeddings ranges from 0.7 - 1.0 only
    similarities = target_embeddings @ predicted_embeddings.T

    return np.average(np.diag(similarities))


def grade_model_retrieval(
    gt_dataset: list[dict],
    retrieved_docs: list[str],
    grade_docs_prompt: str,
    grader_llm: BaseLanguageModel,
) -> float:
    """Using LangChains QAEvalChain we use a LLM as grader to get a metric how well the found document chunks from the vectorstore provide the answer for the label QA pairs.

    Args:
        gt_dataset (List[Dict]): _description_
        retrieved_docs (List[str]): _description_
        grade_docs_prompt (str): _description_
        grader_llm (Optional[BaseLanguageModel], optional): _description_.

    Returns:
        float: average score across all labels
    """
    logger.info("Grading retrieved document chunks.")

    if grade_docs_prompt not in ("", None):
        prompt = GRADE_DOCS_PROMPT
    else:
        prompt = None

    # Note: GPT-4 grader is advised by OAI
    eval_chain = QAEvalChain.from_llm(llm=grader_llm, prompt=prompt)

    outputs = eval_chain.evaluate(
        gt_dataset,
        retrieved_docs,
        question_key="question",
        answer_key="answer",
        prediction_key="result",
    )

    # vectorize the function for efficiency
    v_extract_llm_metric = np.vectorize(extract_llm_metric)

    outputs = np.array([output["results"] for output in outputs])
    retrieval_accuracy = v_extract_llm_metric(outputs, "GRADE")

    return np.average(retrieval_accuracy)


def grade_model_answer(
    gt_dataset: list[dict[str, str]],
    predictions: list[str],
    grade_answer_prompt: str,
    grader_llm: BaseLanguageModel,
) -> tuple[float, float, float]:
    """Calculates scores how well the generated answer fits the label answers.

    Args:
        gt_dataset (List[Dict[str, str]]): _description_
        predictions (List[str]): _description_
        grade_answer_prompt (str): _description_
        grader_llm (Optional[BaseLanguageModel], optional): _description_.

    Returns:
        float: average scores, for "fast" grading only average corectness score, otherwise we get the grading LLM to calculate correctness, comprehensiveness and readability scores.
    """
    logger.info("Grading generated answers.")

    if grade_answer_prompt == "fast":
        prompt = GRADE_ANSWER_PROMPT_FAST
    elif grade_answer_prompt == "3cats_zero_shot":
        prompt = GRADE_ANSWER_PROMPT_3CATEGORIES_ZERO_SHOT
    elif grade_answer_prompt == "3cats_zero_shot_with_reason":
        prompt = GRADE_ANSWER_PROMPT_3CATEGORIES_ZERO_SHOT_WITH_REASON
    else:
        prompt = None

    # Note: GPT-4 grader is advised by OAI model_name="gpt-4"
    eval_chain = QAEvalChain.from_llm(llm=grader_llm, prompt=prompt, verbose=False)

    outputs = eval_chain.evaluate(
        gt_dataset, predictions, question_key="question", prediction_key="result"
    )

    # vectorize the function for efficiency
    v_extract_llm_metric = np.vectorize(extract_llm_metric)

    outputs = np.array([output["results"] for output in outputs])

    correctness = v_extract_llm_metric(outputs, "CORRECTNESS")
    comprehensiveness = v_extract_llm_metric(outputs, "COMPREHENSIVENESS")
    readability = v_extract_llm_metric(outputs, "READABILITY")

    return (
        np.average(correctness),
        np.average(comprehensiveness),
        np.average(readability),
    )


def grade_rouge(
    gt_dataset: list[dict[str, str]], predictions: list[dict[str, str]]
) -> tuple[float, float]:
    """Calculates rouge1 and rouge2 scores of the label and generated answers

    Args:
        references (List[str]): _description_
        predictions (List[str]): _description_

    Returns:
        tuple[float, float]: rouge1 and rouge2 scores
    """
    logger.debug("Calculating ROUGE scores.")

    label_answers = [qa_pair["answer"] for qa_pair in gt_dataset]
    predicted_answers = [qa_pair["result"] for qa_pair in predictions]

    rouge_score = evaluate.load("rouge")
    score = rouge_score.compute(references=label_answers, predictions=predicted_answers)

    return score["rouge1"], score["rouge2"]
