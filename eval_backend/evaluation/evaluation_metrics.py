import numpy as np

from langchain.evaluation.qa import QAEvalChain
from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel

import evaluate

from eval_backend.commons.prompts import (
    GRADE_ANSWER_PROMPT_5_CATEGORIES_5_GRADES_ZERO_SHOT,
    GRADE_ANSWER_PROMPT_3_CATEGORIES_4_GRADES_FEW_SHOT,
    GRADE_ANSWER_PROMPT_FAST,
    GRADE_RETRIEVER_PROMPT,
)

from eval_backend.evaluation.utils import extract_llm_metric

from typing import Optional
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

    # try using embeddings of answers of evaluation set from vectorstore
    # if not available, we calculate them again
    try:
        import chromadb
        import yaml

        CONFIG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
        VS_CLIENT = chromadb.PersistentClient(CONFIG["ChromaDBPathEvalSet"])

        collection = VS_CLIENT.get_collection(name=embedding_model.model)
        ids = [qa["metadata"]["id"] for qa in gt_dataset]

        target_embeddings = np.array(
            collection.get(ids=ids, include=["embeddings"])["embeddings"]
        ).reshape(num_qa_pairs, -1)

        logger.info("Embeddings for label answers loaded successfully.")

    except Exception as ex:
        logger.info(
            f"Embeddings for label answers could not be loaded from vectorstore, {str(ex)}"
        )

        target_embeddings = np.array(
            embedding_model.embed_documents(label_answers)
        ).reshape(num_qa_pairs, -1)

    predicted_embeddings = np.array(
        embedding_model.embed_documents(predicted_answers)
    ).reshape(num_qa_pairs, -1)

    # similarities between openai embeddings ranges from 0.7 - 1.0 only
    similarities = target_embeddings @ predicted_embeddings.T

    return np.nanmean(np.diag(similarities))


def grade_model_retrieval(
    gt_dataset: list[dict],
    retrieved_docs: list[str],
    grader_llm: BaseLanguageModel,
    grade_docs_prompt: Optional[str] = "default",
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
    logger.debug("Grading retrieved document chunks.")

    if grade_docs_prompt == "default":
        prompt = GRADE_RETRIEVER_PROMPT
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

    return np.nanmean(retrieval_accuracy)


def grade_model_answer(
    gt_dataset: list[dict[str, str]],
    predictions: list[str],
    grader_llm: BaseLanguageModel,
    grade_answer_prompt: Optional[str] = "few_shot",
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
    logger.debug("Grading generated answers.")

    MAX_GRADE = 0
    if grade_answer_prompt == "fast":
        prompt = GRADE_ANSWER_PROMPT_FAST
        MAX_GRADE = 5
    elif grade_answer_prompt == "zero_shot":
        prompt = GRADE_ANSWER_PROMPT_5_CATEGORIES_5_GRADES_ZERO_SHOT
        MAX_GRADE = 5
    elif grade_answer_prompt == "few_shot":
        prompt = GRADE_ANSWER_PROMPT_3_CATEGORIES_4_GRADES_FEW_SHOT
        MAX_GRADE = 3
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
        np.nanmean(correctness) / MAX_GRADE,
        np.nanmean(comprehensiveness) / MAX_GRADE,
        np.nanmean(readability) / MAX_GRADE,
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
