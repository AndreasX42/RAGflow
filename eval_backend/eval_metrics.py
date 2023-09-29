import numpy as np

from langchain.evaluation.qa import QAEvalChain
from langchain.chat_models import ChatOpenAI
from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel

import evaluate

from typing import Dict, List, Optional, Callable

from .prompts import (
    GRADE_ANSWER_PROMPT_FAST,
    GRADE_ANSWER_PROMPT_BIAS_CHECK,
    GRADE_ANSWER_PROMPT_OPENAI,
    GRADE_ANSWER_PROMPT,
    GRADE_ANSWER_PROMPT_WITH_GRADING,
    GRADE_ANSWER_PROMPT_WITH_GRADING_JUSTIFICATION,
    GRADE_DOCS_PROMPT_FAST,
    GRADE_DOCS_PROMPT,
)


def grade_embedding_similarity(
    gt_dataset: List[Dict[str, str]],
    predictions: List[str],
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
    num_qa_pairs = len(gt_dataset)

    target_embeddings = np.array(
        embedding_model.embed_documents([qa_pair["answer"] for qa_pair in gt_dataset])
    ).reshape(num_qa_pairs, -1)
    predicted_embeddings = np.array(
        embedding_model.embed_documents(predictions)
    ).reshape(num_qa_pairs, -1)

    # similarities between openai embeddings ranges from 0.7 - 1.0 only
    similarities = target_embeddings @ predicted_embeddings.T

    return np.average(np.diag(similarities))


def grade_model_retrieval(
    gt_dataset: List[Dict],
    predictions: List[str],
    grade_docs_prompt: str,
    grader_llm: Optional[BaseLanguageModel] = ChatOpenAI(
        model_name="gpt-3.5-turbo", temperature=0
    ),
) -> float:
    """Using LangChains QAEvalChain we use a LLM as grader to get a metric how well the found document chunks from the vectorstore provide the answer for the label QA pairs.

    Args:
        gt_dataset (List[Dict]): _description_
        predictions (List[str]): _description_
        grade_docs_prompt (str): _description_
        grader_llm (Optional[BaseLanguageModel], optional): _description_. Defaults to ChatOpenAI( model_name="gpt-3.5-turbo", temperature=0 ).

    Returns:
        float: average score across all labels
    """
    if grade_docs_prompt == "Fast":
        prompt = GRADE_DOCS_PROMPT_FAST
    else:
        prompt = GRADE_DOCS_PROMPT

    # Note: GPT-4 grader is advised by OAI
    eval_chain = QAEvalChain.from_llm(llm=grader_llm, prompt=prompt)

    graded_outputs = eval_chain.evaluate(
        gt_dataset,
        predictions,
        question_key="question",
        answer_key="answer",
        prediction_key="result",
    )

    average_grade = np.average(
        [
            0 if "incorrect" in grade["results"][:30].lower() else 1
            for grade in graded_outputs
        ]
    )

    return graded_outputs, average_grade


def grade_model_answer(
    gt_datase: List[Dict[str, str]],
    predictions: List[str],
    grade_answer_prompt: str,
    grader_llm: Optional[BaseLanguageModel] = ChatOpenAI(
        model_name="gpt-3.5-turbo", temperature=0
    ),
) -> float:
    """Calculates scores how well the generated answer fits the label answers.

    Args:
        gt_datase (List[Dict[str, str]]): _description_
        predictions (List[str]): _description_
        grade_answer_prompt (str): _description_
        grader_llm (Optional[BaseLanguageModel], optional): _description_. Defaults to ChatOpenAI( model_name="gpt-3.5-turbo", temperature=0 ).

    Returns:
        float: average score
    """
    if grade_answer_prompt == "Fast":
        prompt = GRADE_ANSWER_PROMPT_FAST
    elif grade_answer_prompt == "Descriptive w/ bias check":
        prompt = GRADE_ANSWER_PROMPT_BIAS_CHECK
    elif grade_answer_prompt == "OpenAI grading prompt":
        prompt = GRADE_ANSWER_PROMPT_OPENAI
    elif grade_answer_prompt == "Grade w/o Reasoning":
        prompt = GRADE_ANSWER_PROMPT_WITH_GRADING
    elif grade_answer_prompt == "Grade w/ Reasoning":
        prompt = GRADE_ANSWER_PROMPT_WITH_GRADING_JUSTIFICATION
    else:
        prompt = GRADE_ANSWER_PROMPT

    # Note: GPT-4 grader is advised by OAI model_name="gpt-4"
    eval_chain = QAEvalChain.from_llm(llm=grader_llm, prompt=prompt, verbose=False)

    graded_outputs = eval_chain.evaluate(
        gt_datase, predictions, question_key="question", prediction_key="result"
    )

    average_grade = np.average([int(grade["results"][-1]) for grade in graded_outputs])

    return graded_outputs, average_grade


def grade_rouge(references: List[str], predictions: List[str]) -> tuple[float, float]:
    """Calculates rouge1 and rouge2 scores of the label and generated answers

    Args:
        references (List[str]): _description_
        predictions (List[str]): _description_

    Returns:
        tuple[float, float]: rouge1 and rouge2 scores
    """
    rouge_score = evaluate.load("rouge")
    score = rouge_score.compute(references=references, predictions=predictions)

    return score["rouge1"], score["rouge2"]
