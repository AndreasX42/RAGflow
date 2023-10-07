import logging
import asyncio
from tqdm.asyncio import tqdm as tqdm_asyncio
import glob

from datetime import datetime

from backend.utils import get_retriever, get_qa_llm, write_json, read_json

from backend.evaluation.evaluation_metrics import (
    grade_embedding_similarity,
    grade_model_answer,
    grade_model_retrieval,
    grade_rouge,
)

from backend.utils.doc_processing import aload_and_chunk_docs
from backend.evaluation.utils import process_retrieved_docs, write_generated_data_to_csv
from backend.commons.configurations import Hyperparameters

logger = logging.getLogger(__name__)


async def arun_eval_for_hp(
    gt_dataset: list[dict[str, str]],
    hp: Hyperparameters,
    document_store: list[str],
) -> dict:
    """Entry point for initiating the evaluation based on the provided hyperparameters and documents.

    Args:
        gt_dataset (list[dict[str, str]]): _description_
        hp (BaseConfigurations): _description_
        docs_path (str): _description_

    Returns:
        dict: _description_
    """
    scores = {
        "embedding_cosine_sim": -1,
        "correct_ans": -1,
        "comprehensive_ans": -1,
        "readable_ans": -1,
        "retriever_score": -1,
        "rouge1": -1,
        "rouge2": -1,
    }

    # create chunks of all provided documents
    chunks = await aload_and_chunk_docs(hp, document_store)

    retriever = get_retriever(
        splits=chunks,
        embedding_model=hp.embedding_model,
        num_retrieved_docs=hp.num_retrieved_docs,
        search_type=hp.search_type,
    )

    # chunks are no longer needed
    del chunks

    # llm for answering queries
    qa_llm = get_qa_llm(retriever, hp.qa_llm)

    # list(dict[question, result, source_documents])
    predicted_answers = await asyncio.gather(
        *[qa_llm.acall(qa_pair) for qa_pair in gt_dataset]
    )

    # list of retrieved docs for each qa_pair
    process_retrieved_docs(predicted_answers, hp.id)

    # Calculate embedding similarities
    sim_s = grade_embedding_similarity(
        gt_dataset, predicted_answers, hp.embedding_model
    )

    scores["embedding_cosine_sim"] = sim_s

    # Calculate ROUGE scores
    rouge1_s, rouge2_s = grade_rouge(gt_dataset, predicted_answers)
    scores["rouge1"] = rouge1_s
    scores["rouge2"] = rouge2_s

    # if we want a llm to grade the predicted answers as well
    if hp.use_llm_grader:
        # grade predicted answers
        correctness_s, comprehensiveness_s, readability_s = grade_model_answer(
            gt_dataset,
            predicted_answers,
            hp.grader_llm,
            hp.grade_answer_prompt,
        )
        scores["correct_ans"] = correctness_s
        scores["comprehensive_ans"] = comprehensiveness_s
        scores["readable_ans"] = readability_s

        # grade quality of retrieved documents used to answer the questions
        retriever_s = grade_model_retrieval(
            gt_dataset,
            predicted_answers,
            hp.grader_llm,
            hp.grade_docs_prompt,
        )

        scores["retriever_score"] = retriever_s

    # preprocess dict before writing to json
    scores |= {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]}
    result_dict = hp.to_dict()
    result_dict |= {"scores": scores}

    return result_dict, predicted_answers


async def arun_evaluation(
    document_store_path: str,
    eval_params_path: str,
    eval_dataset_path: str,
    eval_results_path: str,
    hp_runs_data_path: str,
) -> None:
    # load evaluation dataset
    gt_dataset = read_json(eval_dataset_path)

    document_store = glob.glob(f"{document_store_path}/*.pdf")

    hyperparams_list = read_json(eval_params_path)
    hp_list = [Hyperparameters.from_dict(d) for d in hyperparams_list]

    tasks = [arun_eval_for_hp(gt_dataset, hp, document_store) for hp in hp_list]

    # run evaluations for all hyperparams
    results = await tqdm_asyncio.gather(*tasks, total=len(tasks))

    eval_scores, predicted_answers = list(zip(*results))

    # write eval metrics to json
    write_json(eval_scores, eval_results_path)

    # write predicted answers and retrieved docs to csv
    write_generated_data_to_csv(predicted_answers, hp_runs_data_path)
