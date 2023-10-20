import logging
import asyncio
from tqdm.asyncio import tqdm as tqdm_asyncio
import glob
import os

from datetime import datetime

from backend.utils import get_retriever, get_qa_llm, write_json, read_json
from backend.utils.doc_processing import aload_and_chunk_docs
from backend.evaluation.utils import process_retrieved_docs, write_generated_data_to_csv
from backend.commons.configurations import Hyperparameters

from backend.evaluation.metrics import (
    answer_embedding_similarity,
    predicted_answer_accuracy,
    retriever_accuracy,
    rouge_score,
)

logger = logging.getLogger(__name__)


async def arun_eval_for_hp(
    label_dataset: list[dict[str, str]],
    hp: Hyperparameters,
    document_store: list[str],
    user_id: str,
) -> dict:
    """Entry point for initiating the evaluation based on the provided hyperparameters and documents.

    Args:
        label_dataset (list[dict[str, str]]): _description_
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

    retriever = get_retriever(chunks, hp)

    # chunks are no longer needed
    del chunks

    # llm for answering queries
    qa_llm = get_qa_llm(retriever, hp.qa_llm)

    # list(dict[question, result, source_documents])
    predicted_answers = await asyncio.gather(
        *[qa_llm.acall(qa_pair) for qa_pair in label_dataset]
    )

    # list of retrieved docs for each qa_pair
    process_retrieved_docs(predicted_answers, hp.id)

    # Calculate embedding similarities of label and predicted answers
    emb_sim_score = answer_embedding_similarity.grade_embedding_similarity(
        label_dataset, predicted_answers, hp.embedding_model, user_id
    )

    scores["embedding_cosine_sim"] = emb_sim_score

    # Calculate ROUGE scores
    scores["rouge1"], scores["rouge2"] = rouge_score.grade_rouge(
        label_dataset, predicted_answers
    )

    # if we want a llm to grade the predicted answers as well
    if hp.use_llm_grader:
        # grade predicted answers
        (
            scores["correct_ans"],
            scores["comprehensive_ans"],
            scores["readable_ans"],
        ) = predicted_answer_accuracy.grade_predicted_answer(
            label_dataset,
            predicted_answers,
            hp.grader_llm,
            hp.grade_answer_prompt,
        )

        # grade quality of retrieved documents used to answer the questions
        scores["retriever_score"] = retriever_accuracy.grade_retriever(
            label_dataset,
            predicted_answers,
            hp.grader_llm,
            hp.grade_docs_prompt,
        )

    # preprocess dict before writing to json and add additional information like a timestamp
    scores |= {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]}
    result_dict = hp.to_dict()
    result_dict |= {"scores": scores}

    # if in test mode store additional information about retriever and vectorstore objects
    if os.environ.get("EXECUTION_CONTEXT") == "TEST":
        result_dict |= {"retriever_params": retriever.dict()}
        result_dict |= {"vectorstore_params": retriever.vectorstore._collection.dict()}

    return result_dict, predicted_answers


async def arun_evaluation(
    document_store_path: str,
    hyperparameters_path: str,
    label_dataset_path: str,
    hyperparameters_results_path: str,
    hyperparameters_results_data_path: str,
    user_id: str,
    api_keys: dict,
) -> None:
    """Entry point for the evaluation of a set of hyperparameters to benchmark the corresponding RAG model.

    Args:
        document_store_path (str): Path to the documents
        hyperparameters_path (str): Path to the hyperparameters
        label_dataset_path (str): Path to the evaluation ground truth dataset
        hyperparameters_results_path (str): Path to the JSON file where results should get stored
        hyperparameters_results_data_path (str): Path to the CSV file where additional evaluation data should get stored
        user_id (str): The userd id, probably in UUID format
        api_keys (dict): All required API keys for the LLM endpoints
    """
    # load evaluation dataset
    label_dataset = read_json(label_dataset_path)

    document_store = glob.glob(f"{document_store_path}/*")

    hyperparams_list = read_json(hyperparameters_path)

    hp_list = [Hyperparameters.from_dict(d, api_keys) for d in hyperparams_list]

    tasks = [
        arun_eval_for_hp(label_dataset, hp, document_store, user_id) for hp in hp_list
    ]

    # run evaluations for all hyperparams
    results = await tqdm_asyncio.gather(*tasks, total=len(tasks))

    eval_scores, predicted_answers = list(zip(*results))

    # write eval metrics to json
    write_json(eval_scores, hyperparameters_results_path)

    # write predicted answers and retrieved docs to csv
    write_generated_data_to_csv(predicted_answers, hyperparameters_results_data_path)
