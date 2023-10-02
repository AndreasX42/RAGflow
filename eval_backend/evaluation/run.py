import logging
import asyncio

import numpy as np
from datetime import datetime

from eval_backend.utils import (
    get_retriever,
    get_qa_llm,
    aget_retrieved_documents,
)

from eval_backend.evaluation.evaluation_metrics import (
    grade_embedding_similarity,
    grade_model_answer,
    grade_model_retrieval,
    grade_rouge,
)

from eval_backend.utils.doc_processing import aload_and_chunk_docs
from eval_backend.commons import Hyperparameters

logger = logging.getLogger(__name__)


async def run_eval(
    gt_dataset: list[dict[str, str]], hp: Hyperparameters, docs_path: str
) -> dict:
    scores = {
        "embedding_cosine_sim": np.array([]),
        "correct_ans": np.array([]),
        "comprehensive_ans": np.array([]),
        "readable_ans": np.array([]),
        "retriever_score": np.array([]),
        "rouge1": np.array([]),
        "rouge2": np.array([]),
    }

    # create chunks of all provided documents
    chunks = await aload_and_chunk_docs(hp, docs_path)

    retriever = get_retriever(
        splits=chunks,
        embedding_model=hp.embedding_model,
        num_retrieved_docs=hp.num_retrieved_docs,
    )

    # chunks are now unnecessary
    del chunks

    # llm for answering queries
    qa_llm = get_qa_llm(retriever=retriever, retrieval_llm=hp.retrieval_llm)

    # dict[question, generated answer]
    predicted_answers = await asyncio.gather(
        *[qa_llm.acall(qa_pair) for qa_pair in gt_dataset]
    )

    sim_s = grade_embedding_similarity(
        gt_dataset, predicted_answers, hp.embedding_model
    )
    scores["embedding_cosine_sim"] = np.append(scores["embedding_cosine_sim"], sim_s)

    rouge1_s, rouge2_s = grade_rouge(gt_dataset, predicted_answers)
    scores["rouge1"] = np.append(scores["rouge1"], rouge1_s)
    scores["rouge2"] = np.append(scores["rouge2"], rouge2_s)

    if hp.use_llm_grader:
        correctness_s, comprehensiveness_s, readability_s = grade_model_answer(
            gt_dataset,
            predicted_answers,
            hp.grader_llm,
            hp.grade_answer_prompt,
        )
        scores["correct_ans"] = np.append(scores["correct_ans"], correctness_s)
        scores["comprehensive_ans"] = np.append(
            scores["comprehensive_ans"], comprehensiveness_s
        )
        scores["readable_ans"] = np.append(scores["readable_ans"], readability_s)

        # dict[question, concatenated string of chunks retrieved]
        retrieved_docs = await asyncio.gather(
            *[aget_retrieved_documents(qa_pair, retriever) for qa_pair in gt_dataset]
        )

        retriever_s = grade_model_retrieval(
            gt_dataset,
            retrieved_docs,
            hp.grader_llm,
            hp.grade_docs_prompt,
        )

        scores["retriever_score"] = np.append(scores["retriever_score"], retriever_s)

    # preprocess dict before writing to json
    scores |= {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]}
    result_dict = hp.to_dict()
    result_dict |= {"scores": scores}

    return result_dict
