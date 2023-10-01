import glob
import dotenv
import tiktoken
import logging
import asyncio
import json

from datetime import datetime

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

import numpy as np

from utils import (
    load_document,
    split_data,
    get_retriever,
    get_qa_llm,
    generate_eval_set,
    write_json,
    aget_retrieved_documents,
)

from eval_metrics import (
    grade_embedding_similarity,
    grade_model_answer,
    grade_model_retrieval,
    grade_rouge,
)

from hyperparameters import Hyperparameters

logging.basicConfig(
    level=20,
    format="%(asctime)s - %(levelname)s - %(name)s:%(filename)s:%(lineno)d - %(message)s",
)

logger = logging.getLogger(__name__)

dotenv.load_dotenv(dotenv.find_dotenv(), override=True)

hyperparams_list = [
    {  # settings for qa generation, for the time being fixed across all evaluation runs
        "qa_generator_llm": ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        "length_function_for_qa_generation": lambda x: len(
            tiktoken.encoding_for_model("text-embedding-ada-002").encode(x)
        ),
        "generate_new_eval_set": False,
    },
    {
        # chunking stratety
        "chunk_size": 512,
        "chunk_overlap": 10,
        "length_function_name": "len",
        # vectordb and retriever
        "embedding_model": "text-embedding-ada-002",
        "num_retrieved_docs": 3,
        # qa llm
        "retrieval_llm": "gpt-3.5-turbo",
        # answer grading llm
        "use_llm_grader": False,  # if you want to generate metrics with llm grading
        "grader_llm": "gpt-4",
        "grade_answer_prompt": "3cats_zero_shot",
        "grade_docs_prompt": "default",
    },
]


async def run_eval(
    chain="",
    retriever="",
    retriever_type="",
    k=3,
    grade_prompt="",
    file_path="./resources",
    eval_dataset_path="./resources/eval_data.json",
    eval_results_path="./resources/eval_results.json",
):
    """After generating gt_dataset we need to calculate the metrics based on Chunking strategy, type of vectorstore, retriever (similarity search), QA LLM

        Dependencies of parameters and build order:
        generating chunks               | depends on chunking strategy
            |__ vector database         | depends on type von DB
                    |__  retriever      | depends on Embedding model
                            |__ QA LLM  | depends on LLM and digests retriever

            |__ grader: llm who grades generated answers

    TODO 1: State of the art retrieval with lots of additional LLM calls:
        - use "MMR" to filter out similiar document chunks during similarity search
        - Use SelfQueryRetriever to find information in query about specific context, e.g. querying specific document only
        - Use ContextualCompressionRetriever to compress or summarize the document chunks before serving them to QA-LLM

        - retriever object would like something like this, additional metadata info for SelfQueryRetriever has to be provided as well
                retriever = ContextualCompressionRetriever(
                    base_compressor=LLMChainExtractor.from_llm(llm),
                    base_retriever=SelfQueryRetriever.from_llm(..., earch_type = "mmr")
                )

        https://learn.deeplearning.ai/langchain-chat-with-your-data/lesson/5/retrieval
        https://learn.deeplearning.ai/langchain-chat-with-your-data/lesson/7/chat

    TODO 2: Integrate Anyscale (Llama2), MosaicML (MPT + EMBDS), Replicate

    Args:
        chain (str, optional): _description_. Defaults to "".
        retriever (str, optional): _description_. Defaults to "".
        retriever_type (str, optional): _description_. Defaults to "".
        k (int, optional): _description_. Defaults to 3.
        grade_prompt (str, optional): _description_. Defaults to "".
        file_path (str, optional): _description_. Defaults to "./resources".
        eval_gt_path (str, optional): _description_. Defaults to "./resources/eval_data.json".
    """

    # generate qa pairs as evaluation dataset
    qa_gen_configs = hyperparams_list[0]

    try:
        with open(eval_dataset_path, "r", encoding="utf-8") as file:
            logger.info("Evaluation dataset found, loading it.")
            gt_dataset = json.load(file)

    except FileNotFoundError:
        logger.warning("Evaluation dataset not found, creating it.")

        for file in glob.glob(f"{file_path}/*.pdf"):
            data = load_document(file)

            # TODO: len function for qa generation should be hyperparam?
            chunks = split_data(
                data=data,
                chunk_size=3000,
                chunk_overlap=0,
                length_function=qa_gen_configs["length_function_for_qa_generation"],
            )
            qa_pairs = generate_eval_set(
                llm=qa_gen_configs["qa_generator_llm"], chunks=chunks
            )

            # only get pairs with sufficiently long answer
            gt_dataset += [
                qa_pair for qa_pair in qa_pairs if len(qa_pair["answer"]) > 150
            ]

            write_json(gt_dataset, filename=eval_dataset_path)

    for i, hyperparams in enumerate(hyperparams_list[1:]):
        logger.info(
            f"Starting hyperparameter evaluation {i}/{len(hyperparams_list)-1}."
        )

        hp = Hyperparameters.from_dict(hyperparams)

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
        # TODO: for now only the msg pdf in resources folder
        chunks_list = []
        for file in glob.glob(f"{file_path}/*.pdf"):
            data = load_document(file)
            chunks = split_data(
                data=data,
                chunk_size=hp.chunk_size,
                chunk_overlap=hp.chunk_overlap,
                length_function=hp.length_function,
            )

            chunks_list += chunks

        retriever = get_retriever(
            splits=chunks_list,
            embedding_model=hp.embedding_model,
            num_retrieved_docs=hp.num_retrieved_docs,
        )

        qa_llm = get_qa_llm(retriever=retriever, retrieval_llm=hp.retrieval_llm)

        # dict[question, generated answer]
        predicted_answers = await asyncio.gather(
            *[qa_llm.acall(qa_pair) for qa_pair in gt_dataset]
        )

        sim_s = grade_embedding_similarity(
            gt_dataset, predicted_answers, hp.embedding_model
        )
        scores["embedding_cosine_sim"] = np.append(
            scores["embedding_cosine_sim"], sim_s
        )

        rouge1_s, rouge2_s = grade_rouge(gt_dataset, predicted_answers)
        scores["rouge1"] = np.append(scores["rouge1"], rouge1_s)
        scores["rouge2"] = np.append(scores["rouge2"], rouge2_s)

        if hp.use_llm_grader:
            correctness_s, comprehensiveness_s, readability_s = grade_model_answer(
                gt_dataset,
                predicted_answers,
                hp.grade_answer_prompt,
                hp.grader_llm,
            )
            scores["correct_ans"] = np.append(scores["correct_ans"], correctness_s)
            scores["comprehensive_ans"] = np.append(
                scores["comprehensive_ans"], comprehensiveness_s
            )
            scores["readable_ans"] = np.append(scores["readable_ans"], readability_s)

            # dict[question, concatenated string of chunks retrieved]
            retrieved_docs = await asyncio.gather(
                *[
                    aget_retrieved_documents(qa_pair, retriever)
                    for qa_pair in gt_dataset
                ]
            )

            retriever_s = grade_model_retrieval(
                gt_dataset,
                retrieved_docs,
                hp.grade_docs_prompt,
                hp.grader_llm,
            )

            scores["retriever_score"] = np.append(
                scores["retriever_score"], retriever_s
            )

        # writing results to json
        scores |= {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]}
        result_dict = hp.to_dict()
        result_dict |= {"scores": scores}

        write_json(result_dict, eval_results_path)


if __name__ == "__main__":
    asyncio.run(run_eval())
