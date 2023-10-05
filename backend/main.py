import glob
import dotenv
import logging
import asyncio
import json
import os
from tqdm.asyncio import tqdm as tqdm_asyncio

import chromadb
from chromadb.config import Settings

from backend.commons.configurations import Hyperparameters, QAConfigurations
from backend.utils import read_json, write_json
from backend.evaluation import run_eval
from backend.testsetgen import agenerate_and_save_dataset

logging.basicConfig(
    level=20,
    format="%(asctime)s - %(levelname)s - %(name)s:%(filename)s:%(lineno)d - %(message)s",
)

from typing import Optional

logger = logging.getLogger(__name__)

dotenv.load_dotenv(dotenv.find_dotenv(), override=True)


async def main(
    docs_path: str = "./resources/document_store/",
    eval_dataset_path: str = "./resources/eval_data.json",
    eval_params_path: str = "./resources/eval_params.json",
    eval_results_path: str = "./resources/eval_results.json",
    debug_enabled: Optional[bool] = True,
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

    CHROMA_CLIENT = chromadb.HttpClient(
        host="localhost",
        port=8000,
        settings=Settings(allow_reset=True, anonymized_telemetry=False),
    )

    document_store = glob.glob(f"{docs_path}/*.pdf")

    with open(eval_params_path, "r", encoding="utf-8") as file:
        hyperparams_list = json.load(file)

    # set up Hyperparameters objects at the beginning to evaluate inputs
    qa_gen_params = QAConfigurations.from_dict(hyperparams_list[0])
    hyperparams_list = [Hyperparameters.from_dict(d) for d in hyperparams_list[1:]]

    ################################################################
    # First phase: Loading or generating evaluation dataset
    ################################################################

    logger.info("Checking for evaluation dataset configs.")

    # generate evaluation dataset
    if qa_gen_params.generate_eval_set or not CHROMA_CLIENT.list_collections():
        if os.path.exists(eval_dataset_path):
            logger.info(
                "Existing evaluation dataset deleted due to 'generate_eval_set'=True."
            )
            os.remove(eval_dataset_path)

        # save list of embedding models for later caching of embeddings
        if qa_gen_params.persist_to_vs:
            emb_models_list = [hp.embedding_model for hp in hyperparams_list]
            kwargs = {"embedding_models": emb_models_list}

        kwargs["debug_enabled"] = debug_enabled

        # reset chromadb before
        CHROMA_CLIENT.reset()

        gt_dataset = await agenerate_and_save_dataset(
            qa_gen_params, document_store, eval_dataset_path, kwargs
        )

    else:
        logger.info("Evaluation dataset found and loaded.")

        # load qa pairs from chromadb
        collection = CHROMA_CLIENT.list_collections()[0]
        metadata = collection.get(include=["metadatas"])["metadatas"]
        gt_dataset = list(
            map(
                lambda qa: {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "metadata": {"id": qa["id"]},
                },
                metadata,
            )
        )

    ################################################################
    # Second phase: Running evaluations
    ################################################################

    print(gt_dataset[0])
    print()
    logger.info("Starting evaluation for all provided hyperparameters.")

    # evaluate hyperparams concurrently
    tasks = [run_eval(gt_dataset, hp, document_store) for hp in hyperparams_list]

    results = await tqdm_asyncio.gather(*tasks, total=len(tasks))

    # write final results to json
    write_json(results, eval_results_path)


if __name__ == "__main__":
    asyncio.run(main())
