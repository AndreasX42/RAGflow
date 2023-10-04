import glob
import dotenv
import logging
import asyncio
import json
import os
from tqdm.asyncio import tqdm as tqdm_asyncio


from eval_backend.commons.configurations import Hyperparameters, QAConfigurations
from eval_backend.utils import read_json, write_json
from eval_backend.evaluation import run_eval
from eval_backend.testsetgen import agenerate_and_save_dataset

logging.basicConfig(
    level=20,
    format="%(asctime)s - %(levelname)s - %(name)s:%(filename)s:%(lineno)d - %(message)s",
)

logger = logging.getLogger(__name__)

dotenv.load_dotenv(dotenv.find_dotenv(), override=True)


async def main(
    chain="",
    retriever="",
    retriever_type="",
    k=3,
    grade_prompt="",
    docs_path="./resources/document_store/",
    eval_dataset_path="./resources/eval_data.json",
    eval_params_path="./resources/eval_params.json",
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

    document_store = glob.glob(f"{docs_path}/*.pdf")

    with open(eval_params_path, "r", encoding="utf-8") as file:
        hyperparams_list = json.load(file)

    # set up Hyperparameters objects at the beginning to evaluate inputs
    hyperparams_list = [Hyperparameters.from_dict(d) for d in hyperparams_list]

    ################################################################
    # First phase: Loading or generating evaluation dataset
    ################################################################

    logger.info("Checking for evaluation dataset configs.")

    qa_gen_configs = {
        "chunk_size": 2048,
        "chunk_overlap": 0,
        "qa_generator_llm": "gpt-3.5-turbo",
        "length_function_name": "text-embedding-ada-002",
        "generate_eval_set": False,
        "persist_to_vs": True,
    }

    hp = QAConfigurations.from_dict(qa_gen_configs)

    if qa_gen_configs["persist_to_vs"]:
        emb_models_list = [hp.embedding_model for hp in hyperparams_list]
        kwargs = {"embedding_models": emb_models_list}

    if hp.generate_eval_set:
        if os.path.exists(eval_dataset_path):
            os.remove(eval_dataset_path)
            logger.info(
                "Existing evaluation dataset deleted due to 'generate_eval_set'=True."
            )

        gt_dataset = await agenerate_and_save_dataset(
            hp, document_store, eval_dataset_path, kwargs
        )
    elif not os.path.exists(eval_dataset_path):
        logger.warning("Evaluation dataset not found, starting generation process.")
        gt_dataset = await agenerate_and_save_dataset(
            hp, document_store, eval_dataset_path, kwargs
        )
    else:
        gt_dataset = read_json(eval_dataset_path)
        logger.info("Evaluation dataset found and loaded.")

    ################################################################
    # Second phase: Running evaluations
    ################################################################

    logger.info("Starting evaluation for all provided hyperparameters.")

    # evaluate hyperparams concurrently
    tasks = [run_eval(gt_dataset, hp, document_store) for hp in hyperparams_list]

    results = await tqdm_asyncio.gather(*tasks, total=len(tasks))

    # write final results to json
    write_json(results, eval_results_path)


if __name__ == "__main__":
    asyncio.run(main())
