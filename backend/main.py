import glob
import dotenv
import logging
import asyncio
import json
import os

from backend.commons.configurations import Hyperparameters, QAConfigurations
from backend.evaluation import arun_eval
from backend.testsetgen import agenerate_and_save_dataset
from backend.commons.chroma import ChromaClient
from backend.utils import read_json

logging.basicConfig(
    level=20,
    format="%(asctime)s - %(levelname)s - %(name)s:%(filename)s:%(lineno)d - %(message)s",
)

logger = logging.getLogger(__name__)

dotenv.load_dotenv(dotenv.find_dotenv(), override=True)


async def main(
    document_store_path: str = "./resources/document_store/",
    eval_dataset_path: str = "./resources/eval_data.json",
    eval_params_path: str = "./resources/eval_params.json",
    eval_results_path: str = "./resources/eval_results.json",
    hp_runs_data_path: str = "./resources/hp_runs_data.csv",
    debug_enabled: bool = True,
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

    with ChromaClient() as client:
        collections = client.list_collections()

    document_store = glob.glob(f"{document_store_path}/*.pdf")

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
    if qa_gen_params.generate_eval_set or not collections:
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
        with ChromaClient() as client:
            client.reset()

        gt_dataset = await agenerate_and_save_dataset(
            qa_gen_params, document_store, eval_dataset_path, kwargs
        )

    else:
        logger.info("Evaluation dataset found and loaded.")

        # load qa pairs from chromadb
        gt_dataset = read_json(eval_dataset_path)

    ################################################################
    # Second phase: Running evaluations
    ################################################################

    logger.info("Starting evaluation for all provided hyperparameters.")

    await arun_eval(
        gt_dataset,
        hyperparams_list,
        document_store,
        eval_results_path,
        hp_runs_data_path,
    )


if __name__ == "__main__":
    asyncio.run(main())
