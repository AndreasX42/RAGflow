import dotenv
import logging
import asyncio

from ragflow.evaluation import arun_evaluation
from ragflow.generation import agenerate_evaluation_set

logger = logging.getLogger(__name__)

dotenv.load_dotenv(dotenv.find_dotenv(), override=True)


async def main(
    document_store_path: str = "./resources/dev/document_store/",
    label_dataset_path: str = "./resources/dev/label_dataset.json",
    label_dataset_gen_params_path: str = "./resources/dev/label_dataset_gen_params.json",
    hyperparameters_path: str = "./resources/dev/hyperparameters.json",
    hyperparameters_results_path: str = "./resources/dev/hyperparameters_results.json",
    hyperparameters_results_data_path: str = "./resources/dev/hyperparameters_results_data.csv",
    user_id: str = "336e131c-d9f3-4185-a402-f5f8875e34f0",
):
    """After generating label_dataset we need to calculate the metrics based on Chunking strategy, type of vectorstore, retriever (similarity search), QA LLM

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
        eval_gt_path (str, optional): _description_. Defaults to "./resources/label_dataset.json".
    """

    # provide API keys here if necessary
    import os

    api_keys = {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY")}

    ################################################################
    # First phase: Loading or generating evaluation dataset
    ################################################################

    logger.info("Checking for evaluation dataset configs.")

    await agenerate_evaluation_set(
        label_dataset_gen_params_path,
        label_dataset_path,
        document_store_path,
        user_id,
        api_keys,
    )

    ################################################################
    # Second phase: Running evaluations
    ################################################################

    logger.info("Starting evaluation for all provided hyperparameters.")

    await arun_evaluation(
        document_store_path,
        hyperparameters_path,
        label_dataset_path,
        hyperparameters_results_path,
        hyperparameters_results_data_path,
        user_id,
        api_keys,
    )


if __name__ == "__main__":
    asyncio.run(main())
