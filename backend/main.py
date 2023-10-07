import dotenv
import logging
import json
import asyncio

from backend.commons.configurations import Hyperparameters, QAConfigurations
from backend.evaluation import arun_evaluation
from backend.testsetgen import aget_or_generate_eval_set

logging.basicConfig(
    level=20,
    format="%(asctime)s - %(levelname)s - %(name)s:%(filename)s:%(lineno)d - %(message)s",
)

logger = logging.getLogger(__name__)

dotenv.load_dotenv(dotenv.find_dotenv(), override=True)


async def main(
    document_store_path: str = "./tmp/document_store/",
    eval_dataset_path: str = "./tmp/eval_data.json",
    qa_gen_params_path: str = "./tmp/qa_gen_params.json",
    eval_params_path: str = "./tmp/eval_params.json",
    eval_results_path: str = "./tmp/eval_results.json",
    hp_runs_data_path: str = "./tmp/hp_runs_data.csv",
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

    with open(qa_gen_params_path, "r", encoding="utf-8") as file:
        qa_gen_params = json.load(file)

    # set up Hyperparameters objects at the beginning to evaluate inputs
    qa_gen_params = QAConfigurations.from_dict(qa_gen_params[-1])

    ################################################################
    # First phase: Loading or generating evaluation dataset
    ################################################################

    logger.info("Checking for evaluation dataset configs.")

    # generate evaluation dataset
    await aget_or_generate_eval_set(
        qa_gen_params, eval_dataset_path, document_store_path
    )

    ################################################################
    # Second phase: Running evaluations
    ################################################################

    logger.info("Starting evaluation for all provided hyperparameters.")

    await arun_evaluation(
        document_store_path,
        eval_params_path,
        eval_dataset_path,
        eval_results_path,
        hp_runs_data_path,
    )


if __name__ == "__main__":
    asyncio.run(main())
