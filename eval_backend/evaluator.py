import glob
import dotenv
import tiktoken
import logging
import asyncio
import json
import os

from langchain.chat_models import ChatOpenAI

from eval_backend.utils import write_json
from eval_backend.evaluation import Hyperparameters
from eval_backend.evaluation import run_eval
from eval_backend.testsetgen import agenerate_eval_set

logging.basicConfig(
    level=20,
    format="%(asctime)s - %(levelname)s - %(name)s:%(filename)s:%(lineno)d - %(message)s",
)

logger = logging.getLogger(__name__)

dotenv.load_dotenv(dotenv.find_dotenv(), override=True)

qa_gen_configs = {
    "qa_generator_llm": ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    "length_function_for_qa_generation": lambda x: len(
        tiktoken.encoding_for_model("text-embedding-ada-002").encode(x)
    ),
    "generate_new_eval_set": False,
}


async def main(
    chain="",
    retriever="",
    retriever_type="",
    k=3,
    grade_prompt="",
    docs_path="./resources",
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

    ################################################################
    # First phase: Loading or generating evaluation dataset
    ################################################################
    if not qa_gen_configs["generate_new_eval_set"] and os.path.exists(
        eval_dataset_path
    ):
        try:
            with open(eval_dataset_path, "r", encoding="utf-8") as file:
                logger.info("Evaluation dataset found, loading it.")
                gt_dataset = json.load(file)
        except FileNotFoundError:
            logger.error(
                f"Evaluation dataset could not be loaded, {eval_dataset_path}."
            )
            qa_gen_configs["generate_new_eval_set"] = True

    elif qa_gen_configs["generate_new_eval_set"]:
        logger.warning("Evaluation dataset not found, starting generation process.")

        # tarnsform list of list of dicts into list of dicts
        qa_pairs = await agenerate_eval_set(
            qa_gen_configs, docs_path=glob.glob(f"{docs_path}/*.pdf")
        )

        # only get pairs with sufficiently long answer
        gt_dataset = [qa_pair for qa_pair in qa_pairs if len(qa_pair["answer"]) > 150]

        # write eval dataset to json
        write_json(gt_dataset, filename=eval_dataset_path)

    else:
        raise AttributeError("Something went wrong loading evaluation dataset.")

    ################################################################
    # Second phase: Running evaluations
    ################################################################
    with open(eval_params_path, "r", encoding="utf-8") as file:
        hyperparams_list = json.load(file)

    # evaluate hyperparams concurrently
    results = await asyncio.gather(
        *[
            run_eval(gt_dataset, hp, docs_path)
            for hp in [Hyperparameters.from_dict(d) for d in hyperparams_list]
        ]
    )

    # write final results to json
    write_json(results, eval_results_path)


if __name__ == "__main__":
    asyncio.run(main())
