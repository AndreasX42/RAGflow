from json import JSONDecodeError
import itertools
import asyncio

from eval_backend.utils import load_and_chunk_doc
from eval_backend.commons import Hyperparameters

from langchain.chains import QAGenerationChain
from langchain.docstore.document import Document
from eval_backend.commons.prompts import QA_GENERATION_PROMPT_SELECTOR

import logging

logger = logging.getLogger(__name__)


async def get_qa_from_chunk(
    chunk: Document, qa_generator_chain: QAGenerationChain, eval_set: list
):
    try:
        # return list of qa pairs
        qa_pairs = qa_generator_chain.run(chunk.page_content)

        # attach chunk metadata to qa_pair
        for qa_pair in qa_pairs:
            qa_pair["metadata"] = chunk.metadata

        eval_set.append(qa_pairs)
    except JSONDecodeError:
        pass


async def generate_eval_set(
    qa_gen_configs: dict, doc_path: str
) -> list[dict[str, str]]:
    """Generate a pairs of QAs that are used as ground truth in downstream tasks, i.e. RAG evaluations

    Args:
        llm (BaseLanguageModel): the language model used in the QAGenerationChain
        chunks (List[Document]): the document chunks used for QA generation

    Returns:
        List[Dict[str, str]]: returns a list of dictionary of question - answer pairs
    """

    logger.info("Start generating evaluation dataset.")

    hp = Hyperparameters.from_dict(qa_gen_configs)

    # load data and chunk doc
    chunks = load_and_chunk_doc(hp, doc_path)

    llm = hp.retrieval_llm
    qa_generator_chain = QAGenerationChain.from_llm(
        llm, prompt=QA_GENERATION_PROMPT_SELECTOR.get_prompt(llm)
    )
    eval_set = []

    tasks = [get_qa_from_chunk(chunk, qa_generator_chain, eval_set) for chunk in chunks]

    await asyncio.gather(*tasks)

    eval_set = list(itertools.chain.from_iterable(eval_set))

    return eval_set


async def agenerate_eval_set(qa_gen_configs: dict, docs_path: list[str]) -> list[dict]:
    """Asynchronous wrapper around the agenerate_eval_set function.

    Args:
        qa_gen_configs (dict): _description_
        docs_path (list[str]): _description_

    Returns:
        list[dict]: _description_
    """
    loop = asyncio.get_event_loop()

    # Set up generate_eval_set to be run concurrently using asyncio's default executor.
    tasks = [
        loop.run_in_executor(None, generate_eval_set, qa_gen_configs, doc)
        for doc in docs_path
    ]

    results = await asyncio.gather(*tasks)
    qa_pairs = list(itertools.chain.from_iterable(results))

    return qa_pairs
