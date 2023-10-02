from json import JSONDecodeError
import itertools
import asyncio
import tqdm.asyncio

from langchain.chains import QAGenerationChain
from langchain.docstore.document import Document

from eval_backend.commons.prompts import QA_GENERATION_PROMPT_SELECTOR
from eval_backend.utils import load_and_chunk_doc, write_json
from eval_backend.commons.configurations import QAConfigurations

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


async def agenerate_eval_set_from_doc(
    hp: QAConfigurations, doc_path: str
) -> list[dict[str, str]]:
    """Generate a pairs of QAs that are used as ground truth in downstream tasks, i.e. RAG evaluations

    Args:
        llm (BaseLanguageModel): the language model used in the QAGenerationChain
        chunks (List[Document]): the document chunks used for QA generation

    Returns:
        List[Dict[str, str]]: returns a list of dictionary of question - answer pairs
    """

    logger.info(f"Gtarting QA generation process for {doc_path}.")

    # load data and chunk doc
    chunks = load_and_chunk_doc(hp, doc_path)

    llm = hp.qa_generator_llm
    qa_generator_chain = QAGenerationChain.from_llm(
        llm, prompt=QA_GENERATION_PROMPT_SELECTOR.get_prompt(llm)
    )
    eval_set = []

    tasks = [get_qa_from_chunk(chunk, qa_generator_chain, eval_set) for chunk in chunks]

    await asyncio.gather(*tasks)

    eval_set = list(itertools.chain.from_iterable(eval_set))

    return eval_set


async def agenerate_eval_set(hp: QAConfigurations, docs_path: list[str]) -> list[dict]:
    """Asynchronous wrapper around the agenerate_eval_set function.

    Args:
        qa_gen_configs (dict): _description_
        docs_path (list[str]): _description_

    Returns:
        list[dict]: _description_
    """

    results = await asyncio.gather(
        *[agenerate_eval_set_from_doc(hp, doc_path) for doc_path in docs_path]
    )

    qa_pairs = list(itertools.chain.from_iterable(results))

    return qa_pairs


async def generate_and_save_dataset(
    hp: QAConfigurations, docs_path: str, eval_dataset_path: str
):
    """Generate a new evaluation dataset and save it to a JSON file."""

    logger.info("Starting QA gernation suite.")

    # tarnsform list of list of dicts into list of dicts
    gt_dataset = await agenerate_eval_set(hp, docs_path)

    # only get pairs with sufficiently long answer
    # gt_dataset = [qa_pair for qa_pair in qa_pairs if len(qa_pair["answer"]) > 150]

    # write eval dataset to json
    write_json(gt_dataset, eval_dataset_path)

    return gt_dataset
