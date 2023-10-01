from langchain.chains import QAGenerationChain

from json import JSONDecodeError
import itertools
import asyncio

from eval_backend.utils import load_and_chunk_doc

import logging

logger = logging.getLogger(__name__)


def generate_eval_set(hp: dict, doc_path: str) -> list[dict[str, str]]:
    """Generate a pairs of QAs that are used as ground truth in downstream tasks, i.e. RAG evaluations

    Args:
        llm (BaseLanguageModel): the language model used in the QAGenerationChain
        chunks (List[Document]): the document chunks used for QA generation

    Returns:
        List[Dict[str, str]]: returns a list of dictionary of question - answer pairs
    """

    logger.debug("Generating qa pairs.")

    # load data and chunk doc
    chunks = load_and_chunk_doc(doc_path, hp)

    llm = hp["qa_generator_llm"]
    qa_generator_chain = QAGenerationChain.from_llm(llm)
    eval_set = []

    for i, chunk in enumerate(chunks):
        if llm.get_num_tokens(chunk.page_content) < 1500:
            print(f"Not considering {i}/{len(chunks)}")
            continue

        try:
            qa_pair = qa_generator_chain.run(chunk.page_content)
            eval_set.append(qa_pair)
        except JSONDecodeError:
            print(f"Error occurred inside QAChain in chunk {i}/{len(chunks)}")

    eval_set = list(itertools.chain.from_iterable(eval_set))

    return eval_set


async def agenerate_eval_set(qa_gen_configs: dict, docs_path: list[str]) -> list[dict]:
    loop = asyncio.get_event_loop()

    # Set up generate_eval_set to be run concurrently using asyncio's default executor.
    futures = [
        loop.run_in_executor(None, generate_eval_set, qa_gen_configs, doc)
        for doc in docs_path
    ]

    results = await asyncio.gather(*futures)
    qa_pairs = list(itertools.chain.from_iterable(results))

    return qa_pairs
