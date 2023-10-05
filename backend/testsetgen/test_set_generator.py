from json import JSONDecodeError
import itertools
import asyncio
from tqdm.asyncio import tqdm as tqdm_asyncio

from langchain.chains import QAGenerationChain
from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings

from backend.commons.prompts import QA_GENERATION_PROMPT_SELECTOR
from backend.utils import aload_and_chunk_docs, write_json
from backend.commons.configurations import QAConfigurations

import uuid
from typing import Optional
import logging

logger = logging.getLogger(__name__)

import chromadb
from chromadb.config import Settings


async def get_qa_from_chunk(
    chunk: Document,
    qa_generator_chain: QAGenerationChain,
) -> list[dict]:
    try:
        # return list of qa pairs
        qa_pairs = qa_generator_chain.run(chunk.page_content)

        # attach chunk metadata to qa_pair
        for qa_pair in qa_pairs:
            qa_pair["metadata"] = dict(**chunk.metadata)
            qa_pair["metadata"].update({"id": str(uuid.uuid4())})

        return qa_pairs
    except JSONDecodeError:
        return []


async def agenerate_eval_set_from_doc(
    hp: QAConfigurations,
    doc_path: str,
    kwargs: Optional[dict] = None,
) -> list[dict[str, str]]:
    """Generate a pairs of QAs that are used as ground truth in downstream tasks, i.e. RAG evaluations

    Args:
        llm (BaseLanguageModel): the language model used in the QAGenerationChain
        chunks (List[Document]): the document chunks used for QA generation

    Returns:
        List[Dict[str, str]]: returns a list of dictionary of question - answer pairs
    """

    logger.debug(f"Starting QA generation process for {doc_path}.")

    # load data and chunk doc
    chunks = await aload_and_chunk_docs(hp, [doc_path])

    llm = hp.qa_generator_llm
    qa_generator_chain = QAGenerationChain.from_llm(
        llm, prompt=QA_GENERATION_PROMPT_SELECTOR.get_prompt(llm)
    )

    tasks = [get_qa_from_chunk(chunk, qa_generator_chain) for chunk in chunks]

    qa_pairs = await asyncio.gather(*tasks)
    qa_pairs = list(itertools.chain.from_iterable(qa_pairs))

    return qa_pairs


async def agenerate_eval_set(
    hp: QAConfigurations,
    docs_path: list[str],
    kwargs: Optional[dict] = None,
) -> list[dict]:
    """Asynchronous wrapper around the agenerate_eval_set function.

    Args:
        qa_gen_configs (dict): _description_
        docs_path (list[str]): _description_

    Returns:
        list[dict]: _description_
    """
    tasks = [
        agenerate_eval_set_from_doc(hp, doc_path, kwargs) for doc_path in docs_path
    ]

    results = await tqdm_asyncio.gather(*tasks)

    qa_pairs = list(itertools.chain.from_iterable(results))

    return qa_pairs


async def apersist_eval_set_to_vs(
    qa_pairs: list[dict], embedding_model: Embeddings
) -> None:
    CHROMA_CLIENT = chromadb.HttpClient(
        host="localhost",
        port=8000,
        settings=Settings(anonymized_telemetry=False),
    )

    collection_name = embedding_model.model

    # check if collection already exists, if not create a new one with the embeddings
    if collection_name in [
        collection.name for collection in CHROMA_CLIENT.list_collections()
    ]:
        logger.info(f"Collection {collection_name} already exists, skipping it.")
        return None

    collection = CHROMA_CLIENT.create_collection(name=collection_name)

    ids = [qa_pair["metadata"]["id"] for qa_pair in qa_pairs]
    embeddings = await embedding_model.aembed_documents(
        [qa_pair["answer"] for qa_pair in qa_pairs]
    )

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=[
            {
                "question": qa_pair["question"],
                "answer": qa_pair["answer"],
                **qa_pair["metadata"],
            }
            for qa_pair in qa_pairs
        ],
    )

    logger.info(f"Upserted {embedding_model.model} embeddings to vectorstore.")


async def agenerate_and_save_dataset(
    hp: QAConfigurations,
    docs_path: str,
    eval_dataset_path: str,
    kwargs: Optional[dict] = None,
):
    """Generate a new evaluation dataset and save it to a JSON file."""

    logger.info("Starting QA generation suite.")

    # tarnsform list of list of dicts into list of dicts
    gt_dataset = await agenerate_eval_set(hp, docs_path, kwargs)

    # write eval dataset to json
    write_json(gt_dataset, eval_dataset_path)

    # cache answers of qa pairs in vectorstore for each embedding model in hyperparams list
    unique_model_list = list(
        {
            emb_model.model: emb_model for emb_model in kwargs["embedding_models"]
        }.values()
    )

    if hp.persist_to_vs:
        tasks = [
            apersist_eval_set_to_vs(gt_dataset, embedding_model)
            for embedding_model in unique_model_list
        ]

        await asyncio.gather(*tasks)

    return gt_dataset
