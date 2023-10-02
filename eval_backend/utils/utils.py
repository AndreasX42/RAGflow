import numpy as np
import os

from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.retriever import BaseRetriever

# vector db
from langchain.vectorstores import FAISS

from eval_backend.commons.prompts import QA_ANSWER_PROMPT

from typing import Optional

import json
import logging

logger = logging.getLogger(__name__)


def get_retriever(
    splits: list[Document],
    embedding_model: Embeddings,
    num_retrieved_docs: int,
    search_type: Optional[str] = "mmr",
) -> BaseRetriever:
    """Sets up a vector database based on the document chunks and the embedding model provided.
        Here we use FAISS for the vectorstore.

    Args:
        splits (List[Document]): the document chunks
        embedding_model (Embeddings): embedding model
        num_retrieved_docs (int): number of document chunks retrieved doing similarity search

    Returns:
        BaseRretriever: Returns a retriever object
    """
    logger.info("Constructing vectorstore and retriever.")

    vectorstore = FAISS.from_documents(splits, embedding_model)
    retriever = vectorstore.as_retriever(k=num_retrieved_docs, search_type=search_type)

    return retriever


def get_qa_llm(
    retriever: BaseRetriever,
    retrieval_llm: BaseLanguageModel,
) -> RetrievalQA:
    """Sets up a LangChain RetrievalQA model based on a retriever and language model that answers
    queries based on retrieved document chunks.


    Args:
        retriever (BaseRetriever): the retriever
        retrieval_llm (Optional[BaseLanguageModel], optional): language model.

    Returns:
        RetrievalQA: RetrievalQA object
    """
    logger.info("Setting up QA LLM with provided retriever.")

    chain_type_kwargs = {"prompt": QA_ANSWER_PROMPT}

    qa_llm = RetrievalQA.from_chain_type(
        llm=retrieval_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        input_key="question",
    )
    return qa_llm


async def aget_retrieved_documents(
    qa_pair: dict[str, str], retriever: BaseRetriever
) -> dict:
    """Retrieve most similar documents to query asynchronously, postprocess the document chunks and return the qa pair with the retrieved documents string as result

    Args:
        qa_pair (dict[str, str]): _description_
        retriever (BaseRetriever): _description_

    Returns:
        list[dict]: _description_
    """
    query = qa_pair["question"]
    docs_retrieved = await retriever.aget_relevant_documents(query)

    retrieved_doc_text = "\n\n".join(
        f"Retrieved document {i}: {doc.page_content}"
        for i, doc in enumerate(docs_retrieved)
    )
    retrieved_dict = {
        "question": qa_pair["question"],
        "answer": qa_pair["answer"],
        "result": retrieved_doc_text,
    }

    return retrieved_dict


def write_json(data: dict, filename: str) -> None:
    """Function used to store generated QA pairs, i.e. the ground truth.
    TODO: Provide metadata to have 'Data provenance' of the ground truth QA pairs.

    Args:
        data (_type_): _description_
        filename (str, optional): _description_.
    """

    logger.info(f"Writting JSON to {filename}.")

    # Check if file exists
    if os.path.exists(filename):
        # File exists, read the data
        with open(filename, "r", encoding="utf-8") as file:
            json_data = json.load(file)
            # Assuming the data is a list; you can modify as per your requirements
            json_data.extend(data)
    else:
        # File doesn't exist; set data as the new_data
        # This assumes the main structure is a list; modify as needed
        json_data = data  # [data]

    # Write the combined data back to the file
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(json_data, file, indent=4, default=convert_to_serializable)


# Convert non-serializable types
def convert_to_serializable(obj: object) -> str:
    """Preprocessing step before writing to json file

    Args:
        obj (object): _description_

    Returns:
        str: _description_
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    elif callable(obj):  # For <built-in function len> and similar types
        return str(obj)
    elif isinstance(obj, type):  # For <class ...>
        return str(obj)
    return f"WARNING: Type {type(obj).__name__} not serializable!"
