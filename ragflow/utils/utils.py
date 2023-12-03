import numpy as np
import os
from enum import Enum
from datetime import datetime

from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.vectorstore import VectorStoreRetriever

# vector db
from langchain.vectorstores.chroma import Chroma

from ragflow.commons.prompts import QA_ANSWER_PROMPT
from ragflow.commons.configurations import Hyperparameters
from ragflow.commons.chroma import ChromaClient

from typing import Any, Optional, Union

import json
import logging

logger = logging.getLogger(__name__)


def get_retriever(
    chunks: list[Document], hp: Hyperparameters, user_id: str, for_eval: bool = False
) -> Union[tuple[VectorStoreRetriever, VectorStoreRetriever], VectorStoreRetriever]:
    """Sets up a vector database based on the document chunks and the embedding model provided.
        Here we use Chroma for the vectorstore.

    Args:
        chunks (list[Document]): _description_
        hp (Hyperparameters): _description_
        user_id (str): _description_
        for_eval (bool): for evaluation we return a second retriever with k=10

    Returns:
        tuple[VectorStoreRetriever, VectorStoreRetriever]: Returns two retrievers, one for predicting the
        answers and one for grading the retrieval where we need to have the top 10 documents returned
    """
    logger.info("Constructing vectorstore and retriever.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=hp.embedding_model,
        client=ChromaClient().get_client(),
        collection_name=f"userid_{user_id[:8]}_hpid_{hp.id}",
        collection_metadata={
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
            "custom_id": f"userid_{user_id}_hpid_{hp.id}",
            "hnsw:space": hp.similarity_method.value,
        },
    )

    retriever = vectorstore.as_retriever(
        search_type=hp.search_type.value, search_kwargs={"k": hp.num_retrieved_docs}
    )

    # if we are evaluating the semantic accuracy of the retriever as well, we have to return a second version of the retriever with different settings
    if not for_eval:
        return retriever

    retrieverForGrading = vectorstore.as_retriever(
        search_type=hp.search_type.value, search_kwargs={"k": 10}
    )

    return retriever, retrieverForGrading


def get_qa_llm(
    retriever: VectorStoreRetriever,
    qa_llm: BaseLanguageModel,
    return_source_documents: Optional[bool] = True,
) -> RetrievalQA:
    """Sets up a LangChain RetrievalQA model based on a retriever and language model that answers
    queries based on retrieved document chunks.

    Args:
        qa_llm (Optional[BaseLanguageModel], optional): language model.

    Returns:
        RetrievalQA: RetrievalQA object
    """
    logger.debug("Setting up QA LLM with provided retriever.")

    qa_llm_r = RetrievalQA.from_chain_type(
        llm=qa_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_ANSWER_PROMPT},
        input_key="question",
        return_source_documents=return_source_documents,
    )

    return qa_llm_r


def read_json(filename: str) -> Any:
    """Load dataset from a JSON file."""

    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data: list[dict], filename: str, append: Optional[bool] = True) -> None:
    """Function used to store generated QA pairs, i.e. the ground truth.

    Args:
        data (_type_): _description_
        filename (str, optional): _description_.
    """

    logger.info(f"Writting JSON to {filename}.")

    # Check if file exists
    if os.path.exists(filename) and append:
        # File exists, read the data
        with open(filename, "r", encoding="utf-8") as file:
            json_data = json.load(file)
            # Assuming the data is a list; you can modify as per your requirements
            json_data.extend(data)
    else:
        json_data = data

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
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    elif callable(obj):  # For <built-in function len> and similar types
        return str(obj)
    elif isinstance(obj, type):  # For <class ...>
        return str(obj)
    return f"WARNING: Type {type(obj).__name__} not serializable!"
