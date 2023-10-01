from json import JSONDecodeError
import itertools
import os
import re
import numpy as np

from langchain.chains import RetrievalQA, QAGenerationChain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.retriever import BaseRetriever

from langchain.document_loaders import (
    TextLoader,
    Docx2txtLoader,
    # PyPDFLoader,  # already splits data during loading
    UnstructuredPDFLoader,  # returns only 1 Document
    # PyMuPDFLoader,  # returns 1 Document per page
)

# vector db
from langchain.vectorstores import FAISS

from prompts import QA_CHAIN_PROMPT

import tiktoken
import json
import logging

logger = logging.getLogger(__name__)


def get_retriever(
    splits: list[Document],
    embedding_model: Embeddings,
    num_retrieved_docs: int,
    search_type: str = "mmr",
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

    chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}

    qa_llm = RetrievalQA.from_chain_type(
        llm=retrieval_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        input_key="question",
    )
    return qa_llm


def generate_eval_set(
    llm: BaseLanguageModel, chunks: list[Document]
) -> list[dict[str, str]]:
    """Generate a pairs of QAs that are used as ground truth in downstream tasks, i.e. RAG evaluations

    Args:
        llm (BaseLanguageModel): the language model used in the QAGenerationChain
        chunks (List[Document]): the document chunks used for QA generation

    Returns:
        List[Dict[str, str]]: returns a list of dictionary of question - answer pairs
    """

    logger.debug("Generating qa pairs.")

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


def load_document(file: str) -> list[Document]:
    """Loads file from given path into a list of Documents, currently pdf, txt and docx are supported.

    Args:
        file (str): file path

    Returns:
        List[Document]: loaded files as list of Documents
    """
    logger.info(f"Loading file {file}")

    _, extension = os.path.splitext(file)

    if extension == ".pdf":
        loader = UnstructuredPDFLoader(file)
    elif extension == ".txt":
        loader = TextLoader(file, encoding="utf-8")
    elif extension == ".docx":
        loader = Docx2txtLoader(file)
    else:
        logger.warning("Unsupported file type detected!")
        raise Warning("Unsupported file type detected!")

    data = loader.load()
    return data


def split_data(
    data: list[Document],
    chunk_size: int = 4096,
    chunk_overlap: int = 0,
    length_function: callable = len,
) -> list[Document]:
    """Function for splitting the provided data, i.e. List of documents loaded.

    Args:
        data (List[Document]): _description_
        chunk_size (Optional[int], optional): _description_. Defaults to 4096.
        chunk_overlap (Optional[int], optional): _description_. Defaults to 0.
        length_function (_type_, optional): _description_.

    Returns:
        List[Document]: the splitted document chunks
    """

    logger.debug("Splitting data.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
    )

    chunks = text_splitter.split_documents(data)
    return chunks


def extract_llm_metric(text: str, metric: str) -> np.number:
    """Utiliy function for extracting scores from LLM output from grading of generated answers and retrieved document chunks.

    Args:
        text (str): LLM result
        metric (str): name of metric

    Returns:
        number: the found score as integer or np.nan
    """
    match = re.search(f"{metric}: (\d+)", text)
    if match:
        return int(match.group(1))
    return np.nan


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
            json_data.append(data)
    else:
        # File doesn't exist; set data as the new_data
        # This assumes the main structure is a list; modify as needed
        json_data = [data]

    # Write the combined data back to the file
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(json_data, file, indent=4)  # default=convert_to_serializable,
