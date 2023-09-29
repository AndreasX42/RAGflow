from json import JSONDecodeError
import itertools
import numpy as np
import os

from langchain.chains import RetrievalQA, QAGenerationChain
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel

from langchain.document_loaders import (
    TextLoader,
    Docx2txtLoader,
    # PyPDFLoader,  # already splits data during loading
    UnstructuredPDFLoader,  # returns only 1 Document
    # PyMuPDFLoader,  # returns 1 Document per page
)

# vector db
from langchain.vectorstores import FAISS

from .prompts import QA_CHAIN_PROMPT

import tiktoken
import json

from typing import Dict, List, Optional, Callable


def get_retriever(
    splits: List[Document], embedding_model: Embeddings, num_retrieved_docs: int
) -> VectorStoreRetriever:
    """Sets up a vector database based on the document chunks and the embedding model provided.
        Here we use FAISS for the vectorstore.

    Args:
        splits (List[Document]): the document chunks
        embedding_model (Embeddings): embedding model
        num_retrieved_docs (int): number of document chunks retrieved doing similarity search

    Returns:
        VectorStoreRetriever: Returns a retriever object
    """
    vectorstore = FAISS.from_documents(splits, embedding_model)
    retriever = vectorstore.as_retriever(k=num_retrieved_docs)

    return retriever


def get_qa_llm(
    retriever: VectorStoreRetriever,
    retrieval_llm: Optional[BaseLanguageModel] = ChatOpenAI(
        temperature=0, model="gpt-3.5-turbo"
    ),
) -> RetrievalQA:
    """Sets up a LangChain RetrievalQA model based on a retriever and language model that answers
    queries based on retrieved document chunks.


    Args:
        retriever (VectorStoreRetriever): the retriever
        retrieval_llm (Optional[BaseLanguageModel], optional): language model. Defaults to ChatOpenAI( temperature=0, model="gpt-3.5-turbo" ).

    Returns:
        RetrievalQA: RetrievalQA object
    """
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
    llm: BaseLanguageModel, chunks: List[Document]
) -> List[Dict[str, str]]:
    """Generate a pairs of QAs that are used as ground truth in downstream tasks, i.e. RAG evaluations

    Args:
        llm (BaseLanguageModel): the language model used in the QAGenerationChain
        chunks (List[Document]): the document chunks used for QA generation

    Returns:
        List[Dict[str, str]]: returns a list of dictionary of question - answer pairs
    """

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

    eval_pair = list(itertools.chain.from_iterable(eval_set))
    return eval_pair


def load_document(file: str) -> List[Document]:
    """Loads file from given path into a list of Documents, currently pdf, txt and docx are supported.

    Args:
        file (str): file path

    Returns:
        List[Document]: loaded files as list of Documents
    """

    name, extension = os.path.splitext(file)

    if extension == ".pdf":
        loader = UnstructuredPDFLoader(file)
    elif extension == ".txt":
        loader = TextLoader(file, encoding="utf-8")
    elif extension == ".docx":
        loader = Docx2txtLoader(file)

    data = loader.load()
    return data


def split_data(
    data: List[Document],
    chunk_size: Optional[int] = 4096,
    chunk_overlap: Optional[int] = 0,
    length_function: Optional[Callable] = lambda x: len(
        tiktoken.encoding_for_model("text-embedding-ada-002").encode(x)
    ),
) -> List[Document]:
    """Function for splitting the provided data, i.e. List of documents loaded.

    Args:
        data (List[Document]): _description_
        chunk_size (Optional[int], optional): _description_. Defaults to 4096.
        chunk_overlap (Optional[int], optional): _description_. Defaults to 0.
        length_function (_type_, optional): _description_. Defaults to lambdax:len( tiktoken.encoding_for_model("text-embedding-ada-002").encode(x) ).

    Returns:
        List[Document]: the splitted document chunks
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
    )

    chunks = text_splitter.split_documents(data)
    return chunks


def write_json(data, filename="../resources/eval_data.json") -> None:
    """Function used to store generated QA pairs, i.e. the ground truth.
    TODO: Provide metadata to have 'Data provenance' of the ground truth QA pairs.

    Args:
        data (_type_): _description_
        filename (str, optional): _description_. Defaults to "../resources/eval_data.json".
    """
    try:
        # Attempt to open the file for reading
        with open(filename, "r", encoding="utf-8") as file:
            # Load existing data into a list
            file_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, initialize with an empty list
        file_data = []

    # Extend the existing data with the new data
    file_data += data

    # Write the combined data back to the file
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(file_data, file, indent=4)
