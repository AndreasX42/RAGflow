import os
import asyncio
import itertools

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import (
    TextLoader,
    Docx2txtLoader,
    # PyPDFLoader,  # already splits data during loading
    UnstructuredPDFLoader,  # returns only 1 Document
    # PyMuPDFLoader,  # returns 1 Document per page
)

from eval_backend.common import Hyperparameters

import logging

logger = logging.getLogger(__name__)


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


def load_and_chunk_doc(file: str, hp: Hyperparameters) -> list[Document]:
    data = load_document(file)
    chunks = split_data(data, hp.chunk_size, hp.chunk_overlap, hp.length_function)
    return chunks


async def aload_and_chunk_docs(files: list[str], hp: Hyperparameters) -> list[Document]:
    loop = asyncio.get_event_loop()

    futures = [
        loop.run_in_executor(None, load_and_chunk_doc, file, hp) for file in files
    ]

    results = await asyncio.gather(*futures)
    chunks = list(itertools.chain.from_iterable(results))

    return chunks
