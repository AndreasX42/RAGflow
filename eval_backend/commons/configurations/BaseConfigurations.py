from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

import tiktoken
import builtins
import logging

logger = logging.getLogger(__name__)


class BaseConfigurations:
    """Base class for configuration objects."""

    @staticmethod
    def get_language_model(model_name: str) -> BaseLanguageModel:
        if model_name in ("gpt-3.5-turbo", "gpt-4"):
            return ChatOpenAI(model=model_name, temperature=0)

        raise NotImplementedError("LLM model not supported.")

    @staticmethod
    def get_embedding_model(model_name: str) -> Embeddings:
        if model_name == "text-embedding-ada-002":
            return OpenAIEmbeddings(model=model_name)

        raise NotImplementedError("Embedding model not supported.")

    def set_length_function(self, length_function_name: str) -> callable:
        # Extract the function name from the string
        func = length_function_name.strip("<>").split(" ")[-1]

        # Check if the function name exists in Python's built-ins
        if hasattr(builtins, func):
            return getattr(builtins, func)

        else:
            try:
                encoding = tiktoken.encoding_for_model(length_function_name)
                return lambda x: len(encoding.encode(x))
            except Exception as ex:
                logger.error("Length function not supported")
                raise ex

    @staticmethod
    def inverse_language_model(llm: BaseLanguageModel) -> str:
        return llm.model_name

    @staticmethod
    def inverse_embedding_model(emb: Embeddings) -> str:
        return emb.model
