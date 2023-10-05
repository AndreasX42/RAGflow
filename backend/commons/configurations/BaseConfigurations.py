from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI, ChatAnyscale

import tiktoken
import builtins
import logging

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


LLAMA_2_MODELS = (
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf",
    "Llama-2-70b-chat-hf",
)

OPENAI_LLM_MODELS = ("gpt-3.5-turbo", "gpt-4")
OPENAI_EMB_MODELS = "text-embedding-ada-002"

ANYSCALE_LLM_PREFIX = "meta-llama/"
ANYSCALE_API_URL = "https://api.endpoints.anyscale.com/v1"


class BaseConfigurations(ABC):
    """Base class for configuration objects."""

    @staticmethod
    def get_language_model(model_name: str) -> BaseLanguageModel:
        if model_name in OPENAI_LLM_MODELS:
            return ChatOpenAI(model_name=model_name, temperature=0.0)
        elif model_name in LLAMA_2_MODELS:
            return ChatAnyscale(
                model_name=f"{ANYSCALE_LLM_PREFIX}{model_name}",
                anyscale_api_base=ANYSCALE_API_URL,
                temperature=0.0,
            )
        raise NotImplementedError(f"LLM model '{model_name}' not supported.")

    @staticmethod
    def get_embedding_model(model_name: str) -> Embeddings:
        if model_name in OPENAI_EMB_MODELS:
            return OpenAIEmbeddings(model=model_name)

        raise NotImplementedError(f"Embedding model '{model_name}' not supported.")

    @classmethod
    def set_length_function(cls, length_function_name: str) -> callable:
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
                logger.error(f"Length function '{length_function_name}' not supported")
                raise ex

    @staticmethod
    def get_language_model_name(llm: BaseLanguageModel) -> str:
        """Retrieve name of language model from object"""
        return llm.model_name

    @staticmethod
    def get_embedding_model_name(emb: Embeddings) -> str:
        """Retrieve name of embedding model name from object"""
        return emb.model

    @abstractmethod
    def to_dict(self):
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, input_dict):
        pass
