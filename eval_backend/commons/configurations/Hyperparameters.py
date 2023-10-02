from dataclasses import dataclass, asdict, field

from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

from eval_backend.commons.configurations import BaseConfigurations

import tiktoken
import builtins
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True, slots=True)
class Hyperparameters(BaseConfigurations):
    """Class to model hyperparameters."""

    chunk_size: int
    chunk_overlap: int
    num_retrieved_docs: int
    grade_answer_prompt: str
    grade_docs_prompt: str
    use_llm_grader: bool
    retrieval_llm: BaseLanguageModel
    grader_llm: BaseLanguageModel
    embedding_model: Embeddings
    length_function_name: str
    length_function: callable = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "length_function", self.set_length_function(self.length_function_name)
        )

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

    def to_dict(self):
        data = asdict(self)
        data.pop("length_function", None)

        # Modify the dictionary for fields that need special handling
        data["retrieval_llm"] = self.inverse_language_model(data["retrieval_llm"])
        data["grader_llm"] = self.inverse_language_model(data["grader_llm"])
        data["embedding_model"] = self.inverse_embedding_model(data["embedding_model"])

        return data

    @classmethod
    def from_dict(cls, input_dict):
        input_dict["embedding_model"] = cls.get_embedding_model(
            input_dict["embedding_model"]
        )
        input_dict["retrieval_llm"] = cls.get_language_model(
            input_dict["retrieval_llm"]
        )
        input_dict["grader_llm"] = cls.get_language_model(input_dict["grader_llm"])

        return cls(**input_dict)
