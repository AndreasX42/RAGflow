from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel

from eval_backend.commons.configurations import BaseConfigurations

import logging

from pydantic.v1 import BaseModel, Extra, validator
from typing import Any

logger = logging.getLogger(__name__)


class Hyperparameters(BaseModel, BaseConfigurations):
    """Class to model hyperparameters."""

    id: int
    chunk_size: int
    chunk_overlap: int
    num_retrieved_docs: int
    grade_answer_prompt: str
    grade_docs_prompt: str
    search_type: str
    use_llm_grader: bool
    retrieval_llm: BaseLanguageModel
    grader_llm: BaseLanguageModel
    embedding_model: Embeddings
    length_function_name: str
    length_function: Any

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True
        extra = Extra.forbid

    @validator("length_function", pre=False, always=True)
    def populate_length_function(cls, v: callable, values: dict[str, str]):
        return cls.set_length_function(values["length_function_name"])

    def to_dict(self):
        _data = self.dict()
        _data.pop("length_function", None)

        # Modify the dictionary for fields that need special handling
        _data["retrieval_llm"] = _data["retrieval_llm"]["model_name"]
        _data["grader_llm"] = _data["grader_llm"]["model_name"]
        _data["embedding_model"] = _data["embedding_model"]["model"]

        return _data

    @classmethod
    def from_dict(cls, input_dict: dict[str, str]):
        _input = dict(**input_dict)

        _input["embedding_model"] = cls.get_embedding_model(_input["embedding_model"])
        _input["retrieval_llm"] = cls.get_language_model(_input["retrieval_llm"])
        _input["grader_llm"] = cls.get_language_model(_input["grader_llm"])

        return cls(**_input)
