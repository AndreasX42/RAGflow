import logging
from pydantic.v1 import validator

from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.embeddings import Embeddings
from backend.commons.configurations.BaseConfigurations import (
    BaseConfigurations,
    LLM_MODELS,
)

logger = logging.getLogger(__name__)


class QAConfigurations(BaseConfigurations):
    """Class to model qa generation configs."""

    qa_generator_llm: BaseLanguageModel
    generate_eval_set: bool
    persist_to_vs: bool
    embedding_model_list: list[Embeddings]

    @validator("qa_generator_llm", pre=True, always=True)
    def check_language_model_name(cls, v):
        valid_model_names = LLM_MODELS
        if v.model_name not in valid_model_names:
            raise ValueError(f"{v} not in list of valid values {valid_model_names}.")
        return v

    def to_dict(self):
        _data = self.dict()
        _data.pop("length_function", None)

        # Modify the dictionary for fields that need special handling
        _data["qa_generator_llm"] = _data["qa_generator_llm"]["model_name"]
        _data["embedding_model_list"] = [
            model["model"] for model in _data["embedding_model_list"]
        ]

        return _data

    @classmethod
    def from_dict(cls, input_dict: dict[str, str]):
        _input = dict(**input_dict)
        _input["qa_generator_llm"] = cls.get_language_model(_input["qa_generator_llm"])

        # get the list of embedding models, filter the unique models and map them to LangChain objects
        embedding_models = set(_input["embedding_model_list"])

        _input["embedding_model_list"] = [
            cls.get_embedding_model(model) for model in embedding_models
        ]

        return cls(**_input)
