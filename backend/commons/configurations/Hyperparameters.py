from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel

from backend.commons.configurations.BaseConfigurations import (
    BaseConfigurations,
    LLM_MODELS,
    EMB_MODELS,
    CVGradeAnswerPrompt,
    CVGradeDocumentsPrompt,
    CVRetrieverSearchType,
)

import logging

from pydantic.v1 import Field, validator

logger = logging.getLogger(__name__)


class Hyperparameters(BaseConfigurations):
    """Class to model hyperparameters."""

    id: int = Field(ge=0)
    num_retrieved_docs: int = Field(ge=0)
    grade_answer_prompt: CVGradeAnswerPrompt
    grade_docs_prompt: CVGradeDocumentsPrompt
    search_type: CVRetrieverSearchType
    use_llm_grader: bool
    qa_llm: BaseLanguageModel
    grader_llm: BaseLanguageModel
    embedding_model: Embeddings

    @validator("qa_llm", "grader_llm", pre=True, always=True)
    def check_language_model_name(cls, v, field, values):
        valid_model_names = LLM_MODELS
        if v.model_name not in valid_model_names:
            raise ValueError(f"{v} not in list of valid values {valid_model_names}.")
        return v

    @validator("embedding_model", pre=True, always=True)
    def check_embedding_model_name(cls, v):
        valid_model_names = EMB_MODELS
        if v.model not in valid_model_names:
            raise ValueError(f"{v} not in list of valid values {valid_model_names}.")
        return v

    def to_dict(self):
        _data = self.dict()
        _data.pop("length_function", None)

        # Modify the dictionary for fields that need special handling
        _data["qa_llm"] = _data["qa_llm"]["model_name"]
        _data["grader_llm"] = _data["grader_llm"]["model_name"]
        _data["embedding_model"] = _data["embedding_model"]["model"]

        return _data

    @classmethod
    def from_dict(cls, input_dict: dict[str, str]):
        _input = dict(**input_dict)

        _input["embedding_model"] = cls.get_embedding_model(_input["embedding_model"])
        _input["qa_llm"] = cls.get_language_model(_input["qa_llm"])
        _input["grader_llm"] = cls.get_language_model(_input["grader_llm"])

        # add values that may not be set if use_llm_grader=False
        _input.setdefault("grade_answer_prompt", "none")
        _input.setdefault("grade_docs_prompt", "none")
        _input.setdefault("grader_llm", "gpt-3.5-turbo")

        return cls(**_input)
