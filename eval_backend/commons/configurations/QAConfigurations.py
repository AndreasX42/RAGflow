import logging
from typing import Any
from pydantic.v1 import BaseModel, Extra, validator

from langchain.schema.language_model import BaseLanguageModel
from eval_backend.commons.configurations import BaseConfigurations

logger = logging.getLogger(__name__)


class QAConfigurations(BaseModel, BaseConfigurations):
    """Class to model qa generation configs."""

    chunk_size: int
    chunk_overlap: int
    qa_generator_llm: BaseLanguageModel
    generate_eval_set: bool
    persist_to_vs: bool
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
        _data["qa_generator_llm"] = _data["qa_generator_llm"]["model_name"]

        return _data

    @classmethod
    def from_dict(cls, input_dict: dict[str, str]):
        _input = dict(**input_dict)
        _input["qa_generator_llm"] = cls.get_language_model(_input["qa_generator_llm"])

        return cls(**_input)
