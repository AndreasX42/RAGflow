from dataclasses import dataclass, asdict, field
from langchain.schema.language_model import BaseLanguageModel

from eval_backend.commons.configurations import BaseConfigurations

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True, slots=True)
class QAConfigurations(BaseConfigurations):
    """Class to model qa generation configs."""

    chunk_size: int
    chunk_overlap: int
    qa_generator_llm: BaseLanguageModel
    generate_eval_set: bool
    length_function_name: str
    length_function: callable = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "length_function", self.set_length_function(self.length_function_name)
        )

    def to_dict(self):
        data = asdict(self)
        data.pop("length_function", None)

        # Modify the dictionary for fields that need special handling
        data["qa_generator_llm"] = self.inverse_language_model(data["qa_generator_llm"])

        return data

    @classmethod
    def from_dict(cls, input_dict):
        input_dict["qa_generator_llm"] = cls.get_language_model(
            input_dict["qa_generator_llm"]
        )
        return cls(**input_dict)
