import requests
from enum import EnumType, Enum
import os

####################
# LLM Models
####################
LLAMA_2_MODELS = [
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf",
    "Llama-2-70b-chat-hf",
]

OPENAI_LLM_MODELS = ["gpt-3.5-turbo", "gpt-4"]

LLM_MODELS = [*OPENAI_LLM_MODELS, *LLAMA_2_MODELS]

# Anyscale configs
ANYSCALE_LLM_PREFIX = "meta-llama/"
ANYSCALE_API_URL = "https://api.endpoints.anyscale.com/v1"

####################
# Embedding Models
####################
OPENAI_EMB_MODELS = ["text-embedding-ada-002"]

EMB_MODELS = [*OPENAI_EMB_MODELS]


####################
# Enumerations
####################
class CVGradeAnswerPrompt(Enum):
    FAST = "fast"
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    NONE = "none"


class CVGradeDocumentsPrompt(Enum):
    DEFAULT = "default"
    NONE = "none"


class CVRetrieverSearchType(Enum):
    SIMILARITY = "similarity"
    MMR = "mmr"


BACKEND_URL = os.environ.get("EVALBACKEND_URL")
CHROMADB_URL = os.environ.get("CHROMADB_URL")


def fetch_data(api_url: str, endpoint: str):
    """Fetch data from the given endpoint."""
    response = requests.get(f"http://{api_url}{endpoint}")

    # Handle response errors if needed
    response.raise_for_status()

    return response.json()


def test_backend_get_endpoints():
    """Fetch data for all endpoints."""
    endpoints = {
        "/configs/llm_models": LLM_MODELS,
        "/configs/embedding_models": EMB_MODELS,
        "/configs/retriever_search_types": CVRetrieverSearchType,
        "/configs/grade_answer_prompts": CVGradeAnswerPrompt,
        "/configs/grade_documents_prompts": CVGradeDocumentsPrompt,
    }

    for endpoint, expected_values in endpoints.items():
        fetched_data = fetch_data(BACKEND_URL, endpoint)

        # Assert that the fetched data matches the enum values
        if isinstance(expected_values, list):
            assert set(fetched_data) == set(
                expected_values
            ), f"Mismatch in endpoint {endpoint}"

        elif isinstance(expected_values, EnumType):
            assert set(fetched_data) == set(
                item.value for item in expected_values
            ), f"Mismatch in endpoint {endpoint}"

        else:
            raise NotImplementedError(
                f"Type {type(expected_values)} not implemented for endpoint {endpoint}."
            )
