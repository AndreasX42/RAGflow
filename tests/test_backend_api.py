import requests
from enum import EnumType
import os

LLM_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4",
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf",
    "Llama-2-70b-chat-hf",
]

EMB_MODELS = ["text-embedding-ada-002"]

CVGradeAnswerPrompt = ["fast", "zero_shot", "few_shot", "none"]
CVGradeDocumentsPrompt = ["default", "none"]
CVRetrieverSearchType = ["similarity", "mmr"]


BACKEND_HOST = os.environ.get("EVALBACKEND_HOST")
BACKEND_PORT = os.environ.get("EVALBACKEND_PORT")

CHROMADB_HOST = os.environ.get("CHROMADB_HOST")
CHROMADB_PORT = os.environ.get("CHROMADB_PORT")


def fetch_data(host: str, port: str, endpoint: str):
    """Fetch data from the given endpoint."""
    response = requests.get(f"http://{host}:{port}{endpoint}")

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
        fetched_data = fetch_data(BACKEND_HOST, BACKEND_PORT, endpoint)

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
