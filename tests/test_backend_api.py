import requests
from enum import Enum, EnumType

from backend.commons.configurations.BaseConfigurations import (
    LLM_MODELS,
    EMB_MODELS,
    CVGradeAnswerPrompt,
    CVGradeDocumentsPrompt,
    CVRetrieverSearchType,
)

BASE_URL_BACKEND = "http://localhost:8080"
BASE_URL_FRONTEND = "http://localhost:8501"
BASE_URL_CHROMADB = "http://localhost:8000"


def fetch_data(api_url: str, endpoint: str):
    """Fetch data from the given endpoint."""
    response = requests.get(f"{api_url}{endpoint}")

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
        fetched_data = fetch_data(BASE_URL_BACKEND, endpoint)

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
