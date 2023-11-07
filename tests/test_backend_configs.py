from tests.utils import RAGFLOW_HOST, RAGFLOW_PORT, fetch_data

LLM_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4",
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf",
    "Llama-2-70b-chat-hf",
]

EMB_MODELS = ["text-embedding-ada-002"]

CVGradeAnswerPrompt = ["zero_shot", "few_shot", "none"]
CVGradeRetrieverPrompt = ["default", "none"]
CVRetrieverSearchType = ["similarity", "mmr"]
CVSimilarityMethod = ["cosine", "l2", "ip"]


def test_ragflow_config_endpoints():
    """Fetch data for all endpoints."""
    endpoints = {
        "/configs/llm_models": LLM_MODELS,
        "/configs/embedding_models": EMB_MODELS,
        "/configs/retriever_similarity_methods": CVSimilarityMethod,
        "/configs/retriever_search_types": CVRetrieverSearchType,
        "/configs/grade_answer_prompts": CVGradeAnswerPrompt,
        "/configs/grade_documents_prompts": CVGradeRetrieverPrompt,
    }

    for endpoint, expected_values in endpoints.items():
        response = fetch_data(
            method="get", host=RAGFLOW_HOST, port=RAGFLOW_PORT, endpoint=endpoint
        ).json()

        # Assert that the fetched data matches the enum values
        if isinstance(expected_values, list):
            assert set(response) == set(
                expected_values
            ), f"Mismatch in endpoint {endpoint}"

        else:
            raise NotImplementedError(
                f"Type {type(expected_values)} not implemented for endpoint {endpoint}."
            )
