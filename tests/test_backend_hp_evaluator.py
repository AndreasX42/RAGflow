import json
import logging
import chromadb
import pytest

from tests.utils import (
    RAGFLOW_HOST,
    RAGFLOW_PORT,
    CHROMADB_HOST,
    CHROMADB_PORT,
    HP_EVALUATION_ENDPOINT,
    DOCUMENT_STORE_PATH,
    LABEL_DATASET_PATH,
    HYPERPARAMETERS_PATH,
    HYPERPARAMETERS_RESULTS_PATH,
    HYPERPARAMETERS_RESULTS_DATA_PATH,
    fetch_data,
    first_user_id,
    second_user_id,
    third_user_id_with_invalid_uuid,
    user_id_without_upsert,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "user_id_fixture, num_hp_run",
    [("first_user_id", 0), ("second_user_id", 1), ("first_user_id", 2)],
)
def test_ragflow_evaluator(user_id_fixture, num_hp_run, request):
    """Test the evaluation of provided hyperparameter configurations."""

    def get_fixture_value(fixture_name):
        """Get the value associated with a fixture name."""
        return request.getfixturevalue(fixture_name)

    user_id = get_fixture_value(user_id_fixture)

    json_payload = {
        "document_store_path": DOCUMENT_STORE_PATH,
        "label_dataset_path": LABEL_DATASET_PATH,
        "hyperparameters_path": HYPERPARAMETERS_PATH,
        "hyperparameters_results_path": HYPERPARAMETERS_RESULTS_PATH,
        "hyperparameters_results_data_path": HYPERPARAMETERS_RESULTS_DATA_PATH,
        "user_id": user_id,
        "api_keys": {},
    }

    response = fetch_data(
        method="post",
        host=RAGFLOW_HOST,
        port=RAGFLOW_PORT,
        endpoint=HP_EVALUATION_ENDPOINT,
        payload=json_payload,
    )

    client = chromadb.HttpClient(
        host=CHROMADB_HOST,
        port=CHROMADB_PORT,
    )

    collections_list = client.list_collections()
    collection_names = [col.name for col in collections_list]

    # Check if hp_id get incremented correctly for users, the second evaluation of first user should create a collection with hpid_1 as suffix
    if len(collections_list) in range(3, 6):
        for i in range(3):
            assert (
                f"userid_{user_id[:8]}_hpid_{i}" in collection_names
            ), "First user with 3 hp evals should have 3 corresponding collections now"
    elif len(collections_list) in range(6, 9):
        for i in range(3):
            assert (
                f"userid_{user_id[:8]}_hpid_{i}" in collection_names
            ), "Second user with 3 hp evals should have 3 corresponding collections now"
    elif len(collections_list) in range(9, 12):
        for i in range(3, 6):
            assert (
                f"userid_{user_id[:8]}_hpid_{i}" in collection_names
            ), "First user with next 3 hp evals should have 6 corresponding collections now"
    else:
        assert False, "Unexpected collections_list length"

    # Check if hyperparameters from json were set correctly
    assert (
        response.status_code == 200
    ), f"The response from the {HP_EVALUATION_ENDPOINT} endpoint should be ok."

    # test the generated results of the evaluation run
    with open(json_payload["hyperparameters_results_path"], encoding="utf-8") as file:
        hyperparameters_results_list = json.load(file)

    with open(json_payload["hyperparameters_path"], encoding="utf-8") as file:
        hyperparameters_list = json.load(file)

    hyperparameters_results = hyperparameters_results_list[num_hp_run]
    hyperparameters = hyperparameters_list[num_hp_run]

    assert (
        hyperparameters["similarity_method"]
        == hyperparameters_results["vectorstore_params"]["metadata"]["hnsw:space"]
    ), "Retriever similarity methods should match."

    assert (
        hyperparameters["search_type"]
        == hyperparameters_results["retriever_params"]["search_type"]
    ), "Retriever search types should match."

    assert (
        hyperparameters["num_retrieved_docs"]
        == hyperparameters_results["retriever_params"]["search_kwargs"]["k"]
    ), "Number of retrieved docs should match."

    assert (
        hyperparameters["embedding_model"]
        == hyperparameters_results["retriever_params"]["tags"][1]
    ), "Embedding models should match."
