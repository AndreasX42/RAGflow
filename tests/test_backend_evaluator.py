import json
import logging
import glob
import pytest
import os

from tests.utils import (
    BACKEND_HOST,
    BACKEND_PORT,
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


def test_backend_evaluator(first_user_id):
    """Test the evaluation of provided hyperparameter configurations."""

    json_payload = {
        "document_store_path": DOCUMENT_STORE_PATH,
        "label_dataset_path": LABEL_DATASET_PATH,
        "hyperparameters_path": HYPERPARAMETERS_PATH,
        "hyperparameters_results_path": HYPERPARAMETERS_RESULTS_PATH,
        "hyperparameters_results_data_path": HYPERPARAMETERS_RESULTS_DATA_PATH,
        "user_id": first_user_id,
        "api_keys": {},
    }

    response = fetch_data(
        method="post",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        endpoint=HP_EVALUATION_ENDPOINT,
        payload=json_payload,
    )

    assert (
        response.status_code == 200
    ), f"The response from the {HP_EVALUATION_ENDPOINT} endpoint should be ok."

    # test the generated results of the evaluation run
    with open(json_payload["hyperparameters_results_path"], encoding="utf-8") as file:
        hyperparameters_results_list = json.load(file)

    with open(json_payload["hyperparameters_path"], encoding="utf-8") as file:
        hyperparameters_list = json.load(file)

    hyperparameters_results = hyperparameters_results_list[0]
    hyperparameters = hyperparameters_list[0]

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
