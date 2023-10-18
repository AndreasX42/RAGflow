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
        "document_store_path": "./resources/document_store",
        "eval_dataset_path": "./resources/input_eval_data.json",
        "eval_params_path": "./resources/input_eval_params.json",
        "eval_results_path": "./resources/output_eval_results.json",
        "hp_runs_data_path": "./resources/output_hp_runs_data.csv",
        "user_id": first_user_id,
        "api_keys": {},
    }

    response = fetch_data(
        method="post",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        endpoint="/evaluations/start",
        payload=json_payload,
    )

    assert (
        response.status_code == 200
    ), "The response from the '/evaluations/start' endpoint should be ok."

    # test the generated results of the evaluation run
    with open(json_payload["eval_results_path"], encoding="utf-8") as file:
        eval_results_list = json.load(file)

    with open(json_payload["eval_params_path"], encoding="utf-8") as file:
        eval_params_list = json.load(file)

    eval_results = eval_results_list[0]
    eval_params = eval_params_list[0]

    assert (
        eval_params["similarity_method"]
        == eval_results["vectorstore_params"]["metadata"]["hnsw:space"]
    ), "Retriever similarity methods should match."

    assert (
        eval_params["search_type"] == eval_results["retriever_params"]["search_type"]
    ), "Retriever search types should match."

    assert (
        eval_params["num_retrieved_docs"]
        == eval_results["retriever_params"]["search_kwargs"]["k"]
    ), "Number of retrieved docs should match."

    assert (
        eval_params["embedding_model"] == eval_results["retriever_params"]["tags"][1]
    ), "Embedding models should match."


@pytest.fixture(scope="function", autouse=True)
def cleanup_output_files():
    # Setup: Anything before the yield is the setup. You can leave it empty if there's no setup.

    yield  # This will allow the test to run.

    # Teardown: Anything after the yield is the teardown.
    for file in glob.glob("./resources/output_*"):
        try:
            os.remove(file)
        except Exception as e:
            logger.error(f"Error deleting {file}. Error: {e}")
