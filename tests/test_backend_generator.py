import requests
import pytest
import json
import chromadb
import logging

from tests.utils import (
    RAGFLOW_HOST,
    RAGFLOW_PORT,
    CHROMADB_HOST,
    CHROMADB_PORT,
    LABEL_DATASET_ENDPOINT,
    DOCUMENT_STORE_PATH,
    LABEL_DATASET_PATH,
    fetch_data,
    first_user_id,
    second_user_id,
    third_user_id_with_invalid_uuid,
    user_id_without_upsert,
)

logger = logging.getLogger(__name__)


def test_generator_without_upsert(user_id_without_upsert):
    """Test QA generator without upserting to ChromaDB. Since we use a fake LLM provided by LangChain for testing purposes to mock a real one, we are not able to generate real QA pairs and the generated label_dataset.json file is empty."""

    json_payload = {
        "document_store_path": DOCUMENT_STORE_PATH,
        "label_dataset_path": "./resources/output_label_dataset_without_upsert.json",
        "label_dataset_gen_params_path": "./resources/input_label_dataset_gen_params_without_upsert.json",
        "user_id": user_id_without_upsert,
        "api_keys": {},
    }

    response = fetch_data(
        method="post",
        host=RAGFLOW_HOST,
        port=RAGFLOW_PORT,
        endpoint=LABEL_DATASET_ENDPOINT,
        payload=json_payload,
    )

    assert (
        response.status_code == 200
    ), f"The response from the {LABEL_DATASET_ENDPOINT} endpoint should be ok."

    with open(json_payload["label_dataset_path"], encoding="utf-8") as file:
        data = json.load(file)

    assert (
        not data
    ), f"Expected no content in {json_payload['label_dataset_path']}, but found it not empty"


@pytest.mark.parametrize(
    "user_id_fixture, expected_num_collections, expected_error",
    [
        ("first_user_id", 1, None),
        ("second_user_id", 2, None),
        ("third_user_id_with_invalid_uuid", -1, requests.exceptions.HTTPError),
    ],
)
def test_generator_with_upsert(
    user_id_fixture, expected_num_collections, expected_error, request
):
    """Helper function to run generator with upserting twice to test upsert functionality"""

    def get_fixture_value(fixture_name):
        """Get the value associated with a fixture name."""
        return request.getfixturevalue(fixture_name)

    user_id = get_fixture_value(user_id_fixture)

    json_payload = {
        "document_store_path": DOCUMENT_STORE_PATH,
        "label_dataset_path": "./resources/output_label_dataset_with_upsert.json",
        "label_dataset_gen_params_path": "./resources/input_label_dataset_gen_params_with_upsert.json",
        "user_id": user_id,
        "api_keys": {},
    }

    # case when a invalid UUID gets passed to endpoints
    if expected_error:
        with pytest.raises(
            expected_error,
            match=f"Unprocessable Entity for url: http://ragflow-test:8080{LABEL_DATASET_ENDPOINT}",
        ):
            # Assuming the error will arise here when pydantic throws an exception because user id is not a valid UUID
            response = fetch_data(
                method="post",
                host=RAGFLOW_HOST,
                port=RAGFLOW_PORT,
                endpoint=LABEL_DATASET_ENDPOINT,
                payload=json_payload,
            )

        return

    # case for valid payload data
    response = fetch_data(
        method="post",
        host=RAGFLOW_HOST,
        port=RAGFLOW_PORT,
        endpoint=LABEL_DATASET_ENDPOINT,
        payload=json_payload,
    )
    assert (
        response.status_code == 200
    ), "The response from the {LABEL_DATASET_ENDPOINT} endpoint should be ok."

    with open(json_payload["label_dataset_path"], encoding="utf-8") as file:
        data = json.load(file)

    assert (
        data
    ), f"Expected content in {json_payload['label_dataset_path']}, but found it empty or null."
    # connect to chromadb
    client = chromadb.HttpClient(
        host=CHROMADB_HOST,
        port=CHROMADB_PORT,
    )

    collections_list = client.list_collections()

    assert (
        len(collections_list) == expected_num_collections
    ), f"Expected {expected_num_collections} collections in ChromaDB but found {len(collections_list)}."

    # get most recently created collection
    sorted_collections = sorted(
        collections_list, key=lambda col: col.metadata["timestamp"], reverse=True
    )

    collection = sorted_collections[0]

    assert (
        collection.name == f"userid_{user_id[:8]}_qaid_0_TestDummyEmbedding"
    ), "Name of collection should contain first 8 chars of user id and embedding model name."

    assert (
        collection.metadata["custom_id"]
        == f"userid_{user_id}_qaid_0_TestDummyEmbedding"
    ), "The metadata dict of collection should also contain the entire user id"
    embeddings = collection.get(include=["embeddings"])["embeddings"]

    assert len(embeddings) == len(data)

    assert len(embeddings[0]) == 2
